#!/usr/bin/env bash
# Reproducible two-track PipeWire capture rig (headless Linux).
#
# Stands up an ephemeral, fully private PipeWire + WirePlumber daemon,
# creates a system sink (null adapter) and a microphone source
# (libpipewire-module-pipe-tunnel fed by a paced FIFO writer), injects a
# 440 Hz tone into both, runs the app's own two-track recorder:
#
#     $LISTN_CMD record --system-target listn.rig.system \
#                      --mic-target listn.rig.mic --duration N --json
#
# then prints per-track duration/frames/peak/RMS/verdict plus the dominant
# frequency of each WAV and tears the whole rig down (daemon, writer,
# playback, private runtime dir). Nothing outside the private temp dir is
# touched; no real devices are used.
#
# Usage:
#   scripts/linux_capture_rig.sh [--duration SEC] [--out DIR] [--keep] [--list "nodes"]
#   LISTN_CMD="uv run listn" scripts/linux_capture_rig.sh       # use a venv/uv
#
# Exit codes: 0 = both tracks verdict ok (with signal)
#             1 = capture ran but a verdict is not ok
#             3 = rig infrastructure failed (tools missing, daemon/node did
#                 not come up, recorder refused with rc=3)
set -u

DURATION=4
OUT_DIR=""
KEEP=0
LIST_ONLY=0
LISTN_CMD="${LISTN_CMD:-python3 -m listn}"

while [ $# -gt 0 ]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    --keep)     KEEP=1; shift ;;
    --list)     LIST_ONLY=1; shift ;;
    -h|--help)  sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "rig: unknown argument '$1' (see --help)" >&2; exit 3 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM_NODE=listn.rig.system
MIC_NODE=listn.rig.mic
TONE_HZ=440

die() { echo "rig: $*" >&2; exit 3; }

# ---- preflight -------------------------------------------------------------
for tool in pipewire wireplumber pw-dump pw-record pw-cat python3; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool '$tool' not found on PATH"
done
# The numpy check must use the interpreter LISTN_CMD actually runs, not the
# system python3: with `LISTN_CMD=".venv/bin/python -m listn"` numpy lives in
# the venv and the system interpreter cannot import it.
LISTN_PY="${LISTN_CMD%% *}"
case "$LISTN_PY" in
  *python*) ;;
  *) LISTN_PY="python3" ;;   # e.g. LISTN_CMD="uv run listn" → no python prefix
esac
"$LISTN_PY" -c 'import numpy' >/dev/null 2>&1 \
  || die "numpy is required (the recorder analyzes WAVs with it), but '$LISTN_PY' cannot import it — install it for that interpreter (e.g. 'uv sync --group dev')"

# ---- private rig dir -------------------------------------------------------
RIG="$(mktemp -d "${TMPDIR:-/tmp}/listn-capture-rig.XXXXXX")" || die "mktemp failed"
mkdir -p "$RIG/run" "$RIG/xdg/pipewire/pipewire.conf.d" "$RIG/capture"
chmod 700 "$RIG/run"
PIDS=""
cleanup() {
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
  wait 2>/dev/null
  if [ "$KEEP" = 1 ]; then
    echo "rig: kept $RIG"
  else
    rm -rf "$RIG"
  fi
}
trap cleanup EXIT
export XDG_RUNTIME_DIR="$RIG/run"
export XDG_CONFIG_HOME="$RIG/xdg"

# ---- tones + node config ---------------------------------------------------
# tone.raw: 1 s of 440 Hz s16le mono 48 kHz, looped into the mic FIFO.
# tone.wav: DURATION+3 s stereo WAV, played into the null sink by pw-cat.
"$LISTN_PY" - "$RIG" "$DURATION" "$TONE_HZ" <<'PYEOF'
import math, struct, sys, wave
import numpy as np
rig, dur, hz = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rate = 48000
t = np.arange(rate) / rate
raw = (0.4 * 32767 * np.sin(2 * math.pi * hz * t)).astype("<i2")
with open(f"{rig}/tone.raw", "wb") as f:
    f.write(raw.tobytes())
n = int((dur + 3) * rate)
t = np.arange(n) / rate
stereo = (0.4 * 32767 * np.sin(2 * math.pi * hz * t)).astype("<i2")
w = wave.open(f"{rig}/tone.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
w.writeframes(np.column_stack([stereo, stereo]).tobytes())
w.close()
PYEOF

mkfifo "$RIG/mic.fifo"
cat > "$RIG/xdg/pipewire/pipewire.conf.d/10-listn-rig.conf" <<EOF
context.objects = [
  { factory = adapter
    args = {
      factory.name     = support.null-audio-sink
      node.name        = "$SYSTEM_NODE"
      node.description = "listn rig system output"
      media.class      = "Audio/Sink"
      audio.position   = [ "FL" "FR" ]
    }
  }
]
context.modules = [
  { name = libpipewire-module-pipe-tunnel
    args = {
      tunnel.mode      = "source"
      pipe.filename    = "$RIG/mic.fifo"
      audio.format     = "S16LE"
      audio.rate       = 48000
      audio.channels   = 1
      node.name        = "$MIC_NODE"
      node.description = "listn rig mic tone"
      media.class      = "Audio/Source"
    }
  }
]
EOF

# ---- daemon + FIFO writer --------------------------------------------------
# The writer opens the FIFO first (blocks until the daemon's pipe-tunnel
# module opens the read end) and then paces ~1:1 with the graph. Holding the
# write end open the whole time matters: an empty FIFO reads as EOF and the
# tunnel source goes silent (observed: exactly 1024 zero frames, then stop).
"$LISTN_PY" - "$RIG/mic.fifo" "$RIG/tone.raw" <<'PYEOF' >"$RIG/writer.log" 2>&1 &
import os, sys, time
fifo, tone = sys.argv[1], sys.argv[2]
fd = os.open(fifo, os.O_WRONLY)
data = open(tone, "rb").read()
chunk, pace = 480 * 2, 480 / 48000 * 0.95   # ~10 ms, slightly ahead of realtime
pos = 0
while True:
    end = min(pos + chunk, len(data))
    payload = data[pos:end]
    if end == len(data):
        pos = 0
    else:
        pos = end
    os.write(fd, payload)
    time.sleep(pace)
PYEOF
PIDS="$PIDS $!"

pipewire >"$RIG/pipewire.log" 2>&1 &
PIDS="$PIDS $!"
ok=""
for _ in $(seq 1 100); do
  [ -S "$XDG_RUNTIME_DIR/pipewire-0" ] && { ok=1; break; }
  sleep 0.1
done
[ -n "$ok" ] || { tail -5 "$RIG/pipewire.log" >&2; die "pipewire daemon did not create its socket"; }

wireplumber >"$RIG/wireplumber.log" 2>&1 &
PIDS="$PIDS $!"
sleep 1.5

# ---- wait for both nodes ---------------------------------------------------
nodes_seen=""
ok=""
for _ in $(seq 1 100); do
  nodes_seen="$(pw-dump 2>/dev/null | "$LISTN_PY" -c '
import json, sys
try: objs = json.load(sys.stdin)
except Exception: sys.exit(0)
for o in objs:
    if o.get("type") == "PipeWire:Interface:Node":
        p = o.get("info", {}).get("props", {})
        if p.get("node.name"):
            print(p["node.name"], p.get("media.class") or "-")
')"
  echo "$nodes_seen" | grep -q "^$SYSTEM_NODE " && echo "$nodes_seen" | grep -q "^$MIC_NODE " \
    && { ok=1; break; }
  sleep 0.1
done
if [ -z "$ok" ]; then
  echo "rig: nodes seen: $(echo $nodes_seen)" >&2
  grep -iE '\[E\]|failed' "$RIG/pipewire.log" "$RIG/wireplumber.log" 2>/dev/null | head -5 >&2
  die "rig nodes did not appear (expected $SYSTEM_NODE and $MIC_NODE)"
fi
echo "rig: daemon up ($(pipewire --version 2>/dev/null | head -1)), nodes ready:"
echo "$nodes_seen" | sed 's/^/rig:   /'

if [ "$LIST_ONLY" = 1 ]; then
  ( cd "$REPO_ROOT" && LISTN_DATA_DIR="$RIG/data" $LISTN_CMD record --list ) || exit 3
  exit 0
fi

# ---- tone injection + capture ---------------------------------------------
pw-cat --target "$SYSTEM_NODE" --playback "$RIG/tone.wav" >"$RIG/pw-cat.log" 2>&1 &
PIDS="$PIDS $!"
sleep 0.5

echo "rig: recording $DURATION s with: $LISTN_CMD record --system-target $SYSTEM_NODE --mic-target $MIC_NODE --duration $DURATION"
( cd "$REPO_ROOT" && LISTN_DATA_DIR="$RIG/data" $LISTN_CMD record \
    --system-target "$SYSTEM_NODE" --mic-target "$MIC_NODE" \
    --duration "$DURATION" --out "$RIG/capture" --json \
    >"$RIG/report.json" 2>"$RIG/record.log" )
rc=$?
if [ "$rc" = 3 ]; then
  cat "$RIG/record.log" >&2
  die "the recorder refused to run (rc=3); see log above"
fi
if [ ! -s "$RIG/report.json" ]; then
  cat "$RIG/record.log" >&2
  die "recorder produced no JSON report (rc=$rc)"
fi

# ---- verdict ---------------------------------------------------------------
"$LISTN_PY" - "$RIG" "$rc" "$OUT_DIR" <<'PYEOF'
import json, math, os, shutil, sys
import numpy as np
import wave

rig, rc, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
report = json.load(open(os.path.join(rig, "report.json")))

def dominant(path):
    """Dominant frequency (Hz) of the loudest 0.5 s window, or None."""
    try:
        w = wave.open(path, "rb")
    except Exception:
        return None
    rate = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    if data.size < rate // 2:
        return None
    win = rate // 2
    amps = np.abs(data[: len(data) // win * win].reshape(-1, win))
    seg = data[np.argmax(amps.max(axis=1)) * win:][:win]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return round(float(np.fft.rfftfreq(len(seg), 1 / rate)[spec.argmax()]), 1)

ok = True
for name in ("system", "mic"):
    t = report["tracks"][name]
    dom = dominant(t.get("file") or "")
    dom_s = f" dominant={dom} Hz" if dom is not None else ""
    err = f" error: {t['error']}" if t.get("error") else ""
    print(f"{name}: state={t['state']} {t['duration']:.3f} s {t['frames']} frames "
          f"peak={t['peak_db']} dBFS rms={t['rms_db']} dBFS verdict={t['verdict']}{dom_s}{err}")
    if t["state"] != "stopped" or t["verdict"] != "ok":
        ok = False

if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    for f in ("report.json", "record.log", "capture/system.wav", "capture/mic.wav"):
        src = os.path.join(rig, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(f)))

if ok and report.get("ok") and rc == 0:
    print("rig result: OK — both tracks recorded with real signal")
    sys.exit(0)
print(f"rig result: NOT OK (report.ok={report.get('ok')}, recorder rc={rc})")
sys.exit(1)
PYEOF
exit $?
