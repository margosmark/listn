#!/usr/bin/env bash
# One-command real-hardware verification for listn's Linux two-track capture.
#
# Run this ON the target machine (Mark's CachyOS/KDE laptop) from a checkout,
# inside the desktop session:
#
#     LISTN_CMD="uv run listn" scripts/linux_hardware_check.sh
#
# It walks the real PipeWire session (no synthetic daemon, no fake devices)
# through five phases and ends with a fenced report block ready to paste
# back into the card:
#
#   1. preflight    tools on PATH, session daemon reachable via
#                   XDG_RUNTIME_DIR, `listn` importable; /dev/snd is
#                   reported whether present or absent, never assumed
#   2. discovery    `listn record --list`, plus every graph node labelled
#                   with its media class straight from pw-dump
#   3. capture      4-6 s two-track run through the app's own CLI, with
#                   per-track duration/frames/peak/RMS/verdict — success
#                   requires measured signal, silence is never success
#   4. pause/resume scripted pause mid-recording through the local API:
#                   observed state change, a real silent window in the
#                   stored tracks and a [paused] transcript marker
#   5. hot-plug     interactive: unplug the mic mid-recording and expect
#                   `mic: failed … target node disappeared`; replug and
#                   confirm a fresh capture works again
#
# Interactive phases prompt and never guess; with no human at the keyboard
# (--non-interactive or non-tty stdin) they are SKIPPED and the report says
# so. The script degrades honestly: a missing microphone is a finding
# (exit 1), not a crash.
#
# Everything runs against an isolated temporary data directory — your real
# listn library is never read or written — and with model downloads and
# telemetry forced off (HF_HUB_OFFLINE etc.); the engine simply falls back
# to recording-only mode when no model is cached. Only daemons this script
# starts are killed on exit; the session PipeWire is never touched.
#
# Usage:
#   scripts/linux_hardware_check.sh [--duration SEC] [--out DIR] [--keep]
#                                   [--list] [--json] [--non-interactive]
#                                   [--system-target NODE] [--mic-target NODE]
#                                   [--port N]
#   LISTN_CMD="uv run listn" scripts/linux_hardware_check.sh    # venv/uv app
#
# Knobs: --duration (default 5, capture length in s), --out (copy reports
# and logs there; WAVs stay in the temp dir, keep it with --keep), --list
# (preflight + discovery only), --json (machine-readable result on stdout,
# progress on stderr), --non-interactive (skip the prompted phases),
# --system-target/--mic-target (override the session defaults), --port
# (web-UI port for the pause phase; default: first free in 8799-8808).
#
# Exit codes: 0 = every executed check passed with measured signal
#             1 = the script ran but at least one check failed or was
#                 refused (a skipped interactive check is reported, not
#                 counted as a pass)
#             3 = prerequisite missing (tools, session daemon, listn)
set -u

# fd 3 keeps the real stdout: in --json mode fd 1 is redirected to stderr so
# the final JSON document (printed to fd 3) is the only thing on stdout.
exec 3>&1

DURATION=5
OUT_DIR=""
KEEP=0
LIST_ONLY=0
JSON=0
NONINTERACTIVE=0
SYS_TARGET_ARG=""
MIC_TARGET_ARG=""
PORT=""
LISTN_CMD="${LISTN_CMD:-python3 -m listn}"
LISTN_ARGV=()

while [ $# -gt 0 ]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --system-target) SYS_TARGET_ARG="$2"; shift 2 ;;
    --mic-target)    MIC_TARGET_ARG="$2"; shift 2 ;;
    --keep)     KEEP=1; shift ;;
    --list)     LIST_ONLY=1; shift ;;
    --json)     JSON=1; shift ;;
    --non-interactive) NONINTERACTIVE=1; shift ;;
    -h|--help)  sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "hwcheck: unknown argument '$1' (see --help)" >&2; exit 3 ;;
  esac
done

# LISTN_CMD is a convenience command string, not shell code: split it as argv
# without evaluating substitutions, redirects, or other shell syntax.
# Bash mapfile cannot represent NUL; use a delimiter-safe JSON-free parser.
LISTN_ARGV=()
while IFS= read -r -d '' arg; do LISTN_ARGV+=("$arg"); done < <("${PYTHON:-python3}" -c 'import shlex,sys; sys.stdout.buffer.write(b"\0".join(x.encode() for x in shlex.split(sys.argv[1]))+b"\0")' "$LISTN_CMD" 2>/dev/null)
[ "${#LISTN_ARGV[@]}" -gt 0 ] || { echo "hwcheck: invalid LISTN_CMD" >&2; exit 3; }

case "$DURATION" in
  ''|*[!0-9]*) echo "hwcheck: --duration must be an integer from 1 to 3600" >&2; exit 3 ;;
esac
[ "$DURATION" -ge 1 ] && [ "$DURATION" -le 3600 ] || { echo "hwcheck: --duration must be an integer from 1 to 3600" >&2; exit 3; }
if [ -n "$PORT" ]; then
  case "$PORT" in ''|*[!0-9]*) echo "hwcheck: --port must be an integer from 1 to 65535" >&2; exit 3 ;; esac
  [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { echo "hwcheck: --port must be an integer from 1 to 65535" >&2; exit 3; }
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STARTED="$(date '+%Y-%m-%d %H:%M:%S %Z')"
PASS=0; FAIL=0; SKIP=0

die() {
  echo "hwcheck: $*" >&2
  if [ "$JSON" = 1 ] && command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys; print(json.dumps({"tool":"listn-linux-hardware-check","error":sys.argv[1],"exit_code":3}))' "$*" >&3 \
      || true
  fi
  exit 3
}

# In --json mode stdout carries only the final JSON document; everything the
# phases print goes to stderr.
[ "$JSON" = 1 ] && exec 1>&2
[ -t 0 ] || NONINTERACTIVE=1

say()  { printf '%s\n' "$*"; }
head_(){ say ""; say "===== $* ====="; }
prompt() {  # prompt "question" "default"
  if [ "$NONINTERACTIVE" = 1 ]; then
    REPLY="${2:-}"
    say "  $1 [non-interactive: assuming '$REPLY']"
    return
  fi
  printf '%s ' "$1"
  read -r REPLY || REPLY="${2:-}"
  REPLY="${REPLY:-${2:-}}"
  say "$REPLY"
}
verdict() {  # verdict PASS|FAIL|SKIP "phase" "detail"
  case "$1" in
    PASS) PASS=$((PASS+1)); say "  [PASS] $2 — $3" ;;
    FAIL) FAIL=$((FAIL+1)); say "  [FAIL] $2 — $3" ;;
    SKIP) SKIP=$((SKIP+1)); say "  [SKIP] $2 — $3" ;;
  esac
  if [ -n "${RESULTS:-}" ]; then
    "$LISTN_PY" -c 'import json,sys; print(json.dumps({"phase":sys.argv[2],"status":sys.argv[1].lower(),"detail":sys.argv[3]}))' \
      "$1" "$2" "$3" >> "$RESULTS" || true
  fi
}

# ---- preflight -------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "this check verifies the Linux capture path; run it on the Linux machine"
for tool in pipewire wireplumber pw-dump pw-record python3; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool '$tool' not found on PATH (install PipeWire)"
done
# The interpreter must be the one LISTN_CMD actually runs: with
# LISTN_CMD=".venv/bin/python -m listn" listn and numpy live in the venv and
# the system python3 sees neither (same rule as linux_capture_rig.sh).
LISTN_PY="${LISTN_CMD%% *}"
case "$LISTN_PY" in
  *python*) ;;
  *) LISTN_PY="python3" ;;   # e.g. LISTN_CMD="uv run listn" → no python prefix
esac
( cd "$REPO_ROOT" && "$LISTN_PY" -c 'import listn' ) >/dev/null 2>&1 \
  || die "'$LISTN_PY' cannot import listn — run from a checkout with dependencies installed (e.g. 'uv sync --group dev', then LISTN_CMD='uv run listn')"
"$LISTN_PY" -c 'import numpy' >/dev/null 2>&1 \
  || die "numpy is required (reports are measured from the WAVs), but '$LISTN_PY' cannot import it — install it for that interpreter (e.g. 'uv sync --group dev')"
[ -n "${XDG_RUNTIME_DIR:-}" ] || die "XDG_RUNTIME_DIR is not set — run this inside the desktop session (the session bus is required)"
[ -S "$XDG_RUNTIME_DIR/pipewire-0" ] || die "no PipeWire session socket at \$XDG_RUNTIME_DIR/pipewire-0 — the session audio daemon is not running"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/listn-hwcheck.XXXXXX")" || die "mktemp failed"
chmod 700 "$TMP"
RESULTS="$TMP/results.jsonl"
: > "$RESULTS"
SERVER_PID=""
REC_PID=""
cleanup() {
  [ -n "$REC_PID" ] && { kill "$REC_PID" 2>/dev/null; wait "$REC_PID" 2>/dev/null; }
  if [ -n "$SERVER_PID" ]; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  if [ -n "$OUT_DIR" ] && [ -d "$TMP" ]; then
    mkdir -p "$OUT_DIR" 2>/dev/null \
      && cp "$TMP"/*.json "$TMP"/*.log "$TMP"/results.jsonl "$OUT_DIR"/ 2>/dev/null \
      && say "hwcheck: reports and logs copied to $OUT_DIR (WAVs are not copied; use --keep)"
  fi
  if [ "$KEEP" = 1 ]; then
    say "hwcheck: kept $TMP"
  elif [ -d "$TMP" ]; then
    rm -rf "$TMP"
  fi
}
trap cleanup EXIT

# Isolated data directory + no downloads, no telemetry: the engine may fall
# back to recording-only mode; that is a finding in the report, never a
# reason to touch the network.
export LISTN_DATA_DIR="$TMP/data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true PYANNOTE_METRICS_ENABLED=0
mkdir -p "$LISTN_DATA_DIR"

# ---- helpers (written into the private temp dir, removed on exit) ----------
cat > "$TMP/nodes.py" <<'PYEOF'
"""Label every pw-dump node with its media class (stdin: pw-dump JSON)."""
import json, sys
try:
    objs = json.load(sys.stdin)
except Exception:
    sys.exit(0)
rows = []
for o in objs:
    if o.get("type") != "PipeWire:Interface:Node":
        continue
    p = (o.get("info", {}).get("props", {}) or {})
    name = p.get("node.name")
    if not name:
        continue
    rows.append((p.get("media.class") or "?", name,
                 p.get("node.description") or "", o.get("id")))
for media_class, name, desc, oid in sorted(rows):
    tail = f" — {desc}" if desc else ""
    print(f"{name}  [{media_class}]  (id {oid}){tail}")
PYEOF

cat > "$TMP/analyze_capture.py" <<'PYEOF'
"""Report a `listn record --json` capture honestly.

usage: analyze_capture.py REPORT_JSON RC MODE [TARGETS_ENV]

Prints per-track duration/frames/peak/RMS/verdict. In `capture` mode writes
the target node names to TARGETS_ENV for later phases and exits 0 only when
the recorder reported ok (both tracks stopped clean, mic not silent).
In `expectfail` mode exits 0 only when the mic track failed with the
node-disappeared contract, 1 when it kept recording (exit code not a claim
of failure), 2 when it failed some other way.
"""
import json, re, shlex, sys

path, rc, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out_env = sys.argv[4] if len(sys.argv) > 4 else ""

try:
    with open(path) as f:
        report = json.load(f)
except Exception as exc:
    print(f"no capture report JSON ({exc})")
    sys.exit(1)

tracks = report.get("tracks") or {}
for name in ("system", "mic"):
    t = tracks.get(name) or {}
    err = f" error: {t['error']}" if t.get("error") else ""
    peak = t.get("peak_db")
    rms = t.get("rms_db")
    print(f"{name}: state={t.get('state')} {t.get('duration', 0):.3f} s "
          f"{t.get('frames', 0)} frames "
          f"peak={peak if peak is not None else '-inf'} dBFS "
          f"rms={rms if rms is not None else '-inf'} dBFS "
          f"verdict={t.get('verdict') or 'none'}{err}")

if out_env and out_env != "/dev/null":
    targets = report.get("targets") or {}
    with open(out_env, "w") as f:
        f.write("SYS_TARGET=%s\n" % shlex.quote((targets.get("system", {}) or {}).get("name", "")))
        f.write("MIC_TARGET=%s\n" % shlex.quote((targets.get("mic", {}) or {}).get("name", "")))

if mode == "capture":
    ok = bool(report.get("ok")) and int(rc) == 0
    print(f"report.ok={report.get('ok')} recorder rc={rc}")
    sys.exit(0 if ok else 1)

# expectfail: the honest-failure contract for a vanished mic node.
mic = tracks.get("mic") or {}
err = mic.get("error") or ""
if mic.get("state") == "failed" and re.search(r"target node .* disappeared", err):
    print(f"mic failed exactly as expected: {err}")
    sys.exit(0)
if mic.get("state") in ("stopped", "recording"):
    print("mic track was still recording when the run ended — the unplug was not observed")
    sys.exit(1)
print(f"mic track failed, but not with the node-disappeared contract: {err or mic.get('state')}")
sys.exit(2)
PYEOF

cat > "$TMP/api.py" <<'PYEOF'
"""Call the local listn API (loopback only). usage: api.py METHOD PATH [JSON_BODY]

Prints "<status>\\n<body>"; status 0 means the request never reached the
server. Reads HWCHECK_PORT from the environment.
"""
import json, os, sys, urllib.error, urllib.request

port = os.environ["HWCHECK_PORT"]
method, path = sys.argv[1], sys.argv[2]
body = sys.argv[3].encode() if len(sys.argv) > 3 else None
req = urllib.request.Request(
    f"http://127.0.0.1:{port}{path}", data=body, method=method,
    headers={"Content-Type": "application/json"} if body else {})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(resp.status)
        print(resp.read().decode())
except urllib.error.HTTPError as exc:
    print(exc.code)
    print(exc.read().decode())
except Exception as exc:
    print(0)
    print(str(exc))
PYEOF

cat > "$TMP/pause_phase.py" <<'PYEOF'
"""Scripted pause/resume mid-recording through the app's own API.

usage: pause_phase.py SYSTEM_TARGET MIC_TARGET

Starts a pipewire live session on the loopback server (HWCHECK_PORT),
records 3 s, pauses for 3 s, resumes for 2 s, stops, then verifies the
evidence in the stored session: a >=2 s silent window in each engine WAV
(pause writes zeros) and a [paused] marker in the transcript. Exits 0 only
when every piece of evidence is present — the recorder's own per-track
report must also be ok.
"""
import json, os, sys, time, urllib.error, urllib.request

port = os.environ["HWCHECK_PORT"]
system_target, mic_target = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{base}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, (json.loads(raw) if raw else None)
        except Exception:
            return exc.code, raw
    except Exception as exc:
        return 0, str(exc)


status, body = call("POST", "/api/live/start", {
    "capture": "pipewire", "system_target": system_target,
    "mic_target": mic_target, "title": "hardware check pause/resume"})
print(f"POST /api/live/start -> HTTP {status}")
if status != 200:
    print(f"refused: {json.dumps(body)[:400] if not isinstance(body, str) else body[:400]}")
    sys.exit(1)

time.sleep(3)
_, live = call("GET", "/api/live")
frames = lambda d: {k: (d or {}).get(k, {}).get("frames", 0)
                    for k in ("system", "mic")}
before = frames(((live or {}).get("pipewire") or {}).get("tracks"))
print(f"recording: system {before['system']} frames, mic {before['mic']} frames")

status, _ = call("POST", "/api/live/pause")
print(f"POST /api/live/pause -> HTTP {status}")
pause_seen = status == 200
time.sleep(3)
_, live = call("GET", "/api/live")
live_state = ((live or {}).get("live") or {})
paused_flag = bool(live_state.get("paused"))
during = frames(((live or {}).get("pipewire") or {}).get("tracks"))
print(f"during pause: engine state={live_state.get('state')} paused={paused_flag}; "
      f"system {during['system']} frames, mic {during['mic']} frames "
      f"(the recorder keeps running; the session drops audio)")

status, _ = call("POST", "/api/live/resume")
print(f"POST /api/live/resume -> HTTP {status}")
resume_seen = status == 200
time.sleep(2)

status, stop = call("POST", "/api/live/stop")
print(f"POST /api/live/stop -> HTTP {status}")
session_id = (stop or {}).get("id") if isinstance(stop, dict) else None
print(f"session id: {session_id or '<none>'}")
pw = (stop or {}).get("pipewire") or {} if isinstance(stop, dict) else {}
for name in ("system", "mic"):
    t = (pw.get("tracks") or {}).get(name) or {}
    peak = t.get("peak_db")
    rms = t.get("rms_db")
    err = f" error: {t['error']}" if t.get("error") else ""
    print(f"recorder {name}: state={t.get('state')} {t.get('duration', 0):.3f} s "
          f"{t.get('frames', 0)} frames peak={peak if peak is not None else '-inf'} dBFS "
          f"rms={rms if rms is not None else '-inf'} dBFS "
          f"verdict={t.get('verdict') or 'none'}{err}")
recorder_ok = bool(pw.get("ok"))

time.sleep(2)  # let the engine finalize its WAVs and transcript

if not session_id:
    print("no session was stored — nothing to measure")
    sys.exit(1)

import wave

import numpy as np

session_dir = os.path.join(os.environ["LISTN_DATA_DIR"], "sessions", session_id)
markers = 0
transcript = os.path.join(session_dir, "transcript.json")
if os.path.exists(transcript):
    try:
        with open(transcript) as f:
            segments = json.load(f)
        markers = sum(1 for s in segments if s.get("text") == "[paused]")
    except Exception as exc:
        print(f"transcript unreadable: {exc}")
print(f"[paused] transcript markers: {markers}")

ok = markers >= 1 and pause_seen and resume_seen and paused_flag and recorder_ok
measured = 0
for name in ("audio-system.wav", "audio-mic.wav"):
    path = os.path.join(session_dir, name)
    if not os.path.exists(path):
        print(f"{name}: not written")
        ok = False
        continue
    try:
        w = wave.open(path, "rb")
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int32)
    except Exception as exc:
        print(f"{name}: unreadable ({exc})")
        ok = False
        continue
    if data.size == 0:
        print(f"{name}: empty")
        ok = False
        continue
    quiet = np.abs(data) <= 32  # below -60 dBFS
    padded = np.concatenate(([0], quiet.view(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    longest = float((ends - starts).max() / rate) if starts.size else 0.0
    print(f"{name}: {data.size / rate:.1f} s stored, longest silent window {longest:.1f} s")
    measured += 1
    if longest < 2.0:
        ok = False

if measured == 0:
    print("no engine session audio to measure — pause wrote nothing")
    ok = False
sys.exit(0 if ok else 1)
PYEOF

cat > "$TMP/report.py" <<'PYEOF'
"""Assemble the final report block (and the --json document) from results.jsonl.

Reads PASS/FAIL/SKIP counts from the environment, plus the host facts, and
prints a fenced block that can be pasted straight into the card. In --json
mode (HWCHECK_JSON=1) prints the machine-readable document to stdout
instead — the bash wrapper redirects everything else to stderr there.
"""
import json, os, sys

results = []
path = os.environ.get("RESULTS", "")
if path and os.path.exists(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

passed = sum(1 for r in results if r["status"] == "pass")
failed = sum(1 for r in results if r["status"] == "fail")
skipped = sum(1 for r in results if r["status"] == "skip")
exit_code = 1 if failed else 0
result = "PROBLEMS FOUND" if failed else (
    "ALL CHECKS PASSED" if not skipped else "ALL EXECUTED CHECKS PASSED (some were skipped)")

host = {
    "os": os.environ.get("HOST_OS", ""),
    "kernel": os.environ.get("HOST_KERNEL", ""),
    "session": os.environ.get("HOST_SESSION", ""),
    "pipewire": os.environ.get("PW_VERSION", ""),
    "wireplumber": os.environ.get("WP_VERSION", ""),
    "dev_snd": os.environ.get("SND_STATE", ""),
    "repo": os.environ.get("REPO_ROOT", ""),
    "head": os.environ.get("REPO_HEAD", ""),
    "listn_cmd": os.environ.get("LISTN_CMD", ""),
}

if os.environ.get("HWCHECK_JSON") == "1":
    print(json.dumps({
        "tool": "listn-linux-hardware-check",
        "started": os.environ.get("STARTED", ""),
        "host": host,
        "checks": results,
        "counts": {"pass": passed, "fail": failed, "skip": skipped},
        "result": result,
        "exit_code": exit_code,
    }, indent=2))
    sys.exit(0)

bar = "=" * 78
print("```text")
print(bar)
print(f"listn Linux hardware check — {os.environ.get('STARTED', '')}")
print(f"host:     {host['os']} · kernel {host['kernel']}")
print(f"session:  {host['session']}")
print(f"audio:    {host['pipewire']} · {host['wireplumber']}")
print(f"dev/snd:  {host['dev_snd']}")
print(f"repo:     {host['head']} @ {host['repo']}")
print(f"command:  LISTN_CMD='{host['listn_cmd']}' scripts/linux_hardware_check.sh")
print(bar)
for r in results:
    print(f"[{r['status'].upper():4}] {r['phase']}: {r['detail']}")
print(bar)
print(f"result: {result} (pass={passed} fail={failed} skip={skipped}) — exit {exit_code}")
print(bar)
print("```")
sys.exit(0)
PYEOF

# ---- host facts (for the report block) -------------------------------------
HOST_OS="$(grep ^PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2)"
HOST_KERNEL="$(uname -r)"
HOST_SESSION="type=${XDG_SESSION_TYPE:-?} desktop=${XDG_CURRENT_DESKTOP:-?}"
PW_VERSION="$(pipewire --version 2>/dev/null | grep -m1 -o 'libpipewire [0-9][^ ]*' || true)"
WP_VERSION="$(wireplumber --version 2>/dev/null | grep -m1 -o 'libwireplumber [0-9][^ ]*' || true)"
PW_VERSION="${PW_VERSION:-pipewire (version unknown)}"
WP_VERSION="${WP_VERSION:-wireplumber (version unknown)}"
if [ -d /dev/snd ]; then
  SND_STATE="present ($(ls /dev/snd 2>/dev/null | tr '\n' ' ' | sed 's/ $//'))"
else
  SND_STATE="ABSENT — no sound devices visible; capture below is the real test"
fi
REPO_HEAD="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
export STARTED HOST_OS HOST_KERNEL HOST_SESSION PW_VERSION WP_VERSION SND_STATE \
  REPO_ROOT REPO_HEAD LISTN_CMD RESULTS
BUS_STATE="unreachable"
[ -S "${XDG_RUNTIME_DIR:-/nonexistent}/bus" ] && BUS_STATE="session bus socket present"
[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ] && BUS_STATE="session bus address set"

say "listn Linux hardware check — $STARTED"
say "repo: $REPO_ROOT @ $REPO_HEAD · LISTN_CMD='$LISTN_CMD'"

# ---- phase 1: preflight ----------------------------------------------------
head_ "1/5 preflight"
say "  pipewire:   ${PW_VERSION:-unknown}"
say "  wireplumber:${WP_VERSION:- unknown}"
say "  /dev/snd:   $SND_STATE"
say "  session:    $HOST_SESSION · $BUS_STATE"
say "  daemon:     $XDG_RUNTIME_DIR/pipewire-0 reachable"
verdict PASS preflight "pipewire/wireplumber/pw-dump/pw-record on PATH; session daemon reachable; listn importable via '$LISTN_PY'"


# ---- phase 2: discovery ----------------------------------------------------
head_ "2/5 discovery — the real session graph"
( cd "$REPO_ROOT" && "${LISTN_ARGV[@]}" record --list ) 2>"$TMP/discovery.log"
DISC_RC=$?
[ -s "$TMP/discovery.log" ] && sed 's/^/  /' "$TMP/discovery.log"
# Independent view: label every node with its media class straight from the
# daemon, so the report carries the graph's own words, not only the CLI's.
pw-dump > "$TMP/graph.json" 2>/dev/null
"$LISTN_PY" "$TMP/nodes.py" < "$TMP/graph.json" 2>/dev/null | sed 's/^/  /'
COUNTS="$("$LISTN_PY" - "$TMP/graph.json" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        objs = json.load(f)
except Exception:
    print("0 0"); sys.exit(0)
sinks = sources = 0
for o in objs:
    if o.get("type") != "PipeWire:Interface:Node":
        continue
    cls = (o.get("info", {}).get("props", {}) or {}).get("media.class")
    sinks += cls == "Audio/Sink"
    sources += cls == "Audio/Source"
print(sinks, sources)
PYEOF
)"
SINKS="${COUNTS%% *}"; SOURCES="${COUNTS##* }"
SINKS=${SINKS:-0}; SOURCES=${SOURCES:-0}
if [ "$DISC_RC" = 0 ]; then
  EXTRA=""
  [ "$SOURCES" = 0 ] && EXTRA=" — NO Audio/Source visible: no microphone, capture will fail"
  verdict PASS discovery "$SINKS sink(s), $SOURCES source(s) discovered$EXTRA"
else
  verdict FAIL discovery "`record --list` failed (rc=$DISC_RC) — see output above"
fi

# Resolve the target pair the app itself would use (labelled default /
# fallback); later phases need explicit names for the local API.
# Resolve targets with the app's own select_targets: it re-checks media
# classes, so a session default that names a sink (observed with
# WirePlumber on machines without mics) is refused and a real source is
# chosen — exactly what `listn record` would capture.
"$LISTN_PY" - "$TMP/devices.env" <<'PYEOF' 2>/dev/null || true
import sys
from listn.pipewire import CaptureError, discover, select_targets
import shlex
try:
    targets = select_targets(discover(), None, None)
except CaptureError:
    targets = None
with open(sys.argv[1], "w") as f:
    f.write("SYS_TARGET=%s\n" % shlex.quote(targets.system.node.name if targets else ""))
    f.write("MIC_TARGET=%s\n" % shlex.quote(targets.mic.node.name if targets else ""))
PYEOF
# shellcheck disable=SC1090
[ -f "$TMP/devices.env" ] && . "$TMP/devices.env"
SYS_TARGET="${SYS_TARGET_ARG:-${SYS_TARGET:-}}"
MIC_TARGET="${MIC_TARGET_ARG:-${MIC_TARGET:-}}"
say "  targets: system=${SYS_TARGET:-<none>}  mic=${MIC_TARGET:-<none>}"

if [ "$LIST_ONLY" = 1 ]; then
  head_ "report"
  export HWCHECK_JSON="$JSON"
  "$LISTN_PY" "$TMP/report.py" >&3
  [ "$FAIL" -gt 0 ] && exit 1
  exit 0
fi

# ---- phase 3: capture ------------------------------------------------------
head_ "3/5 two-track capture (${DURATION} s, the app's own CLI)"
if [ "$DISC_RC" != 0 ]; then
  verdict SKIP capture "discovery failed — nothing to record against"
else
  prompt "Play some audio (a video, music) and be ready to speak for ${DURATION} s — press Enter to start..." ""
  mkdir -p "$TMP/capture"
  TARGET_ARGS=()
  [ -n "$SYS_TARGET" ] && TARGET_ARGS+=(--system-target "$SYS_TARGET")
  [ -n "$MIC_TARGET" ] && TARGET_ARGS+=(--mic-target "$MIC_TARGET")
  ( cd "$REPO_ROOT" && "${LISTN_ARGV[@]}" record --duration "$DURATION" \
      "${TARGET_ARGS[@]+"${TARGET_ARGS[@]}"}" \
      --out "$TMP/capture" --json ) >"$TMP/capture.json" 2>"$TMP/capture.log"
  CAP_RC=$?
  sed 's/^/  /' "$TMP/capture.log"
  "$LISTN_PY" "$TMP/analyze_capture.py" "$TMP/capture.json" "$CAP_RC" capture "$TMP/targets.env" \
    > "$TMP/capture.report" 2>&1
  ANALYZE_RC=$?
  sed 's/^/  /' "$TMP/capture.report"
  if [ "$ANALYZE_RC" = 0 ]; then
    verdict PASS capture "recorder ok: mic carries measured signal (${DURATION} s run; a silent system track alone is allowed - per-track numbers above)"
  else
    verdict FAIL capture "the recorder refused or the mic lacks real audio (rc=$CAP_RC) - were you speaking? (per-track numbers above)"
  fi
  # The nodes that actually worked are the ones later phases should use.
  [ -f "$TMP/targets.env" ] && . "$TMP/targets.env"
  SYS_TARGET="${SYS_TARGET_ARG:-${SYS_TARGET:-}}"
  MIC_TARGET="${MIC_TARGET_ARG:-${MIC_TARGET:-}}"
fi

# ---- phase 4: pause/resume -------------------------------------------------
head_ "4/5 pause/resume (engine session through the local API)"
if [ "$DISC_RC" != 0 ] || [ -z "$SYS_TARGET" ] || [ -z "$MIC_TARGET" ]; then
  verdict SKIP pause/resume "no usable system+mic target pair discovered — cannot start a session"
else
  if [ -z "$PORT" ]; then
    PORT="$("$LISTN_PY" - <<'PYEOF'
import socket
for p in range(8799, 8809):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
PYEOF
)"
  fi
  [ -n "$PORT" ] || verdict SKIP pause/resume "no free port in 8799-8808 for the temporary web UI" || true
  if [ -n "$PORT" ]; then
    say "  keep some audio playing; the session runs ~10 s against an isolated data dir"
    say "  starting the local web UI on 127.0.0.1:$PORT (loopback only)..."
    setsid "${LISTN_ARGV[@]}" ui --no-browser --port "$PORT" >>"$TMP/ui.log" 2>&1 &
    SERVER_PID=$!
    export HWCHECK_PORT="$PORT"
    UP=""
    for _ in $(seq 1 60); do
      st="$("$LISTN_PY" "$TMP/api.py" GET /api/pipewire/devices | head -1)"
      [ "$st" = "200" ] && { UP=1; break; }
      sleep 0.5
    done
    if [ -z "$UP" ]; then
      tail -5 "$TMP/ui.log" 2>/dev/null | sed 's/^/  /'
      verdict FAIL pause/resume "the web UI did not answer on 127.0.0.1:$PORT within 30 s"
    else
      "$LISTN_PY" "$TMP/pause_phase.py" "$SYS_TARGET" "$MIC_TARGET" > "$TMP/pause.report" 2>&1
      PAUSE_RC=$?
      sed 's/^/  /' "$TMP/pause.report"
      if [ "$PAUSE_RC" = 0 ]; then
        verdict PASS pause/resume "pause observed mid-recording, resume returned, stored tracks carry a real silent window and a [paused] marker"
      else
        verdict FAIL pause/resume "pause/resume did not leave the expected evidence — see the measurements above"
      fi
    fi
    if [ -n "$SERVER_PID" ]; then
      kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null
      wait "$SERVER_PID" 2>/dev/null
      SERVER_PID=""
    fi
  fi
fi

# ---- phase 5: hot-plug -----------------------------------------------------
head_ "5/5 hot-plug (unplug/replug the microphone)"
if [ "$NONINTERACTIVE" = 1 ]; then
  verdict SKIP "hot-plug" "non-interactive run — unplug/replug behaviour NOT verified on this machine"
elif [ "$DISC_RC" != 0 ] || [ -z "$MIC_TARGET" ]; then
  verdict SKIP "hot-plug" "no microphone target discovered — nothing to unplug"
else
  prompt "Do you have an external mic (USB or Bluetooth) you can unplug right now? [y/N]" "n"
  if [ "${REPLY,,}" != "y" ] && [ "${REPLY,,}" != "yes" ]; then
    verdict SKIP "hot-plug" "no external mic available to unplug — unplug/replug behaviour NOT verified on this machine"
  else
    prompt "Which node will you unplug? Enter for the mic target in use ($MIC_TARGET):" "$MIC_TARGET"
    UNPLUG_NODE="${REPLY:-$MIC_TARGET}"
    mkdir -p "$TMP/hotplug"
    say "  starting a 15 s capture on '$UNPLUG_NODE'; unplug within the first ~12 s"
    HOT_TARGET_ARGS=()
    [ -n "$SYS_TARGET" ] && HOT_TARGET_ARGS+=(--system-target "$SYS_TARGET")
    ( cd "$REPO_ROOT" && "${LISTN_ARGV[@]}" record --duration 15 \
        --mic-target "$UNPLUG_NODE" "${HOT_TARGET_ARGS[@]}" \
        --out "$TMP/hotplug" --json ) >"$TMP/hotplug.json" 2>"$TMP/hotplug.log" &
    REC_PID=$!
    sleep 3
    prompt ">>> UNPLUG the mic NOW, then press Enter..." ""
    wait "$REC_PID"; HOT_RC=$?
    REC_PID=""
    say "  listn record exit code after unplug: $HOT_RC"
    "$LISTN_PY" "$TMP/analyze_capture.py" "$TMP/hotplug.json" "$HOT_RC" expectfail \
      > "$TMP/hotplug.report" 2>&1
    ANALYZE_RC=$?
    sed 's/^/  /' "$TMP/hotplug.report"
    case "$ANALYZE_RC" in
      0) verdict PASS "hot-plug unplug" "the mic track FAILED honestly with 'target node … disappeared' — no silent re-target to another source" ;;
      1) prompt "The mic track kept recording — did you actually unplug it during the recording? [y/N]" "n"
         if [ "${REPLY,,}" = "y" ] || [ "${REPLY,,}" = "yes" ]; then
           verdict FAIL "hot-plug unplug" "the app kept recording after the unplug — the vanished target was NOT detected"
         else
           verdict SKIP "hot-plug unplug" "unplug not performed during the window — behaviour NOT verified"
         fi ;;
      *) verdict FAIL "hot-plug unplug" "the mic track failed, but not with the node-disappeared contract — see the per-track report above" ;;
    esac
    prompt ">>> REPLUG the mic, wait ~5 s, then press Enter..." ""
    sleep 2
    say "  re-running discovery:"
    ( cd "$REPO_ROOT" && "${LISTN_ARGV[@]}" record --list ) 2>/dev/null | sed 's/^/  /'
    mkdir -p "$TMP/recover"
    prompt "Speak for the next 4 s — press Enter to start the recovery capture..." ""
    ( cd "$REPO_ROOT" && "${LISTN_ARGV[@]}" record --duration 4 \
        --mic-target "$UNPLUG_NODE" "${HOT_TARGET_ARGS[@]}" \
        --out "$TMP/recover" --json ) >"$TMP/recover.json" 2>"$TMP/recover.log"
    REC_RC=$?
    sed 's/^/  /' "$TMP/recover.log"
    "$LISTN_PY" "$TMP/analyze_capture.py" "$TMP/recover.json" "$REC_RC" capture /dev/null \
      > "$TMP/recover.report" 2>&1
    ANALYZE_RC=$?
    sed 's/^/  /' "$TMP/recover.report"
    if [ "$ANALYZE_RC" = 0 ]; then
      verdict PASS "hot-plug replug" "capture recovered after replug — mic records with signal again"
    else
      verdict FAIL "hot-plug replug" "capture did not recover after replug (rc=$REC_RC) — if the node was renamed, re-run with --mic-target <new node>"
    fi
  fi
fi

# ---- report ----------------------------------------------------------------
head_ "report"
export HWCHECK_JSON="$JSON"
"$LISTN_PY" "$TMP/report.py" >&3
[ "$FAIL" -gt 0 ] && exit 1
exit 0
