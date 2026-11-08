"""The hardware-check script (``scripts/linux_hardware_check.sh``).

The full check needs a real PipeWire session and takes ~30 s, so the default
run only checks the script's syntax, interface and the honesty rules encoded
in its analyzer (silence is never success; a vanished mic node must fail
exactly with the node-disappeared contract). Set ``LISTN_HW_IT=1`` to run the
whole check end to end against a live daemon — that is the one-command gate
``docs/LINUX_CAPTURE.md`` quotes:

    LISTN_HW_IT=1 uv run pytest tests/test_hardware_check.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "linux_hardware_check.sh"
bash = shutil.which("bash") or ""  # tests are skipped when bash is absent

pytestmark = pytest.mark.skipif(not bash, reason="bash not available")


def run_script(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run([bash, str(SCRIPT), *args], capture_output=True,
                          text=True, env=full_env)


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK), "scripts/linux_hardware_check.sh must be executable"


def test_script_syntax():
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


def test_help_documents_exit_codes_and_modes():
    proc = run_script("--help")
    assert proc.returncode == 0, proc.stderr
    assert "Exit codes" in proc.stdout
    assert "--duration" in proc.stdout
    assert "--json" in proc.stdout
    assert "LISTN_CMD" in proc.stdout


def test_rejects_unknown_arguments():
    proc = run_script("--bogus")
    assert proc.returncode == 3
    assert "unknown argument" in proc.stderr


def test_missing_session_runtime_dir_exits_3():
    """No XDG_RUNTIME_DIR means no session bus: a prerequisite, not a crash."""
    proc = run_script("--list", env={"XDG_RUNTIME_DIR": ""})
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "XDG_RUNTIME_DIR" in proc.stderr
    # A set-but-dead runtime dir (no pipewire socket) is also a prerequisite.
    proc = run_script("--list", env={"XDG_RUNTIME_DIR": "/nonexistent-hwcheck-runtime"})
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "pipewire-0" in proc.stderr


def test_missing_runtime_dir_json_mode_is_machine_readable():
    proc = run_script("--list", "--json",
                      env={"XDG_RUNTIME_DIR": "/nonexistent-hwcheck-runtime"})
    assert proc.returncode == 3
    doc = json.loads(proc.stdout)
    assert doc["exit_code"] == 3
    assert "pipewire-0" in doc["error"]


def _analyze(args: list[str]) -> subprocess.CompletedProcess:
    """Run the script's own analyzer, extracted from its heredoc."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="hwcheck-analyze-")
    try:
        text = SCRIPT.read_text()
        start = text.index("cat > \"$TMP/analyze_capture.py\" <<'PYEOF'")
        body_start = text.index("\n", start) + 1
        body_end = text.index("\nPYEOF", body_start)
        Path(tmp, "analyze_capture.py").write_text(text[body_start:body_end] + "\n")
        return subprocess.run(
            [os.environ.get("LISTN_PY", "python3"), str(Path(tmp, "analyze_capture.py")), *args],
            capture_output=True, text=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_report(tmp_path: Path, mic_state: str, mic_error: str | None,
                  verdict: str = "ok", ok: bool = False) -> str:
    report = {
        "ok": ok,
        "tracks": {
            "system": {"state": "stopped", "duration": 4.0, "frames": 64000,
                       "peak_db": -8.0, "rms_db": -11.0, "verdict": "ok", "error": None},
            "mic": {"state": mic_state, "duration": 4.0, "frames": 64000,
                    "peak_db": -14.0, "rms_db": -17.0, "verdict": verdict,
                    "error": mic_error},
        },
        "targets": {"system": {"name": "sys.node"}, "mic": {"name": "mic.node"}},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return str(path)


def test_analyzer_capture_mode_requires_report_ok(tmp_path):
    """A silent mic (report.ok false) must fail the capture check."""
    path = _write_report(tmp_path, "stopped", None, verdict="silent", ok=False)
    proc = _analyze([path, "0", "capture", os.devnull])
    assert proc.returncode == 1, proc.stdout
    assert "verdict=silent" in proc.stdout


def test_analyzer_capture_mode_passes_only_clean_report(tmp_path):
    path = _write_report(tmp_path, "stopped", None, verdict="ok", ok=True)
    proc = _analyze([path, "0", "capture", os.devnull])
    assert proc.returncode == 0, proc.stdout


def test_analyzer_expectfail_requires_node_disappeared(tmp_path):
    """The unplug contract: mic failed because the target node vanished."""
    path = _write_report(
        tmp_path, "failed",
        "target node 'mic.node' disappeared from the PipeWire graph while "
        "recording; stopped this track instead of recording another source")
    proc = _analyze([path, "1", "expectfail"])
    assert proc.returncode == 0, proc.stdout


def test_analyzer_expectfail_flags_surviving_mic(tmp_path):
    """Mic kept recording after the unplug: not the expected failure."""
    path = _write_report(tmp_path, "stopped", None, verdict="ok", ok=True)
    proc = _analyze([path, "0", "expectfail"])
    assert proc.returncode == 1, proc.stdout


def test_analyzer_expectfail_flags_other_failures(tmp_path):
    path = _write_report(tmp_path, "failed", "pw-record exited with code 1")
    proc = _analyze([path, "1", "expectfail"])
    assert proc.returncode == 2, proc.stdout


@pytest.mark.skipif(
    not os.getenv("LISTN_HW_IT"),
    reason="set LISTN_HW_IT=1 to run the full hardware check against a live daemon (~40 s)",
)
def test_hardware_check_end_to_end():
    """The whole check against a live session-style daemon.

    Stands up a private PipeWire + WirePlumber daemon (null sink + tone-fed
    pipe-tunnel mic, like the capture rig), points XDG_RUNTIME_DIR at it and
    runs the script non-interactively with --json. Asserts the honesty
    contract: exit 0 only when the capture carried real signal and
    pause/resume left measurable evidence; stdout is pure JSON.
    """
    pw, wp, pwc = (shutil.which(t) for t in ("pipewire", "wireplumber", "pw-cat"))
    venv_py = REPO / ".venv" / "bin" / "python"
    py = os.environ.get("LISTN_PY") or (
        str(venv_py) if venv_py.exists() else shutil.which("python3") or "")
    if not (pw and wp and pwc and py):
        pytest.skip("pipewire/wireplumber/pw-cat/python not all available")
    # The daemon's Unix socket has a ~108-char path limit and pytest's tmp_path
    # can exceed it, so the rig lives in a short /tmp path of its own.
    rig = Path(tempfile.mkdtemp(prefix="hwit."))
    shutil.rmtree(rig, ignore_errors=True)
    (rig / "run").mkdir(parents=True)
    (rig / "xdg" / "pipewire" / "pipewire.conf.d").mkdir(parents=True)
    (rig / "run").chmod(0o700)
    fifo = rig / "mic.fifo"
    os.mkfifo(fifo)
    dropin = (
        "context.objects = [\n"
        "  { factory = adapter\n"
        "    args = { factory.name = support.null-audio-sink\n"
        "             node.name = \"hwcheck.system\"\n"
        "             node.description = \"hwcheck system output\"\n"
        "             media.class = \"Audio/Sink\"\n"
        "             audio.position = [ \"FL\" \"FR\" ] } }\n"
        "]\n"
        "context.modules = [\n"
        "  { name = libpipewire-module-pipe-tunnel\n"
        "    args = { tunnel.mode = \"source\"\n"
        "             pipe.filename = \"%s\"\n"
        "             audio.format = \"S16LE\"\n"
        "             audio.rate = 48000\n"
        "             audio.channels = 1\n"
        "             node.name = \"hwcheck.mic\"\n"
        "             node.description = \"hwcheck mic tone\"\n"
        "             media.class = \"Audio/Source\" } }\n"
        "]\n" % fifo
    )
    (rig / "xdg" / "pipewire" / "pipewire.conf.d" / "10.conf").write_text(dropin)
    # 1 s of 440 Hz looped into the mic FIFO (held open: an empty FIFO reads
    # as EOF and the tunnel source goes silent — observed rig behaviour).
    tone_raw = rig / "tone.raw"
    subprocess.run(
        [py, "-c",
         "import math,sys,numpy as np;"
         "rate=48000;t=np.arange(rate)/rate;"
         "raw=(0.4*32767*np.sin(2*math.pi*440*t)).astype('<i2');"
         "open(sys.argv[1],'wb').write(raw.tobytes())",
         str(tone_raw)], check=True)
    tone_wav = rig / "tone.wav"
    subprocess.run(
        [py, "-c",
         "import math,sys,wave,numpy as np;"
         "rate=48000;n=30*rate;t=np.arange(n)/rate;"
         "st=(0.4*32767*np.sin(2*math.pi*440*t)).astype('<i2');"
         "w=wave.open(sys.argv[1],'wb');w.setnchannels(2);w.setsampwidth(2);"
         "w.setframerate(rate);w.writeframes(np.column_stack([st,st]).tobytes());"
         "w.close()",
         str(tone_wav)], check=True)
    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(rig / "run")
    env["XDG_CONFIG_HOME"] = str(rig / "xdg")
    procs: list = []

    def spawn(*cmd) -> "subprocess.Popen":
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        procs.append(p)
        return p

    try:
        spawn(py, "-c",
              "import os,sys,time;"
              "fd=os.open(sys.argv[1],os.O_WRONLY);"
              "data=open(sys.argv[2],'rb').read();"
              "chunk,pace=960,480/48000*0.95;pos=0\n"
              "while True:\n"
              " end=min(pos+chunk,len(data));os.write(fd,data[pos:end]);"
              "pos=0 if end==len(data) else end;time.sleep(pace)",
              str(fifo), str(tone_raw))
        spawn(pw)
        for _ in range(100):
            if (rig / "run" / "pipewire-0").exists():
                break
            time.sleep(0.1)
        else:
            pytest.fail("pipewire daemon did not create its socket")
        spawn(wp)
        time.sleep(1.5)
        spawn(pwc, "--target", "hwcheck.system", "--playback", str(tone_wav))
        time.sleep(0.5)
        proc = subprocess.run(
            [bash, str(SCRIPT), "--duration", "4", "--non-interactive", "--json"],
            capture_output=True, text=True,
            env={**env, "LISTN_CMD": f"{py} -m listn", "PYTHONPATH": str(REPO)})
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        doc = json.loads(proc.stdout)  # stdout must be pure JSON
        by_phase = {c["phase"]: c["status"] for c in doc["checks"]}
        assert by_phase["capture"] == "pass", out
        assert by_phase["pause/resume"] == "pass", out
        assert doc["counts"]["fail"] == 0
        assert doc["exit_code"] == 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(rig, ignore_errors=True)


def _heredoc(name):
    text = SCRIPT.read_text()
    marker = f"cat > \"$TMP/{name}\" <<'PYEOF'"
    start = text.index("\n", text.index(marker)) + 1
    return text[start:text.index("\nPYEOF", start)] + "\n"


@pytest.mark.parametrize("suffix", ["", "' space"])
def test_capture_targets_do_not_execute_shell(tmp_path, suffix):
    marker = tmp_path / "pwned"
    path = _write_report(tmp_path, "stopped", None, ok=True)
    report = json.loads(Path(path).read_text())
    name = f"node$(touch {marker}){suffix}"
    report["targets"]["system"]["name"] = name
    Path(path).write_text(json.dumps(report))
    envfile = tmp_path / "targets.env"
    proc = _analyze([path, "0", "capture", str(envfile)])
    assert proc.returncode == 0
    result = subprocess.run([bash, "-c", '. "$1"; printf "%s" "$SYS_TARGET"', "bash", str(envfile)], capture_output=True, text=True)
    assert result.returncode == 0
    assert not marker.exists()
    assert result.stdout == name


@pytest.mark.parametrize("suffix", ["", "' space"])
def test_discovery_target_names_are_shell_escaped(tmp_path, suffix):
    import sys
    script = SCRIPT.read_text()
    start = script.index('import sys\nfrom listn.pipewire import CaptureError, discover, select_targets')
    body = script[start:script.index('\nPYEOF', start)]
    marker = tmp_path / "pwned"
    name = f"node$(touch {marker}){suffix}"
    wrapper = """import sys, types
from types import SimpleNamespace as N
m = types.ModuleType('listn.pipewire')
m.CaptureError = type('CaptureError', (Exception,), {})
m.discover = lambda: None
m.select_targets = lambda *args: N(system=N(node=N(name=sys.argv[2])), mic=N(node=N(name='mic')))
sys.modules['listn.pipewire'] = m
""" + body
    out = tmp_path / "devices.env"
    subprocess.run([sys.executable, "-c", wrapper, str(out), name], check=True)
    result = subprocess.run([bash, "-c", '. "$1"; printf "%s" "$SYS_TARGET"', "bash", str(out)], capture_output=True, text=True)
    assert result.returncode == 0
    assert not marker.exists()
    assert result.stdout == name


@pytest.mark.parametrize("tracks,expected", [(('system',), 1), (('mic',), 1), (('system', 'mic'), 0)])
def test_pause_requires_both_wavs(tmp_path, tracks, expected):
    import sys
    import wave
    session = tmp_path / "sessions" / "test"
    session.mkdir(parents=True)
    (session / "transcript.json").write_text(json.dumps([{"text": "[paused]"}]))
    for track in tracks:
        with wave.open(str(session / f"audio-{track}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 24000)
    body = _heredoc("pause_phase.py")
    start = body.index('status, body = call("POST", "/api/live/start"')
    stub = """time.sleep = lambda _: None
def call(method, path, payload=None):
    if path == '/api/live/stop': return 200, {'id': 'test', 'pipewire': {'ok': True}}
    if path == '/api/live': return 200, {'live': {'paused': True}}
    return 200, {}
"""
    code = body[:start] + stub + body[start:]
    env = dict(os.environ, HWCHECK_PORT="1", LISTN_DATA_DIR=str(tmp_path))
    proc = subprocess.run([sys.executable, "-c", code, "sys", "mic"], env=env, capture_output=True, text=True)
    assert proc.returncode == expected, proc.stdout + proc.stderr


@pytest.mark.parametrize("option,payload", [
    ("--port", "1$(touch {marker})"),
    ("--duration", "5$(touch {marker})"),
    ("--out", "$(touch {marker})"),
    ("--system-target", "node$(touch {marker})"),
    ("--mic-target", "node$(touch {marker})"),
])
def test_external_argument_payloads_never_execute(tmp_path, option, payload):
    marker = tmp_path / "injected"
    value = payload.format(marker=marker)
    proc = run_script(option, value, "--list",
                      env={"XDG_RUNTIME_DIR": "/nonexistent-hwcheck-runtime"})
    assert not marker.exists(), proc.stdout + proc.stderr
    assert proc.returncode == 3


def test_listn_cmd_shell_metacharacters_are_not_evaluated(tmp_path):
    marker = tmp_path / "injected"
    command = f"python3 -m listn $(touch {marker})"
    proc = run_script("--list", env={"LISTN_CMD": command,
                                       "XDG_RUNTIME_DIR": "/nonexistent-hwcheck-runtime"})
    assert not marker.exists(), proc.stdout + proc.stderr
    assert proc.returncode == 3


def test_valid_duration_and_port_are_accepted_by_argument_validation():
    # Validation precedes environment preflight, so a valid value reaches the
    # expected missing-session error rather than being rejected as malformed.
    proc = run_script("--duration", "5", "--port", "8801", "--list",
                      env={"XDG_RUNTIME_DIR": "/nonexistent-hwcheck-runtime"})
    assert proc.returncode == 3
    assert "--port must" not in proc.stderr
    assert "--duration must" not in proc.stderr
