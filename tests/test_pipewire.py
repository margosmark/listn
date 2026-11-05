"""PipeWire capture prototype tests — no daemon, no hardware, synthetic audio.

Discovery parses pw-dump payloads recorded from a real headless PipeWire
1.4.2 daemon; the recorder runs a stand-in ``pw-record`` that speaks the
same contract as the real tool (raw int16 PCM on stdout, exit code 1 after
SIGINT, per-target behaviour keyed by node name).
"""

from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from listn import pipewire as pw
from listn.ipc import TRACK_MIC, TRACK_SYSTEM

# ---------------------------------------------------------------------------
# Stand-in pw-record
# ---------------------------------------------------------------------------

FAKE_PW_RECORD = r'''#!/usr/bin/env python3
"""Fake pw-record: raw int16 PCM on stdout, exit 1 on SIGINT (PW 1.4.2)."""
import hashlib, math, os, signal, struct, sys, time

args = sys.argv[1:]

def option(name, default=None):
    try:
        return args[args.index(name) + 1]
    except ValueError:
        return default

target = option("--target", "")
rate = int(option("--rate", "48000"))
channels = int(option("--channels", "2"))

# listn binds application streams by their unique object.serial (the real
# pw-record resolves digits as serials too). Resolve serial -> node.name
# through the fake graph so the audio still "lives on the node": mode
# keywords (silent/die/...) and the content key follow the node, exactly
# like the daemon, instead of following the digits.
if target.isdigit():
    try:
        import json as _json
        with open(os.environ["FAKE_PW_DUMP_FILE"]) as _fh:
            for _obj in _json.load(_fh):
                _props = (_obj.get("info") or {}).get("props") or {}
                if str(_props.get("object.serial")) == target:
                    target = _props.get("node.name") or target
                    break
    except (OSError, ValueError, KeyError):
        pass

log = os.environ.get("FAKE_PW_ARGV_FILE")
if log:
    with open(log, "a") as fh:
        fh.write(" ".join(args) + "\n")

mode = "tone"
for candidate in ("stubborn", "die", "silent", "fallback"):
    if candidate in target:
        mode = candidate
        break

# The audio lives on the node, like the real graph: two processes aimed at
# the same target emit the same waveform, aimed at different targets
# different ones. "fallback" models pw-record binding to the session
# default source whatever --target says (observed on real hardware), so
# both tracks then carry one node's content — the two-track independence
# check in report() has to catch that.
key = "session-default-source" if mode == "fallback" else target
digest = hashlib.sha256(key.encode("utf-8")).digest()
hz = 300.0 + digest[0] * 6.0
amp = 0.30 + (digest[1] % 90) / 450.0
phase = digest[2] / 255.0 * 2.0 * math.pi

state = {"stop": False}

def on_stop(signum, frame):
    state["stop"] = True

if mode == "stubborn":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
else:
    signal.signal(signal.SIGINT, on_stop)

CHUNK = max(1, rate // 10)          # 0.1 s of audio per write
written = 0
while not state["stop"]:
    if mode == "die" and written >= rate // 4:
        sys.stderr.write("fake device failure: node vanished\n")
        sys.stderr.flush()
        sys.exit(3)
    base = written
    frame = []
    for i in range(CHUNK):
        if mode == "silent":
            value = 0
        else:
            value = int(amp * 32767 * math.sin(2 * math.pi * hz * (base + i) / rate + phase))
        for _ in range(channels):
            frame.append(value)
    sys.stdout.buffer.write(struct.pack("<%dh" % len(frame), *frame))
    sys.stdout.buffer.flush()
    written += CHUNK
    time.sleep(0.01)                # ~10x real time: tests stay fast
sys.exit(1)
'''


@pytest.fixture
def fake_pw(tmp_path, monkeypatch):
    script = tmp_path / "pw-record"
    script.write_text(FAKE_PW_RECORD)
    script.chmod(0o755)
    argv_log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_PW_ARGV_FILE", str(argv_log))
    return {"script": str(script), "argv_log": argv_log, "out": tmp_path / "out"}


# ---------------------------------------------------------------------------
# Graph fixtures (shapes copied from a real pw-dump of PipeWire 1.4.2)
# ---------------------------------------------------------------------------

def sample_dump():
    node = lambda i, props: {  # noqa: E731
        "id": i, "type": "PipeWire:Interface:Node", "version": 3,
        "permissions": ["r", "x", "m"], "info": {"props": props},
    }
    return [
        {"id": 0, "type": "PipeWire:Interface:Core", "version": 4,
         "permissions": ["r", "x", "m"], "info": {"cookie": 1, "user-name": "u", "host-name": "h"}},
        node(30, {"media.class": None, "node.name": "Dummy-Driver", "object.serial": 30}),
        node(32, {"media.class": "Audio/Sink", "node.name": "alsa_output.pci-0000.analog-stereo",
                  "object.serial": 32, "priority.session": 1000, "node.description": "Speakers"}),
        node(34, {"media.class": "Audio/Source", "node.name": "alsa_input.usb-mic.mono-fallback",
                  "object.serial": 34, "priority.session": 2000, "node.description": "USB Mic"}),
        node(36, {"media.class": "Audio/Source", "node.name": "bluez_sink.AA.monitor",
                  "object.serial": 36, "monitor.sink": 32}),
        {"id": 40, "type": "PipeWire:Interface:Metadata", "version": 3,
         "permissions": ["r", "w", "x"], "props": {"metadata.name": "default", "object.serial": 41},
         "metadata": [
             {"subject": 0, "key": "default.audio.sink", "type": "Spa:String:JSON",
              "value": {"name": "alsa_output.pci-0000.analog-stereo"}},
             {"subject": 0, "key": "default.audio.source", "type": "Spa:String:JSON",
              "value": {"id": 34}},
             {"subject": 0, "key": "clock.rate", "type": "", "value": 48000},
         ]},
    ]


def node(node_id, name, media_class, serial=None, priority=None, monitor_of=None, description=None):
    return pw.Node(id=node_id, name=name, serial=node_id if serial is None else serial,
                   media_class=media_class, priority=priority, monitor_of=monitor_of,
                   description=description)


def graph(sink="sys.out", source="mic.in", default_sink="sys.out", default_source="mic.in",
          nodes=None):
    nodes = nodes if nodes is not None else [
        node(30, "Dummy-Driver", None),
        node(32, sink, "Audio/Sink", priority=100),
        node(34, source, "Audio/Source", priority=50),
    ]
    return pw.Graph(nodes=nodes, default_sink=default_sink, default_source=default_source)


def make_targets(sys_name="sys.tone", mic_name="mic.tone"):
    return pw.Targets(
        system=pw.TrackTarget("system", TRACK_SYSTEM,
                              node(32, sys_name, "Audio/Sink"), "default"),
        mic=pw.TrackTarget("mic", TRACK_MIC,
                           node(34, mic_name, "Audio/Source"), "default"),
    )


class CannedRunner:
    """Answers pw-dump with canned output; spawn() is the real subprocess."""

    def __init__(self, stdout="", rc=0, stderr="", exc=None):
        self.stdout, self.rc, self.stderr, self.exc = stdout, rc, stderr, exc
        self.calls = []
        self.spawned = []

    def run(self, argv, timeout=None):
        self.calls.append(list(argv))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, self.rc, self.stdout, self.stderr)

    def spawn(self, argv, **kwargs):
        proc = subprocess.Popen(argv, **kwargs)
        self.spawned.append(proc)
        return proc


# ---------------------------------------------------------------------------
# Discovery and target selection
# ---------------------------------------------------------------------------

def test_parse_graph_reads_nodes_defaults_and_identity():
    g = pw.parse_graph(sample_dump())
    assert {n.name for n in g.nodes} == {
        "Dummy-Driver", "alsa_output.pci-0000.analog-stereo",
        "alsa_input.usb-mic.mono-fallback", "bluez_sink.AA.monitor"}
    assert g.default_sink == "alsa_output.pci-0000.analog-stereo"
    # default.audio.source arrived as {"id": 34} and resolved to the name
    assert g.default_source == "alsa_input.usb-mic.mono-fallback"
    assert [n.name for n in g.sinks] == ["alsa_output.pci-0000.analog-stereo"]
    # the monitor is a source but never a microphone candidate
    assert [n.name for n in g.sources] == ["alsa_input.usb-mic.mono-fallback"]
    monitor = g.by_name("bluez_sink.AA.monitor")
    assert monitor.is_monitor and monitor.monitor_of == 32


def test_parse_graph_rejects_unexpected_json():
    with pytest.raises(pw.CaptureError, match="expected a list"):
        pw.parse_graph({"id": 1})
    with pytest.raises(pw.CaptureError, match="no PipeWire nodes"):
        pw.parse_graph([])


def test_discover_parses_runner_output():
    runner = CannedRunner(stdout=json.dumps(sample_dump()))
    g = pw.discover(runner=runner)
    assert runner.calls == [["pw-dump"]]
    assert g.default_sink.endswith("analog-stereo")


def test_discover_reports_pw_dump_failure_with_stderr_tail():
    runner = CannedRunner(stdout="", rc=255, stderr="x: pw_context_connect() failed: Host is down\n")
    with pytest.raises(pw.CaptureError, match="Host is down"):
        pw.discover(runner=runner)


def test_discover_reports_bad_json_and_missing_binary():
    with pytest.raises(pw.CaptureError, match="did not return JSON"):
        pw.discover(runner=CannedRunner(stdout="{nope"))
    with pytest.raises(pw.CaptureError, match="could not be executed"):
        pw.discover(runner=CannedRunner(exc=FileNotFoundError("pw-dump")))
    with pytest.raises(pw.CaptureError, match="timed out"):
        pw.discover(runner=CannedRunner(exc=subprocess.TimeoutExpired("pw-dump", 1)))


def test_discover_missing_binary_message_without_runner(monkeypatch):
    monkeypatch.setattr(pw.shutil, "which", lambda name: None)
    with pytest.raises(pw.CaptureError, match="pw-dump not found"):
        pw.discover()


def node_payload(node_obj, **extra):
    props = {"node.name": node_obj.name, "object.serial": node_obj.serial,
             "media.class": node_obj.media_class, **extra}
    return json.dumps([{"id": node_obj.id, "type": "PipeWire:Interface:Node",
                        "version": 3, "info": {"props": props}}])


def test_node_present_identity_checks():
    """pw-dump <id> semantics observed on PipeWire 1.4.2.

    A vanished id prints nothing and exits 0; ids are recycled, so the
    serial (and name) must still match for the node to count as present.
    """
    watched = node(32, "sys.out", "Audio/Sink", serial=32)
    assert pw._node_present(CannedRunner(stdout=node_payload(watched), rc=0), watched) is True
    # id gone
    assert pw._node_present(CannedRunner(stdout="", rc=0), watched) is False
    # id recycled by a different node (serial and name differ)
    recycled = json.dumps([{"id": 32, "type": "PipeWire:Interface:Node", "version": 3,
                            "info": {"props": {"node.name": "other", "object.serial": 500}}}])
    assert pw._node_present(CannedRunner(stdout=recycled, rc=0), watched) is False
    # daemon unreachable: uncertainty must not kill a healthy recording
    assert pw._node_present(CannedRunner(stdout="", rc=255), watched) is True
    assert pw._node_present(CannedRunner(exc=FileNotFoundError()), watched) is True
    assert pw._node_present(CannedRunner(stdout="{broken", rc=0), watched) is True


def collision_dump(node_id, node_name, node_serial):
    """``pw-dump <id>`` JSON where an unrelated object comes first.

    Reproduced on PipeWire 1.4.2 (issue #4): the first element is a
    Metadata object whose ``object.serial`` equals the queried id, while
    the real node sits second with a different serial (ids and serials are
    independent counters, so this collision recurs).
    """
    return json.dumps([
        {"id": 30, "type": "PipeWire:Interface:Metadata", "version": 3,
         "permissions": ["r", "w", "x"],
         "props": {"metadata.name": "settings", "object.serial": node_id}},
        {"id": node_id, "type": "PipeWire:Interface:Node", "version": 3,
         "permissions": ["r", "x", "m"],
         "info": {"props": {"media.class": "Audio/Sink", "node.name": node_name,
                            "object.serial": node_serial}}},
    ])


def collision_payloads():
    return {32: collision_dump(32, "sys.tone", 34),
            34: collision_dump(34, "mic.tone", 36)}


def test_node_present_ignores_leading_object_with_matching_serial():
    """Regression for issue #4: select by id, not by position.

    ``payload[0]`` here is a Metadata object with ``object.serial == 32``
    while the watched node is ``id 32, serial 34``. Reading positionally
    compared serial 32 against 34 and reported the healthy node as gone,
    which made ``TwoTrackRecorder.start()`` refuse to start.
    """
    watched = node(32, "test.system", "Audio/Sink", serial=34)
    payload = json.loads(collision_dump(32, "test.system", 34))
    # the fixture must actually reproduce the collision
    assert payload[0]["id"] != watched.id
    assert payload[0]["props"]["object.serial"] == watched.id
    assert payload[0]["props"]["object.serial"] != watched.serial
    runner = CannedRunner(stdout=json.dumps(payload), rc=0)
    assert pw._node_present(runner, watched) is True
    assert runner.calls == [["pw-dump", "32"]]


def test_node_present_false_when_no_element_carries_the_id():
    """No element with the queried id means the node is gone (issue #4)."""
    watched = node(32, "test.system", "Audio/Sink", serial=34)
    only_other_objects = json.dumps([
        {"id": 30, "type": "PipeWire:Interface:Metadata",
         "props": {"metadata.name": "settings", "object.serial": 32}},
        "not-an-object",
        {"id": 33, "type": "PipeWire:Interface:Node",
         "info": {"props": {"node.name": "someone-else", "object.serial": 33}}},
        {"type": "PipeWire:Interface:Node",  # no id at all
         "info": {"props": {"node.name": "test.system", "object.serial": 34}}},
    ])
    assert pw._node_present(CannedRunner(stdout=only_other_objects, rc=0), watched) is False
    # a non-list payload with content never identifies the node either
    assert pw._node_present(CannedRunner(stdout='{"id": 32}', rc=0), watched) is False
    # still gone when the dump is empty
    assert pw._node_present(CannedRunner(stdout="", rc=0), watched) is False


def test_recorder_start_accepts_present_nodes_despite_serial_collision(fake_pw):
    """End-to-end for issue #4: start() must not refuse a healthy node."""
    targets = pw.Targets(
        system=pw.TrackTarget("system", TRACK_SYSTEM,
                              node(32, "sys.tone", "Audio/Sink", serial=34), "default"),
        mic=pw.TrackTarget("mic", TRACK_MIC,
                           node(34, "mic.tone", "Audio/Source", serial=36), "default"),
    )
    payloads = collision_payloads()

    class PerIdDumpRunner(CannedRunner):
        def run(self, argv, timeout=None):
            self.calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, payloads[int(argv[1])], "")

    recorder = pw.TwoTrackRecorder(fake_pw["out"] / "collision", targets,
                                   runner=PerIdDumpRunner(),
                                   pw_record=fake_pw["script"], node_watch_interval=0.05)
    recorder.start()  # the bug: CaptureError "already gone" was raised here
    time.sleep(0.3)
    report = recorder.stop()
    assert report["ok"] is True, report
    assert report["tracks"]["system"]["verdict"] == "ok"
    assert report["tracks"]["mic"]["verdict"] == "ok"


def test_select_targets_uses_session_defaults():
    targets = pw.select_targets(graph())
    assert targets.system.selection == targets.mic.selection == "default"
    assert targets.system.node.name == "sys.out"
    assert targets.mic.node.name == "mic.in"
    assert (targets.system.track_id, targets.mic.track_id) == (TRACK_SYSTEM, TRACK_MIC)


def test_select_targets_ignores_default_source_that_is_a_sink():
    # Observed with WirePlumber 0.5.8 on a machine without microphones:
    # default.audio.source named an Audio/Sink node.
    g = graph(default_source="sys.out")
    targets = pw.select_targets(g)
    assert targets.mic.node.name == "mic.in"
    assert targets.mic.selection == "fallback"


def test_select_targets_never_picks_a_monitor_as_microphone():
    nodes = [
        node(32, "sys.out", "Audio/Sink", priority=100),
        node(36, "sys.out.monitor", "Audio/Source", monitor_of=32, priority=999),
        node(34, "real.mic", "Audio/Source", priority=10),
    ]
    g = pw.Graph(nodes=nodes, default_sink="sys.out", default_source="sys.out.monitor")
    targets = pw.select_targets(g)
    assert targets.mic.node.name == "real.mic"  # the monitor was skipped
    # with only a monitor around this must be a hard error, not a bad guess:
    g2 = pw.Graph(nodes=nodes[:2], default_sink="sys.out", default_source="sys.out.monitor")
    with pytest.raises(pw.CaptureError, match="No Audio/Source"):
        pw.select_targets(g2)


def test_select_targets_explicit_selector_and_wrong_class():
    g = graph()
    targets = pw.select_targets(g, system="32", mic="mic.in")
    assert targets.system.selection == "explicit" and targets.system.node.id == 32
    with pytest.raises(pw.CaptureError, match="not an Audio/Sink"):
        pw.select_targets(g, system="mic.in")
    with pytest.raises(pw.CaptureError, match="not an Audio/Source"):
        pw.select_targets(g, mic="sys.out")
    with pytest.raises(pw.CaptureError, match="No PipeWire node named 'ghost'"):
        pw.select_targets(g, system="ghost")


def test_select_targets_requires_output_and_falls_back_by_priority():
    g = pw.Graph(nodes=[node(34, "mic.in", "Audio/Source")],
                 default_sink=None, default_source=None)
    with pytest.raises(pw.CaptureError, match="No Audio/Sink"):
        pw.select_targets(g)
    g2 = pw.Graph(nodes=[
        node(32, "low", "Audio/Sink", priority=10),
        node(33, "high", "Audio/Sink", priority=900),
        node(34, "mic.in", "Audio/Source"),
    ], default_sink=None, default_source=None)
    targets = pw.select_targets(g2)
    assert targets.system.node.name == "high" and targets.system.selection == "fallback"


# ---------------------------------------------------------------------------
# WAV analysis (synthetic audio, including deliberately broken files)
# ---------------------------------------------------------------------------

def write_wav(path, seconds=1.0, rate=16000, channels=1, amp=0.5, freq=440.0):
    frames = int(seconds * rate)
    t = np.arange(frames) / rate
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono[:, None]
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path


def test_analyze_wav_levels_and_verdicts(tmp_path):
    ok = pw.analyze_wav(write_wav(tmp_path / "ok.wav", amp=0.5))
    assert ok["verdict"] == "ok" and ok["duration"] == pytest.approx(1.0, abs=0.01)
    assert ok["channels"] == 1 and ok["samplerate"] == 16000
    assert ok["rms_db"] == pytest.approx(ok["peak_db"] - 3.0, abs=0.2)
    quiet = pw.analyze_wav(write_wav(tmp_path / "quiet.wav", amp=0.003))
    assert quiet["verdict"] == "quiet"
    silent = pw.analyze_wav(write_wav(tmp_path / "silent.wav", amp=0.0))
    assert silent["verdict"] == "silent" and silent["peak_db"] <= pw.SILENT_PEAK_DB


def test_analyze_wav_flags_full_scale_clipping(tmp_path):
    """A peak that is the format maximum across a large share of the file
    is clipping, not a measured success.

    The real-hardware failure measured 7.2% of samples pinned at full
    scale while the peak was still reported as ``-0.0 dBFS`` and the
    verdict said ``ok``.
    """
    rate, n = 48000, 48000
    t = np.arange(n) / rate
    pcm = (0.5 * 32767 * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    pcm[: int(n * 0.072)] = 32767          # a pinned block, like the laptop files
    with wave.open(str(tmp_path / "clipped.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    result = pw.analyze_wav(tmp_path / "clipped.wav")
    assert result["verdict"] == "clipped", result
    assert result["peak_db"] == 0.0                       # the format maximum, honestly labelled
    assert result["fs_fraction"] == pytest.approx(0.072, abs=0.001)
    # a loud but not rail-pinned track stays ok — the threshold must not
    # fail normal full-scale peaks
    loud = pw.analyze_wav(write_wav(tmp_path / "loud.wav", rate=48000, amp=1.0))
    assert loud["verdict"] == "ok", loud
    assert loud["fs_fraction"] < pw.CLIPPED_FS_FRACTION


def test_compare_tracks_separates_same_from_different(tmp_path):
    a = write_wav(tmp_path / "a.wav", seconds=0.5, rate=48000, amp=0.4, freq=440.0)
    b = write_wav(tmp_path / "b.wav", seconds=0.5, rate=48000, amp=0.4, freq=440.0)
    c = write_wav(tmp_path / "c.wav", seconds=0.5, rate=48000, amp=0.4, freq=997.0)
    same = pw.compare_tracks(a, b)
    assert same["error"] is None and same["identical"] is True
    assert same["identical_fraction"] == 1.0
    assert same["corr"] == pytest.approx(1.0, abs=1e-6)
    assert "corr=" in same["cause"] and "samples match" in same["cause"]
    diff = pw.compare_tracks(a, c)
    assert diff["identical"] is False
    assert diff["identical_fraction"] < pw.IDENTICAL_SAMPLE_FRACTION
    assert diff["corr"] < pw.IDENTICAL_CORR
    # a comparison that cannot run never claims identity
    missing = pw.compare_tracks(a, tmp_path / "absent.wav")
    assert missing["identical"] is False and missing["error"]


def test_analyze_wav_missing_and_not_wav(tmp_path):
    missing = pw.analyze_wav(tmp_path / "nope.wav")
    assert missing["verdict"] is None and missing["error"] == "no file was written"
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wave file" * 10)
    broken = pw.analyze_wav(junk)
    assert broken["error"] == "not a WAV file"


def test_analyze_wav_unfinalized_header_still_reports_content(tmp_path):
    """SIGKILL leaves RIFF size 8 / data size 0 on PipeWire 1.4.2."""
    path = write_wav(tmp_path / "killed.wav", seconds=0.5, amp=0.4)
    raw = bytearray(path.read_bytes())
    raw[4:8] = (8).to_bytes(4, "little")
    raw[40:44] = (0).to_bytes(4, "little")
    path.write_bytes(raw)
    with pytest.raises(wave.Error):
        wave.open(str(path), "rb")  # the stdlib refuses it, as observed
    result = pw.analyze_wav(path)
    assert "not finalized" in result["error"]
    assert result["frames"] > 0 and result["duration"] == pytest.approx(0.5, abs=0.02)
    assert result["verdict"] == "ok"  # content still measured, error still set


def test_iter_blocks_yields_engine_shaped_pcm(tmp_path):
    stereo = write_wav(tmp_path / "st.wav", seconds=0.5, channels=2, amp=0.25, rate=48000)
    blocks = list(pw.iter_blocks(stereo, block_seconds=0.1))
    assert blocks and all(rate == 48000 for rate, _ in blocks)
    joined = np.concatenate([b for _, b in blocks])
    assert joined.ndim == 2 and joined.shape[1] == 2
    assert float(np.max(np.abs(joined))) == pytest.approx(0.25, abs=0.02)
    mono = write_wav(tmp_path / "mono.wav", seconds=0.3, amp=0.5, rate=48000)
    mono_blocks = list(pw.iter_blocks(mono, block_seconds=0.1))
    assert all(b.ndim == 1 for _, b in mono_blocks)
    total = sum(b.size for _, b in mono_blocks)
    assert total == 0.3 * 48000


# ---------------------------------------------------------------------------
# Recorder lifecycle: failure and silence paths with a fake pw-record
# ---------------------------------------------------------------------------

def start_recorder(fake_pw, targets=None, probe=None, **kwargs):
    kwargs.setdefault("pw_record", fake_pw["script"])
    kwargs.setdefault("probe", probe or (lambda n: True))
    kwargs.setdefault("node_watch_interval", 0.05)
    recorder = pw.TwoTrackRecorder(fake_pw["out"], targets or make_targets(), **kwargs)
    recorder.start()
    return recorder


def wait_for(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_recorder_captures_two_independent_tracks(fake_pw):
    recorder = start_recorder(fake_pw)
    time.sleep(0.35)
    report = recorder.stop()
    assert report["ok"] is True, report
    system, mic = report["tracks"]["system"], report["tracks"]["mic"]
    for track in (system, mic):
        assert track["state"] == "stopped" and track["verdict"] == "ok"
        assert track["duration"] > 0.3
        assert track["error"] is None
        assert track["exit_code"] in pw.CLEAN_STOP_CODES
        assert Path(track["file"]).is_file()
    # Files are mono downmixes (the engine's session files are mono too);
    # the stereo capture itself is asserted in the argv test below.
    assert system["channels"] == 1 and mic["channels"] == 1
    assert system["file"] != mic["file"]
    # both WAVs are finalized and readable by the stdlib
    for track in (system, mic):
        with wave.open(track["file"], "rb") as wf:
            assert wf.getnframes() / wf.getframerate() > 0.3
    # The feature itself: the two files must be genuinely different signals
    comparison = pw.compare_tracks(system["file"], mic["file"])
    assert comparison["error"] is None, comparison
    assert comparison["identical"] is False, comparison
    assert comparison["identical_fraction"] < pw.IDENTICAL_SAMPLE_FRACTION
    assert comparison["corr"] is None or comparison["corr"] < pw.IDENTICAL_CORR
    assert report["targets"]["system"]["name"] == "sys.tone"



def test_recorder_argv_follows_pw_record_contract(fake_pw):
    recorder = start_recorder(fake_pw)
    time.sleep(0.2)
    recorder.stop()
    lines = [line for line in fake_pw["argv_log"].read_text().splitlines() if line]
    assert len(lines) == 2
    by_target = {}
    for line in lines:
        args = line.split()
        assert args[0] == "--raw" and args[-1] == "-"
        assert args[args.index("--format") + 1] == "s16"
        assert args[args.index("--rate") + 1] == "48000"
        by_target[args[args.index("--target") + 1]] = args
    assert by_target["sys.tone"][by_target["sys.tone"].index("--channels") + 1] == "2"
    assert by_target["mic.tone"][by_target["mic.tone"].index("--channels") + 1] == "1"
    # An Audio/Sink target must request monitor capture explicitly: on real
    # hardware plain --target <sink> bound to the session default source
    # instead (both tracks recorded the mic, corr=1.0), so the property is
    # passed unconditionally for sinks and never for the microphone.
    assert (by_target["sys.tone"][by_target["sys.tone"].index("--properties") + 1]
            == "stream.capture.sink=true")
    assert "--properties" not in by_target["mic.tone"]


def test_recorder_never_calls_silence_a_success(fake_pw):
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.tone",
                                                            mic_name="mic.silent"))
    time.sleep(0.35)
    report = recorder.stop()
    assert report["tracks"]["mic"]["verdict"] == "silent"
    assert report["tracks"]["system"]["verdict"] == "ok"
    assert report["ok"] is False  # a silent microphone is never a successful capture


def test_recorder_refuses_two_tracks_with_one_signal(fake_pw):
    """Both pw-record processes bound to one source — the real-hardware
    fallback (``pw-record --target <sink>`` binding to the session default
    source: corr=1.000000, 96.6% identical samples measured on Mark's
    laptop) — must never be reported as a successful capture, even though
    each track on its own looks healthy.
    """
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.fallback",
                                                            mic_name="mic.fallback"))
    time.sleep(0.35)
    report = recorder.stop()
    assert report["ok"] is False, report
    for name in ("system", "mic"):
        track = report["tracks"][name]
        assert track["state"] == "failed", track
        assert track["error"] and "system and mic tracks are identical" in track["error"], track
        # per-track signal analysis alone cannot see the defect:
        assert track["verdict"] == "ok", track
    assert "samples match" in report["tracks"]["mic"]["error"]


def test_recorder_silent_system_alone_stays_ok(fake_pw):
    """Nobody talking silences the system track; the engine ignores it too."""
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.silent",
                                                            mic_name="mic.tone"))
    time.sleep(0.35)
    report = recorder.stop()
    assert report["tracks"]["system"]["verdict"] == "silent"
    assert report["tracks"]["mic"]["verdict"] == "ok"
    assert report["ok"] is True


def test_recorder_missing_pw_record_is_actionable(fake_pw):
    recorder = pw.TwoTrackRecorder(fake_pw["out"] / "nb", make_targets(),
                                   pw_record="/nonexistent/pw-record",
                                   probe=lambda n: True)
    with pytest.raises(pw.CaptureError, match="pw-record not found"):
        recorder.start()
    assert recorder.state == "idle"


def test_recorder_rollback_kills_partially_started_capture(fake_pw):
    runner = CannedRunner()
    # The mic probe fails after the system process was already spawned.
    recorder = pw.TwoTrackRecorder(
        fake_pw["out"] / "rb", make_targets(), runner=runner,
        pw_record=fake_pw["script"], node_watch_interval=0.05,
        probe=lambda n: n.name != "mic.tone")
    with pytest.raises(pw.CaptureError, match="already gone"):
        recorder.start()
    assert recorder.state == "idle"
    assert runner.spawned, "the system pw-record should have been spawned first"
    assert wait_for(lambda: runner.spawned[0].poll() is not None), "rollback left a process running"


def test_recorder_fails_only_the_track_whose_process_dies(fake_pw):
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.die",
                                                            mic_name="mic.tone"))
    assert wait_for(lambda: recorder._tracks["system"].state == "failed"), \
        "early pw-record exit must fail the track"
    time.sleep(0.2)
    report = recorder.stop()
    system, mic = report["tracks"]["system"], report["tracks"]["mic"]
    assert system["state"] == "failed"
    assert "exited by itself" in system["error"] and "exit code 3" in system["error"]
    assert "fake device failure" in system["error"]  # stderr tail is surfaced
    assert mic["state"] == "stopped" and mic["verdict"] == "ok"
    assert report["ok"] is False


def test_recorder_fails_track_when_target_node_disappears(fake_pw):
    class FlipProbe:
        def __init__(self, lose_after):
            self.lose_after, self.calls = lose_after, {}

        def __call__(self, n):
            count = self.calls[n.name] = self.calls.get(n.name, 0) + 1
            return count <= self.lose_after.get(n.name, 10 ** 9)

    probe = FlipProbe({"sys.tone": 2})
    recorder = start_recorder(fake_pw, probe=probe)
    time.sleep(0.4)
    report = recorder.stop()
    system, mic = report["tracks"]["system"], report["tracks"]["mic"]
    assert system["state"] == "failed"
    assert "disappeared" in system["error"]
    # the pw-record process was stopped instead of silently re-targeting
    assert system["exit_code"] in (1, -9, 137)
    assert mic["state"] == "stopped" and mic["verdict"] == "ok"
    assert report["ok"] is False


def test_recorder_force_kills_process_that_ignores_sigint(fake_pw):
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.stubborn",
                                                            mic_name="mic.tone"),
                              stop_grace=0.4)
    time.sleep(0.3)
    report = recorder.stop(grace=0.4)
    system = report["tracks"]["system"]
    assert system["state"] == "failed"
    assert "force-killed" in system["error"]
    assert system["exit_code"] == -9
    assert report["tracks"]["mic"]["state"] == "stopped"
    assert report["ok"] is False


def test_recorder_lifecycle_guards(fake_pw):
    recorder = pw.TwoTrackRecorder(fake_pw["out"] / "guards", make_targets(),
                                   pw_record=fake_pw["script"], probe=lambda n: True)
    with pytest.raises(pw.CaptureError, match="never started"):
        recorder.stop()
    recorder.start()
    with pytest.raises(pw.CaptureError, match="already recording"):
        recorder.start()
    time.sleep(0.3)  # let real audio land so the capture counts as successful
    first = recorder.stop()
    second = recorder.stop()
    assert first == second and first["ok"] is True
    with pytest.raises(pw.CaptureError, match="already stopped"):
        recorder.start()


def test_recorder_output_directory_is_private(fake_pw):
    out = fake_pw["out"] / "private"
    recorder = pw.TwoTrackRecorder(out, make_targets(), pw_record=fake_pw["script"],
                                   probe=lambda n: True, node_watch_interval=0.05)
    mode = stat.S_IMODE(os.stat(out).st_mode)
    assert mode == 0o700
    recorder.start()
    time.sleep(0.2)
    recorder.stop()


def test_recorder_on_block_feeds_engine_shape(fake_pw):
    blocks = []

    def on_block(track, rate, block):
        blocks.append((track, rate, block))

    recorder = start_recorder(fake_pw, on_block=on_block)
    time.sleep(0.3)
    recorder.stop()
    assert blocks, "no PCM blocks were delivered"
    tracks = {t for t, _, _ in blocks}
    assert tracks == {TRACK_SYSTEM, TRACK_MIC}
    for track, rate, block in blocks:
        assert rate == 48000 and block.dtype == np.float32
        assert block.ndim == (2 if track == TRACK_SYSTEM else 1)
        if track == TRACK_SYSTEM:
            assert block.shape[1] == 2


# ---------------------------------------------------------------------------
# Integration: captured PCM drives the engine's existing ingest pipeline
# ---------------------------------------------------------------------------

def test_capture_feeds_engine_ingest_session(data_dir, fake_factory, fake_pw):
    from listn.config import Settings
    from listn.engine import Engine

    engine = Engine(transcriber_factory=fake_factory)
    session = engine.start_ingest(
        Settings(model="tiny", chunk_seconds=2.0),
        {"source": "system", "mic": True}, "PipeWire prototype")
    pushes = []

    def on_block(track, rate, block):
        pushes.append(session.push_audio(track, rate, block))

    recorder = start_recorder(fake_pw, on_block=on_block)
    time.sleep(0.4)
    report = recorder.stop()
    assert report["ok"] is True
    assert pushes and all(pushes), "a block was pushed after the session stopped"
    engine.stop_live(wait=True)

    meta = engine.library.get(session.meta.id)
    assert meta.status == "done", meta.error
    root = engine.library.session_path(meta.id)
    for name, expect_channels in (("audio-system.wav", 1), ("audio-mic.wav", 1)):
        path = root / name
        assert path.is_file(), f"{name} missing from {root}"
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == expect_channels
            assert wf.getnframes() / wf.getframerate() > 1.0
    assert meta.duration > 1.0
    assert engine.library.segments(meta.id)
    assert session.silent_tracks == []


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def test_cli_dry_run_prints_selected_targets(monkeypatch, capsys):
    monkeypatch.setattr(pw, "discover", lambda **kw: graph())
    rc = pw.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sys.out" in out and "mic.in" in out and "selection: default" in out


def test_cli_unavailable_returns_exit_3(monkeypatch, capsys):
    def boom(**kw):
        raise pw.CaptureError("pw-dump failed (exit 255): Host is down")

    monkeypatch.setattr(pw, "discover", boom)
    rc = pw.main(["--dry-run"])
    captured = capsys.readouterr()
    assert rc == 3 and "Host is down" in captured.err


def test_cli_capture_exit_codes(monkeypatch, tmp_path, capsys):
    # Put the stand-in on PATH under the name the CLI resolves.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "pw-record"
    script.write_text(FAKE_PW_RECORD)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_PW_ARGV_FILE", str(tmp_path / "argv.log"))
    # Do not let a host PipeWire daemon (or its leftovers) decide node
    # liveness for these synthetic targets.
    monkeypatch.setattr(pw, "_node_present", lambda runner, node: True)
    monkeypatch.setattr(
        pw, "discover", lambda **kw: graph(sink="sys.tone", source="mic.tone",
                                           default_sink="sys.tone", default_source="mic.tone"))

    rc = pw.main(["--duration", "0.35", "--out", str(tmp_path / "ok"), "--json"])
    ok_report = json.loads(capsys.readouterr().out)
    assert rc == 0 and ok_report["ok"] is True
    assert ok_report["tracks"]["mic"]["verdict"] == "ok"

    monkeypatch.setattr(
        pw, "discover", lambda **kw: graph(sink="sys.tone", source="mic.silent",
                                           default_sink="sys.tone", default_source="mic.silent"))
    rc = pw.main(["--duration", "0.35", "--out", str(tmp_path / "sil"), "--json"])
    silence_report = json.loads(capsys.readouterr().out)
    assert rc == 1 and silence_report["ok"] is False
    assert silence_report["tracks"]["mic"]["verdict"] == "silent"

    # Both tracks recording one source (session fallback) is a failure
    # with a stated cause, never a printed success.
    monkeypatch.setattr(
        pw, "discover", lambda **kw: graph(sink="sys.fallback", source="mic.fallback",
                                           default_sink="sys.fallback",
                                           default_source="mic.fallback"))
    rc = pw.main(["--duration", "0.35", "--out", str(tmp_path / "same"), "--json"])
    same_report = json.loads(capsys.readouterr().out)
    assert rc == 1 and same_report["ok"] is False
    assert "identical" in (same_report["tracks"]["system"]["error"] or "")
    rc = pw.main(["--duration", "0.35", "--out", str(tmp_path / "same2")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Capture NOT OK" in captured.out
    assert "Capture OK:" not in captured.out
    assert "identical" in captured.out
