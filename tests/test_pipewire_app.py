"""L1 per-application PipeWire capture — synthetic graph tests.

No daemon, no hardware and no production audio: ``pw-dump`` and
``pw-record`` are stand-ins speaking the same contract as the real tools,
and every graph fixture is written in the shape PipeWire 1.4.2 prints.
The real two-application isolation procedure (two tones on a live Plasma
session) is documented — not executed here — in docs/LINUX_CAPTURE.md.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from listn import capture_contract as cc
from listn import cli
from listn import pipewire as pw
from listn.engine import Engine
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD
from .test_pipewire import FAKE_PW_RECORD, wait_for
from .test_pipewire_wiring import FAKE_PW_DUMP

SINK = "alsa_output.pci-0000.analog-stereo"
MIC = "alsa_input.usb-mic.mono-fallback"
APP_A = "listn.test.appA"
APP_B = "listn.test.appB"
APP_SILENT = "listn.test.app.silent"

# What applications publish about *content*. None of it may reach a report.
PRIVATE_TITLE = "Quarterly Roadmap — the merger nobody announced"
PRIVATE_MEDIA = "Secret Song Title"
CONTENT_NEEDLES = (PRIVATE_TITLE, PRIVATE_MEDIA, "merger", "Secret Song")


# ---------------------------------------------------------------------------
# Graph fixtures (shapes copied from a real pw-dump of PipeWire 1.4.2)
# ---------------------------------------------------------------------------

def _node(node_id, props):
    return {"id": node_id, "type": "PipeWire:Interface:Node", "version": 3,
            "permissions": ["r", "x", "m"], "info": {"props": props}}


def app_dump():
    """A graph holding two application streams, listn's own capture stream,
    a sink, a microphone and a monitor — plus the title-bearing properties
    applications really publish."""
    return [
        _node(30, {"media.class": None, "node.name": "Dummy-Driver",
                   "object.serial": 30}),
        _node(32, {"media.class": "Audio/Sink", "node.name": SINK,
                   "object.serial": 32, "priority.session": 1000,
                   "node.description": "Speakers"}),
        _node(34, {"media.class": "Audio/Source", "node.name": MIC,
                   "object.serial": 34, "priority.session": 2000,
                   "node.description": "USB Mic"}),
        _node(36, {"media.class": "Audio/Source", "node.name": "bluez_sink.AA.monitor",
                   "object.serial": 36, "monitor.sink": 32}),
        _node(50, {"media.class": "Stream/Output/Audio", "node.name": APP_A,
                   "object.serial": 150, "priority.session": 700,
                   "application.name": "AppA",
                   "application.process.binary": "appa",
                   "application.process.id": 4242,
                   # content the app publishes about what it is playing:
                   "node.description": PRIVATE_TITLE,
                   "media.name": PRIVATE_MEDIA,
                   "window.title": PRIVATE_TITLE}),
        _node(52, {"media.class": "Stream/Output/Audio", "node.name": APP_B,
                   "object.serial": 152,
                   "application.name": "AppB",
                   "application.process.binary": "appb",
                   "application.process.id": 4243,
                   "media.name": PRIVATE_MEDIA}),
        # listn's own recorder: a *capture* stream, never an app candidate.
        _node(54, {"media.class": "Stream/Input/Audio", "node.name": "listn.record",
                   "object.serial": 154, "application.name": "pw-record"}),
        {"id": 40, "type": "PipeWire:Interface:Metadata", "version": 3,
         "permissions": ["r", "w", "x"],
         "props": {"metadata.name": "default", "object.serial": 41},
         "metadata": [
             {"subject": 0, "key": "default.audio.sink", "type": "Spa:String:JSON",
              "value": {"name": SINK}},
             {"subject": 0, "key": "default.audio.source", "type": "Spa:String:JSON",
              "value": {"id": 34}},
         ]},
    ]


@pytest.fixture
def fake_pw_app_env(tmp_path, monkeypatch):
    """Stand-in pw-dump/pw-record on PATH, serving :func:`app_dump`."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("pw-record", FAKE_PW_RECORD), ("pw-dump", FAKE_PW_DUMP)):
        script = bindir / name
        script.write_text(body)
        script.chmod(0o755)
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(app_dump()))
    monkeypatch.setenv("FAKE_PW_DUMP_FILE", str(graph_file))
    monkeypatch.setenv("FAKE_PW_ARGV_FILE", str(tmp_path / "argv.log"))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return {"graph_file": graph_file, "argv_log": tmp_path / "argv.log",
            "tmp_path": tmp_path}


@pytest.fixture
def pw_client_app(data_dir, fake_factory, fake_pw_app_env):
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


def read_argv(fake_pw_app_env) -> list[str]:
    path = fake_pw_app_env["argv_log"]
    return path.read_text().splitlines() if path.exists() else []


def run_capture(tmp_path, system, mic=MIC, *, out="rec", **kwargs):
    """Record both tracks with an explicit pair of targets; return (report, recorder)."""
    targets = pw.select_targets(pw.parse_graph(app_dump()), system, mic)
    recorder = pw.TwoTrackRecorder(tmp_path / out, targets, **kwargs)
    recorder.start()
    time.sleep(0.3)
    return recorder.stop(), recorder


# ---------------------------------------------------------------------------
# Enumeration: only output-producing application streams
# ---------------------------------------------------------------------------

def test_app_stream_enumeration_is_output_producing_only():
    graph = pw.parse_graph(app_dump())
    assert [n.name for n in graph.app_streams] == [APP_A, APP_B]
    assert all(n.is_app_stream for n in graph.app_streams)
    # sinks, sources, monitors and *capture* streams are never candidates
    for excluded in (SINK, MIC, "bluez_sink.AA.monitor", "listn.record"):
        assert graph.by_name(excluded).is_app_stream is False
    # whole-sink enumeration is untouched by their presence
    assert [n.name for n in graph.sinks] == [SINK]
    assert [n.name for n in graph.sources] == [MIC]


def test_app_stream_label_is_the_application_name_and_never_a_title():
    graph = pw.parse_graph(app_dump())
    app_a = graph.by_name(APP_A)
    # node.description / media.name / window.title all carry content; the
    # label and the report carry the application's own name instead.
    assert app_a.description == "AppA"
    assert app_a.application_name == "AppA"
    blob = json.dumps(app_a.to_dict(), ensure_ascii=False)
    for needle in CONTENT_NEEDLES:
        assert needle not in blob


def test_app_stream_label_falls_back_to_the_process_binary():
    payload = app_dump()
    props = payload[4]["info"]["props"]
    del props["application.name"]
    graph = pw.parse_graph(payload)
    assert graph.by_name(APP_A).description == "appa"


def test_app_stream_without_any_application_property_has_no_label():
    payload = app_dump()
    props = payload[4]["info"]["props"]
    for key in ("application.name", "application.process.binary"):
        props.pop(key, None)
    graph = pw.parse_graph(payload)
    node = graph.by_name(APP_A)
    assert node.description is None
    assert node.application_name is None
    assert node.to_dict()["description"] is None


def test_identity_is_the_node_name_and_survives_pid_and_id_churn():
    """PipeWire recycles ids and PIDs; the node name is what selection keys on."""
    first = pw.parse_graph(app_dump())
    churned = json.loads(json.dumps(app_dump()))
    for obj in churned:
        props = obj.get("info", {}).get("props", {})
        if props.get("node.name") == APP_A:
            obj["id"] = 91                       # recycled id
            props["object.serial"] = 901         # new serial
            props["application.process.id"] = 31337  # new PID
    second = pw.parse_graph(churned)
    assert cc.stable_identity(first.by_name(APP_A).to_dict()) == APP_A
    assert cc.stable_identity(second.by_name(APP_A).to_dict()) == APP_A
    targets_a = pw.select_targets(first, APP_A, MIC)
    targets_b = pw.select_targets(second, APP_A, MIC)
    assert targets_a.system.node.name == targets_b.system.node.name == APP_A
    assert targets_a.system.node.id != targets_b.system.node.id


# ---------------------------------------------------------------------------
# Selection: one app, whole sink preserved, refusals instead of fallbacks
# ---------------------------------------------------------------------------

def test_select_targets_takes_one_app_stream_for_the_system_track():
    graph = pw.parse_graph(app_dump())
    targets = pw.select_targets(graph, APP_A, MIC)
    system = targets.system
    assert system.selection == "explicit"
    assert system.node.media_class == "Stream/Output/Audio"
    assert cc.source_kind(system.to_dict()) == cc.KIND_APP
    # the microphone is untouched: still an independent Audio/Source track
    assert targets.mic.node.name == MIC
    assert cc.source_kind(targets.mic.to_dict()) == cc.KIND_MIC
    assert targets.mic.selection == "explicit"


def test_select_targets_resolves_a_unique_application_name():
    graph = pw.parse_graph(app_dump())
    targets = pw.select_targets(graph, "appb", MIC)   # case-insensitive label
    assert targets.system.node.name == APP_B
    assert targets.system.selection == "explicit"


def test_whole_sink_mode_is_unchanged_when_application_streams_exist():
    graph = pw.parse_graph(app_dump())
    # no selector -> the session default sink: the documented "all apps" mode
    defaulted = pw.select_targets(graph, None, None)
    assert defaulted.system.node.name == SINK
    assert defaulted.system.selection == "default"
    assert cc.source_kind(defaulted.system.to_dict()) == cc.KIND_SINK_MONITOR
    # an explicit sink stays a sink even though apps are selectable
    explicit = pw.select_targets(graph, SINK, MIC)
    assert explicit.system.node.name == SINK
    assert explicit.system.selection == "explicit"
    assert cc.source_kind(explicit.system.to_dict()) == cc.KIND_SINK_MONITOR


def test_ambiguous_application_name_is_refused():
    """One app holding several streams: pick a node name, never the first."""
    payload = app_dump()
    payload.append(_node(56, {"media.class": "Stream/Output/Audio",
                              "node.name": "listn.test.appA.two",
                              "object.serial": 156,
                              "application.name": "AppA"}))
    graph = pw.parse_graph(payload)
    with pytest.raises(pw.CaptureError) as excinfo:
        pw.select_targets(graph, "AppA", MIC)
    message = str(excinfo.value)
    assert "names 2 PipeWire nodes" in message
    assert APP_A in message and "listn.test.appA.two" in message
    # and it did NOT quietly fall back to the sink
    assert SINK not in message


def test_duplicate_node_names_are_refused():
    payload = app_dump()
    payload.append(_node(56, {"media.class": "Stream/Output/Audio",
                              "node.name": APP_A,     # same node name, new serial
                              "object.serial": 156,
                              "application.name": "AppA"}))
    graph = pw.parse_graph(payload)
    with pytest.raises(pw.CaptureError, match="names 2 PipeWire nodes"):
        pw.select_targets(graph, APP_A, MIC)


def _dup_name_payload(with_serial=True):
    """QA F1's graph: a second stream sharing ``APP_B``'s node.name.

    The id and application-name selectors still resolve to exactly one
    node; only the resulting ``node.name`` collides — the case where
    ``pw-record`` bound the first-created match and recorded the other
    application.
    """
    payload = app_dump()
    dup: dict = {"media.class": "Stream/Output/Audio",
                 "node.name": APP_B,
                 "application.name": "AppB2"}
    if with_serial:
        dup["object.serial"] = 156
    payload.append(_node(56, dup))
    return payload


def test_shared_node_name_is_refused_for_targets_bound_by_name():
    """A name-bound target whose ``node.name`` matches >1 node is refused.

    ``pw-record`` would bind the first-created node of that name, so the
    capture would hold a different node's audio than the report claims.
    This is the post-resolution check QA F1 asked for: the *selector* is
    unambiguous (id or application name) but the name it would be bound
    by is not. Sinks and microphones bind by name, and an application
    stream without a serial has nothing better to bind to.
    """
    # sink selected by id, two sinks share the name
    payload = app_dump()
    payload.append(_node(58, {"media.class": "Audio/Sink", "node.name": SINK,
                              "object.serial": 158, "priority.session": 900}))
    payload.append(_node(60, {"media.class": "Audio/Source", "node.name": MIC,
                              "object.serial": 160, "priority.session": 900}))
    graph = pw.parse_graph(payload)
    with pytest.raises(pw.CaptureError) as excinfo:
        pw.select_targets(graph, "58", MIC)
    message = str(excinfo.value)
    assert f"'{SINK}' names 2 PipeWire nodes" in message
    assert "A capture target must be unambiguous" in message
    assert "would record the first match" in message
    with pytest.raises(pw.CaptureError, match="names 2 PipeWire nodes"):
        pw.select_targets(graph, None, "60")   # mic path (_resolve)
    # an application stream with no serial to bind to: same refusal
    graph = pw.parse_graph(_dup_name_payload(with_serial=False))
    with pytest.raises(pw.CaptureError) as excinfo:
        pw.select_targets(graph, "AppB2", MIC)
    assert f"'{APP_B}' names 2 PipeWire nodes" in str(excinfo.value)
    # while the same graph WITH a serial is fine: binding cannot drift
    targets = pw.select_targets(pw.parse_graph(_dup_name_payload()), "AppB2", MIC)
    assert targets.system.node.serial == 156


def test_missing_app_is_refused_instead_of_falling_back_to_the_sink():
    graph = pw.parse_graph(app_dump())
    with pytest.raises(pw.CaptureError) as excinfo:
        pw.select_targets(graph, "listn.test.gone", MIC)
    message = str(excinfo.value)
    assert "No PipeWire node named 'listn.test.gone'" in message
    # the refusal names what *is* there rather than adopting the default
    assert "Application output streams: " + APP_A in message
    assert SINK in message       # listed as a seen node, never selected
    with pytest.raises(pw.CaptureError):
        pw.select_targets(graph, "No Such App", MIC)


def test_system_track_refuses_an_input_device():
    graph = pw.parse_graph(app_dump())
    with pytest.raises(pw.CaptureError, match="not an Audio/Sink"):
        pw.select_targets(graph, MIC, MIC)
    with pytest.raises(pw.CaptureError, match="not an Audio/Sink"):
        pw.select_targets(graph, "bluez_sink.AA.monitor", MIC)


def test_graph_without_application_streams_still_works():
    """Unsupported/absent app support: whole-sink selection is unaffected."""
    from .test_pipewire import sample_dump
    graph = pw.parse_graph(sample_dump())
    assert graph.app_streams == []
    with pytest.raises(pw.CaptureError, match="Application output streams: none"):
        pw.select_targets(graph, "some.app", MIC)
    targets = pw.select_targets(graph, None, None)
    assert targets.system.node.name == SINK


# ---------------------------------------------------------------------------
# Recorder: the right argv, honest failure when the stream closes
# ---------------------------------------------------------------------------

def test_app_target_argv_binds_by_serial_and_has_no_sink_monitor_property(
        fake_pw_app_env, data_dir, tmp_path):
    """The system track binds the app stream by its unique ``object.serial``.

    ``pw-record --target`` resolves a name to the *first-created* node
    carrying it, so the name must never reach the child process: with two
    streams sharing a ``node.name`` the name would record the other one
    (QA F1). Serials are unique and never reused.
    """
    app_node = pw.parse_graph(app_dump()).by_name(APP_A)
    assert app_node is not None
    serial = app_node.serial
    run_capture(tmp_path, APP_A, out="app")
    app_argv = next(line for line in read_argv(fake_pw_app_env)
                    if f"--target {serial}" in line)
    assert f"--target {APP_A}" not in app_argv
    # an application stream is tapped through its own output ports: asking
    # for monitor capture would describe a target this track does not have
    assert "stream.capture.sink" not in app_argv
    assert "--rate" in app_argv and "--format s16" in app_argv


def test_duplicate_node_name_capture_binds_the_selected_stream_by_serial(
        fake_pw_app_env, data_dir, tmp_path):
    """QA F1 regression: id and application-name selectors against two
    streams that share one ``node.name``.

    Both selectors resolve to the SECOND stream (id 56, serial 156), so
    the recorder must spawn ``pw-record`` with that unique serial — never
    with the shared name, which PipeWire resolves to the first-created
    match and which recorded the other application while ``report.ok``
    claimed this one (reproduced 5/5 on a real daemon before the fix).
    """
    payload = _dup_name_payload()
    fake_pw_app_env["graph_file"].write_text(json.dumps(payload))
    graph = pw.parse_graph(payload)
    for selector in ("56", "AppB2"):
        targets = pw.select_targets(graph, selector, MIC)
        assert targets.system.node.id == 56
        assert targets.system.node.serial == 156
        recorder = pw.TwoTrackRecorder(tmp_path / f"dup-{selector}", targets)
        recorder.start()
        time.sleep(0.3)
        report = recorder.stop()
        assert cc.validate_report(report) == []
    lines = read_argv(fake_pw_app_env)
    assert sum("--target 156" in line for line in lines) == 2
    # the shared name never reaches pw-record for either selection
    assert not any(f"--target {APP_B}" in line for line in lines)


def test_sink_target_argv_still_asks_for_monitor_capture(fake_pw_app_env, data_dir, tmp_path):
    run_capture(tmp_path, SINK, out="sink")
    sink_argv = next(line for line in read_argv(fake_pw_app_env) if SINK in line)
    assert "stream.capture.sink=true" in sink_argv


def test_app_capture_report_satisfies_the_l0_contract(fake_pw_app_env, data_dir, tmp_path):
    report, _ = run_capture(tmp_path, APP_A)
    assert cc.validate_report(report) == []
    assert report["targets"]["system"]["media_class"] == "Stream/Output/Audio"
    assert report["targets"]["mic"]["media_class"] == "Audio/Source"
    assert report["tracks"]["system"]["target"] == report["targets"]["system"]
    assert report["tracks"]["mic"]["target"] == report["targets"]["mic"]
    assert report["ok"] == cc.expected_ok(report)
    assert report["tracks"]["mic"]["verdict"] != "silent"


def test_app_stream_closing_mid_run_fails_the_system_track_and_keeps_the_mic(
        fake_pw_app_env, data_dir, tmp_path):
    """The application closes its stream: fail that track, keep recording the mic.

    ``pw-record`` would happily re-target something else — that is exactly
    what must never be reported as a capture of the selected application.
    """
    targets = pw.select_targets(pw.parse_graph(app_dump()), APP_A, MIC)
    recorder = pw.TwoTrackRecorder(tmp_path / "app", targets, node_watch_interval=0.1)
    recorder.start()
    assert wait_for(lambda: recorder._tracks["system"].frames > 0)
    assert wait_for(lambda: recorder._tracks["mic"].frames > 0)

    graph_file = Path(os.environ["FAKE_PW_DUMP_FILE"])
    payload = [obj for obj in json.loads(graph_file.read_text())
               if (obj.get("info", {}).get("props") or {}).get("node.name") != APP_A]
    graph_file.write_text(json.dumps(payload))

    assert wait_for(lambda: recorder._tracks["system"].state == "failed", timeout=8)
    assert recorder._tracks["mic"].state == "recording", "the mic must keep recording"
    mic_before = recorder._tracks["mic"].frames
    # Condition wait, not wall time: the claim is "the mic keeps delivering
    # audio after the app stream closed", so wait for delivery. The stand-in
    # generator can be CPU-starved for longer than any fixed window.
    assert wait_for(lambda: recorder._tracks["mic"].frames > mic_before,
                    timeout=10.0), (
        f"mic frames frozen at {mic_before} after the app stream closed "
        f"(state={recorder._tracks['mic'].state})")

    report = recorder.stop()
    system = report["tracks"]["system"]
    assert report["ok"] is False
    assert system["state"] == "failed"
    assert "closed while recording" in system["error"]
    assert SINK not in system["error"]          # never a whole-sink fallback
    # partial audio is retained, not laundered into success
    assert report["tracks"]["mic"]["state"] == "stopped"
    assert report["tracks"]["mic"]["verdict"] == "ok"
    assert Path(report["tracks"]["mic"]["file"]).is_file()
    assert cc.validate_report(report) == []


def test_app_capture_start_refuses_when_the_stream_is_already_gone(
        fake_pw_app_env, data_dir, tmp_path):
    graph_file = fake_pw_app_env["graph_file"]
    payload = [obj for obj in json.loads(graph_file.read_text())
               if (obj.get("info", {}).get("props") or {}).get("node.name") != APP_A]
    graph_file.write_text(json.dumps(payload))
    graph = pw.parse_graph(json.loads(graph_file.read_text()))
    # discovery no longer offers it: selecting it is a refusal, not a default
    with pytest.raises(pw.CaptureError, match="No PipeWire node named"):
        pw.select_targets(graph, APP_A, MIC)


def test_silent_app_stream_is_reported_silent(fake_pw_app_env, data_dir, tmp_path):
    """An application that is not playing yet: verdict ``silent``, never ``ok``.

    The run itself stays ``ok`` because the microphone carries signal —
    L0 §5 keeps a silent *system* track from failing the run ("nobody may
    have been playing anything"), and the validator refuses any other
    combination. The honest claim is the per-track verdict.
    """
    payload = app_dump()
    payload.append(_node(58, {"media.class": "Stream/Output/Audio",
                              "node.name": APP_SILENT,
                              "object.serial": 158,
                              "application.name": "SilentApp"}))
    fake_pw_app_env["graph_file"].write_text(json.dumps(payload))
    graph = pw.parse_graph(payload)
    targets = pw.select_targets(graph, APP_SILENT, MIC)
    recorder = pw.TwoTrackRecorder(tmp_path / "silent", targets)
    recorder.start()
    time.sleep(0.3)
    report = recorder.stop()
    system = report["tracks"]["system"]
    assert system["verdict"] == "silent"
    assert system["state"] == "stopped"
    assert system["error"] is None
    assert report["ok"] is True
    assert cc.validate_report(report) == []


# ---------------------------------------------------------------------------
# Wiring: CLI listing, CLI capture, API enumeration, API start
# ---------------------------------------------------------------------------

def test_record_list_shows_application_streams(fake_pw_app_env, data_dir, capsys):
    assert cli.main(["record", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Application output streams (Stream/Output/Audio):" in out
    assert APP_A in out and APP_B in out
    assert "AppA" in out and "AppB" in out
    assert "one application stream" in out
    # a capture stream and a monitor are never offered as a target
    assert "listn.record" not in out
    assert "bluez_sink.AA.monitor" not in out


def test_record_captures_an_application_stream_end_to_end(
        fake_pw_app_env, data_dir, tmp_path, capsys):
    out_dir = tmp_path / "rec"
    rc = cli.main(["record", "--backend", "pipewire",
                   "--system-target", APP_A, "--mic-target", MIC,
                   "--duration", "0.4", "--out", str(out_dir), "--json"])
    assert rc == 0, capsys.readouterr()
    report = json.loads(capsys.readouterr().out)
    assert cc.validate_report(report) == []
    assert report["targets"]["system"]["name"] == APP_A
    assert report["targets"]["system"]["media_class"] == "Stream/Output/Audio"
    assert report["targets"]["system"]["selection"] == "explicit"
    for name in ("system", "mic"):
        assert report["tracks"][name]["state"] == "stopped"
        assert report["tracks"][name]["frames"] > 0
    # content the application published never reaches the JSON report
    blob = json.dumps(report, ensure_ascii=False)
    for needle in CONTENT_NEEDLES:
        assert needle not in blob


def test_record_unknown_app_exits_3_without_starting(
        fake_pw_app_env, data_dir, tmp_path, capsys):
    out_dir = tmp_path / "never"
    rc = cli.main(["record", "--system-target", "listn.test.gone",
                   "--mic-target", MIC, "--duration", "0.2",
                   "--out", str(out_dir)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "No PipeWire node named 'listn.test.gone'" in err
    assert "Application output streams" in err
    assert not out_dir.exists()


def test_record_refuses_shared_name_without_recording_dir(
        fake_pw_app_env, data_dir, tmp_path, capsys):
    """The post-resolution shared-name refusal (QA F1) is CLI exit 3 with
    no recording directory — same refusal family as an ambiguous selector."""
    payload = _dup_name_payload(with_serial=False)
    fake_pw_app_env["graph_file"].write_text(json.dumps(payload))
    out_dir = tmp_path / "never"
    rc = cli.main(["record", "--system-target", "AppB2",
                   "--mic-target", MIC, "--duration", "0.2",
                   "--out", str(out_dir)])
    assert rc == 3
    err = capsys.readouterr().err
    assert f"'{APP_B}' names 2 PipeWire nodes" in err
    assert "A capture target must be unambiguous" in err
    assert not out_dir.exists()


def test_devices_endpoint_lists_application_streams(pw_client_app):
    devices = pw_client_app.get("/api/pipewire/devices").json()
    assert [n["name"] for n in devices["apps"]] == [APP_A, APP_B]
    assert devices["usable"] == {"system": True, "mic": True}
    # the enumeration is metadata only
    blob = json.dumps(devices, ensure_ascii=False)
    for needle in CONTENT_NEEDLES:
        assert needle not in blob
    assert devices["apps"][0]["media_class"] == "Stream/Output/Audio"


def test_live_start_declares_the_app_source_kind(pw_client_app):
    started = pw_client_app.post("/api/live/start", json={
        "capture": "pipewire", "title": "L1 app capture",
        "system_target": APP_A, "mic_target": MIC})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["capture"] == "ingest"
    device = body["device"]
    assert device["source"] == "app"       # the ingest enum's reserved value
    assert device["kind"] == "pipewire"
    assert device["mic"] is True
    assert "AppA" in device["name"]        # the user-visible selection
    # live status keeps the contract's shape with an app target
    status = pw_client_app.get("/api/live").json()["pipewire"]
    assert cc.validate_live_status(status) == []
    assert status["targets"]["system"]["media_class"] == "Stream/Output/Audio"
    stopped = pw_client_app.post("/api/live/stop")
    assert stopped.status_code == 200
    report = stopped.json()["pipewire"]
    assert cc.validate_report(report) == []
    for needle in CONTENT_NEEDLES:
        assert needle not in json.dumps(report, ensure_ascii=False)


def test_live_start_rejects_a_vanished_app_without_side_effects(pw_client_app):
    r = pw_client_app.post("/api/live/start", json={
        "capture": "pipewire", "system_target": "listn.test.gone",
        "mic_target": MIC})
    assert r.status_code == 400
    assert "No PipeWire node named 'listn.test.gone'" in r.json()["detail"]
    assert pw_client_app.get("/api/live").json()["live"] is None
    assert pw_client_app.get("/api/sessions").json()["results"] == []


def test_live_start_rejects_a_shared_name_without_side_effects(
        pw_client_app, fake_pw_app_env):
    """QA F1 over HTTP: shared ``node.name`` on a name-bound target ->
    400 with the refusal, nothing started, no session recorded."""
    payload = app_dump()
    payload.append(_node(58, {"media.class": "Audio/Sink", "node.name": SINK,
                              "object.serial": 158, "priority.session": 900}))
    fake_pw_app_env["graph_file"].write_text(json.dumps(payload))
    r = pw_client_app.post("/api/live/start", json={
        "capture": "pipewire", "title": "shared name",
        "system_target": "58", "mic_target": MIC})
    assert r.status_code == 400
    assert "names 2 PipeWire nodes" in r.json()["detail"]
    assert pw_client_app.get("/api/live").json()["live"] is None
    assert pw_client_app.get("/api/sessions").json()["results"] == []
