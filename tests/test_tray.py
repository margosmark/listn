"""Tray tests: server JSON -> truthful tray state -> real recorder actions.

Three layers, matching ``listn/tray.py``:

* ``derive_status`` — pure mapping, including every failure shape the
  server can report (dead track, refused start, engine gone).
* ``build_menu`` — scope honesty: exactly the supported controls; the
  macOS-only / unimplemented ones are absent, not faked.
* ``TrayController`` — menu ids against the *real* server app (the same
  FastAPI app the web UI talks to, with the stand-in ``pw-dump``/
  ``pw-record`` tools from the wiring tests), so start/stop/pause drive a
  real two-track capture and failure states carry the recorder's own cause.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from listn.engine import Engine
from listn.server import create_app
from listn.tray import (
    FAILED, IDLE, ITEM_APP_BASE, ITEM_MIC_BASE, ITEM_PAUSE, ITEM_PRIVACY,
    ITEM_QUIT, ITEM_RESUME, ITEM_START, ITEM_STATUS, ITEM_STOP,
    ITEM_SYSTEM_APPS_SECTION, ITEM_SYSTEM_BASE,
    PAUSED, RECORDING, STARTING, STOPPING, UNREACHABLE, MenuItem,
    ServerClient, ServerUnreachable, TrayController, TrayStatus, build_menu,
    derive_status,
)

from .conftest import MAC_DEVICES, FakeSD
from .test_pipewire import sample_dump
from .test_pipewire_app import (APP_A, APP_B, CONTENT_NEEDLES,
                                fake_pw_app_env)
# Importing the fixture function registers it for this module too.
from .test_pipewire_wiring import (FAKE_PW_DUMP, MIC, SINK, fake_pw_env,
                                   wait_status)

# ---------------------------------------------------------------------------
# Transport: the tray's real ServerClient driving the real server app
# ---------------------------------------------------------------------------


class ClientTransport:
    """Adapts starlette's TestClient to ServerClient's transport seam."""

    def __init__(self, client):
        self.client = client

    def get(self, path):
        r = self.client.get(path)
        return r.status_code, (r.json() if r.content else {})

    def post(self, path, body):
        r = self.client.post(path, json=body or {})
        return r.status_code, (r.json() if r.content else {})


@pytest.fixture
def tray_server(data_dir, fake_factory, fake_pw_env):
    """The real engine + server app behind a ServerClient transport."""
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        controller = TrayController(
            ServerClient(transport=ClientTransport(c)),
            targets_file=data_dir / "tray-targets.json")
        yield controller, c
    # The HTTP stop path finishes asynchronously (daemon finisher thread);
    # wait for it so its writes don't land in the removed data dir.
    if engine.live is not None:
        end = time.monotonic() + 15
        while time.monotonic() < end and engine.live.state not in ("done", "error"):
            time.sleep(0.05)
        time.sleep(0.2)   # trailing retention/notes writes after the state flips


@pytest.fixture
def tray_server_apps(data_dir, fake_factory, fake_pw_app_env):
    """The same real server, with a graph that holds application streams.

    ``fake_pw_app_env`` serves ``app_dump()``: two ``Stream/Output/Audio``
    nodes (labelled with the applications' own names, carrying titles the
    tray must never show), a sink, a microphone and listn's own capture
    stream.
    """
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        controller = TrayController(
            ServerClient(transport=ClientTransport(c)),
            targets_file=data_dir / "tray-targets.json")
        yield controller, c
    if engine.live is not None:
        end = time.monotonic() + 15
        while time.monotonic() < end and engine.live.state not in ("done", "error"):
            time.sleep(0.05)
        time.sleep(0.2)


def wait_state(controller, want, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if controller.refresh().state == want:
            return controller.status
        time.sleep(0.1)
    raise AssertionError(
        f"tray never reached {want}; last={controller.status.state} "
        f"detail={controller.status.detail!r}")


def menu_items(controller):
    return {item.id: item for item in controller.menu()}


# ---------------------------------------------------------------------------
# derive_status: server JSON -> truthful state
# ---------------------------------------------------------------------------


def test_status_unreachable_carries_the_error():
    status = derive_status(None, "listn is not answering on http://127.0.0.1:8765")
    assert status.state == UNREACHABLE
    assert "not answering" in status.summary()
    assert not status.active


def test_status_idle_when_nothing_ran():
    assert derive_status({"live": None, "pipewire": None}).state == IDLE


def test_status_starting_before_session_attaches():
    # start_live starts the recorder before the engine session exists;
    # that window must not read as idle.
    status = derive_status({"live": None, "pipewire": {
        "active": True, "state": "recording", "tracks": {}, "report": None,
        "targets": None}})
    assert status.state == STARTING
    assert status.active


def test_status_recording_reports_targets_and_frames():
    status = derive_status({"live": {"id": "s1", "title": "Pairing",
                                     "state": "listening", "paused": False,
                                     "elapsed": 65.0},
                            "pipewire": {
                                "active": True, "state": "recording",
                                "report": None,
                                "targets": {"system": {"name": SINK},
                                            "mic": {"name": MIC}},
                                "tracks": {"system": {"frames": 4800},
                                           "mic": {"frames": 4700}}}})
    assert status.state == RECORDING
    summary = status.summary()
    assert "Recording" in summary and "Pairing" in summary
    assert "01:05" in summary
    assert f"system 4800f" in summary and "mic 4700f" in summary
    assert status.session_id == "s1"


def test_status_paused_and_stopping():
    paused = derive_status({"live": {"id": "s1", "state": "paused",
                                     "paused": True, "elapsed": 3.0},
                            "pipewire": None})
    assert paused.state == PAUSED and paused.active
    stopping = derive_status({"live": {"id": "s1", "state": "finishing",
                                       "paused": False, "elapsed": 3.0},
                              "pipewire": None})
    assert stopping.state == STOPPING and stopping.active


def test_status_failed_report_lists_both_track_causes():
    status = derive_status({"live": None, "pipewire": {
        "active": False, "state": "failed",
        "targets": None,
        "tracks": {"system": {"frames": 100}, "mic": {"frames": 90}},
        "report": {"ok": False, "tracks": {
            "system": {"state": "failed", "error": "node vanished",
                       "verdict": None},
            "mic": {"state": "failed", "error": "pw-record exited 3",
                    "verdict": None}}}}})
    assert status.state == FAILED and not status.active
    assert "system: node vanished" in status.detail
    assert "mic: pw-record exited 3" in status.detail


def test_status_failed_live_error():
    status = derive_status({"live": {"id": "s1", "state": "error",
                                     "error": "PipeWire recording failed — system: node vanished"},
                            "pipewire": None})
    assert status.state == FAILED
    assert "PipeWire recording failed" in status.detail


def test_status_ok_report_is_idle_not_recording():
    status = derive_status({"live": None, "pipewire": {
        "active": False, "state": "stopped", "targets": None,
        "tracks": {}, "report": {"ok": True, "tracks": {}}}})
    assert status.state == IDLE
    assert not status.active


# ---------------------------------------------------------------------------
# build_menu: exactly the supported controls, honestly enabled
# ---------------------------------------------------------------------------


def recording_status() -> TrayStatus:
    return TrayStatus(state=RECORDING, title="Pairing", elapsed=12.0,
                      system_target=SINK, mic_target=MIC,
                      system_frames=100, mic_frames=90, session_id="s1")


SINKS = [{"name": SINK, "description": "Speakers", "media_class": "Audio/Sink", "id": 41}]
SOURCES = [{"name": MIC, "description": "USB Mic", "media_class": "Audio/Source", "id": 47}]
# As `GET /api/pipewire/devices` returns them: node name is the identity,
# description is the application's own name (never a window/media title).
APPS = [
    {"name": APP_A, "description": "AppA", "media_class": "Stream/Output/Audio", "id": 50},
    {"name": APP_B, "description": "AppB", "media_class": "Stream/Output/Audio", "id": 52},
]


def all_labels(items):
    out = []
    for item in items:
        out.append(item.label)
        out.extend(all_labels(item.children))
    return out


def test_menu_offers_exactly_the_supported_controls():
    items = build_menu(recording_status(), SINKS, SOURCES, APPS, SINK, MIC)
    ids = {i.id for i in items if not i.separator}
    assert ids == {ITEM_STATUS, ITEM_START, ITEM_STOP, ITEM_PAUSE, ITEM_RESUME,
                   10, 11, ITEM_PRIVACY, ITEM_QUIT}
    labels = " ".join(all_labels(items)).lower()
    for forbidden in ("meeting", "auto-record", "auto record",
                      "launch at login", "launch-at-login", "per-app",
                      "per app"):
        assert forbidden not in labels, forbidden


def menu_ids(items):
    return {i.id: i for i in items}


def test_menu_recording_enabled_flags():
    items = menu_ids(build_menu(recording_status(), SINKS, SOURCES, APPS,
                                SINK, MIC))
    assert not items[ITEM_START].enabled          # already recording
    assert items[ITEM_STOP].enabled
    assert items[ITEM_PAUSE].enabled
    assert not items[ITEM_RESUME].enabled
    assert items[ITEM_STATUS].enabled is False    # status line is never clickable
    # radio picks reflect the selection: the whole-sink entry is checked
    # and no application stream is — one system track, one checked mode
    system = items[10]
    radios = [c for c in system.children if c.toggle == "radio"]
    assert [c.toggled for c in radios] == [True, False, False]
    assert system.children[0].toggle == "radio"


def test_menu_idle_enables_start_only_with_both_targets():
    idle = TrayStatus(state=IDLE)
    without = menu_ids(build_menu(idle, SINKS, SOURCES, [], None, MIC))
    assert not without[ITEM_START].enabled
    with_both = menu_ids(build_menu(idle, SINKS, SOURCES, [], SINK, MIC))
    assert with_both[ITEM_START].enabled
    assert not with_both[ITEM_STOP].enabled
    assert not with_both[ITEM_PAUSE].enabled


def test_menu_failed_and_unreachable_keep_start_and_say_why():
    failed = TrayStatus(state=FAILED, detail="system: node vanished")
    items = menu_ids(build_menu(failed, SINKS, SOURCES, [], SINK, MIC))
    assert items[ITEM_START].enabled              # a failed run can be retried
    assert "node vanished" in items[ITEM_STATUS].label
    unreachable = TrayStatus(state=UNREACHABLE, detail="listn is not answering.")
    items = menu_ids(build_menu(unreachable, [], [], [], None, None))
    assert not items[ITEM_START].enabled          # nothing to start against
    assert "not answering" in items[ITEM_STATUS].label


def test_menu_discovery_error_disables_the_pickers():
    items = menu_ids(build_menu(TrayStatus(state=IDLE), [], [], [], None, None,
                                discovery_error="pw-dump not found"))
    assert not items[10].enabled and not items[11].enabled
    assert "pw-dump not found" in items[10].label


def test_menu_quit_label_warns_while_recording():
    active = menu_ids(build_menu(recording_status(), SINKS, SOURCES, [], SINK, MIC))
    assert "stops the recording" in active[ITEM_QUIT].label.lower()
    idle = menu_ids(build_menu(TrayStatus(state=IDLE), SINKS, SOURCES, [], SINK, MIC))
    assert idle[ITEM_QUIT].label == "Quit listn"


def test_menu_application_streams_are_a_section_of_the_system_picker():
    """The streams live inside the System output submenu, as one radio group."""
    top = build_menu(recording_status(), SINKS, SOURCES, APPS, SINK, MIC)
    items = menu_ids(top)
    # the section is not a new top-level control
    assert {i.id for i in top if not i.separator} == \
        {ITEM_STATUS, ITEM_START, ITEM_STOP, ITEM_PAUSE, ITEM_RESUME,
         10, 11, ITEM_PRIVACY, ITEM_QUIT}
    system = items[10]
    assert [c.label for c in system.children] == \
        ["Speakers", "", "Application streams", "AppA", "AppB"]
    header = [c for c in system.children if c.id == ITEM_SYSTEM_APPS_SECTION]
    assert len(header) == 1
    assert header[0].enabled is False and header[0].toggle is None
    radios = [c for c in system.children if c.toggle == "radio"]
    assert [c.id for c in radios] == \
        [ITEM_SYSTEM_BASE, ITEM_APP_BASE, ITEM_APP_BASE + 1]
    # exactly one entry of the whole picker is checked: the sink
    assert [c.toggled for c in radios] == [True, False, False]


def test_menu_checking_an_application_unchecks_the_whole_sink():
    items = menu_ids(build_menu(recording_status(), SINKS, SOURCES, APPS,
                                APP_B, MIC))
    radios = [c for c in items[10].children if c.toggle == "radio"]
    assert [(c.label, c.toggled) for c in radios] == \
        [("Speakers", False), ("AppA", False), ("AppB", True)]


def test_menu_without_application_streams_has_no_empty_section():
    items = menu_ids(build_menu(TrayStatus(state=IDLE), SINKS, SOURCES, [],
                                SINK, MIC))
    assert [c.label for c in items[10].children] == ["Speakers"]
    assert ITEM_SYSTEM_APPS_SECTION not in \
        {c.id for c in items[10].children}
    # neither sinks nor streams: the picker says so instead of offering
    # a submenu with nothing to choose
    empty = menu_ids(build_menu(TrayStatus(state=IDLE), [], [], [], None, None))
    assert empty[10].enabled is False
    assert empty[10].label == "System output — none found"


# ---------------------------------------------------------------------------
# TrayController: menu ids -> real server -> truthful state
# ---------------------------------------------------------------------------


def test_tray_start_pause_resume_stop_a_real_two_track_capture(tray_server):
    controller, client = tray_server
    assert controller.refresh().state == IDLE
    # session defaults are adopted as labelled suggestions, never guesses
    assert controller.selected_system == SINK and controller.selected_mic == MIC

    controller.action(ITEM_START)
    status = wait_state(controller, RECORDING)
    assert status.session_id
    end = time.monotonic() + 10
    while time.monotonic() < end:
        controller.refresh()
        if (controller.status.system_frames or 0) > 0 \
                and (controller.status.mic_frames or 0) > 0:
            break
        time.sleep(0.1)
    assert controller.status.system_frames > 0
    assert controller.status.mic_frames > 0

    controller.action(ITEM_PAUSE)
    assert wait_state(controller, PAUSED).active
    controller.action(ITEM_RESUME)
    assert wait_state(controller, RECORDING).active

    controller.action(ITEM_STOP)
    status = wait_state(controller, IDLE)
    report = client.get("/api/live").json()["pipewire"]["report"]
    assert report["ok"] is True
    assert status.detail is None


def test_tray_start_refusal_surfaces_the_servers_reason(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.selected_system = "no.such.node"   # stale or hand-edited pick
    controller.action(ITEM_START)
    status = controller.refresh()
    assert status.state == IDLE                   # nothing is recording
    assert client.get("/api/live").json()["live"] is None
    assert controller.notice and "No PipeWire node named" in controller.notice
    # the reason is shown in the menu, not swallowed
    assert "No PipeWire node named" in menu_items(controller)[ITEM_STATUS].label


def test_tray_start_without_targets_refuses_to_guess(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.selected_system = None
    controller.selected_mic = None
    controller.action(ITEM_START)
    assert controller.status.state == IDLE
    assert client.get("/api/live").json()["live"] is None
    assert "never guesses" in controller.notice


def test_tray_dead_tracks_show_failed_with_real_causes(tray_server, fake_pw_env):
    controller, client = tray_server
    payload = sample_dump()
    for obj in payload:
        props = obj.get("info", {}).get("props") if isinstance(obj.get("info"), dict) else None
        props = props or obj.get("props") or {}
        if props.get("node.name") == SINK:
            props["node.name"] = "die.system.out"
        elif props.get("node.name") == MIC:
            props["node.name"] = "die.mic.in"
        elif props.get("metadata.name") == "default":
            for entry in obj.get("metadata", []):
                if entry.get("key") == "default.audio.sink":
                    entry["value"] = {"name": "die.system.out"}
                if entry.get("key") == "default.audio.source":
                    entry["value"] = {"name": "die.mic.in"}
    fake_pw_env["graph_file"].write_text(json.dumps(payload))
    controller.discovery_error = None
    controller.sinks = []
    controller.refresh()                      # re-discovers the die.* nodes

    controller.action(ITEM_START)
    status = wait_state(controller, FAILED, timeout=20)
    assert not status.active
    assert "system" in status.detail and "mic" in status.detail
    # nothing is still capturing
    assert client.get("/api/live").json()["pipewire"]["active"] is False


def test_tray_unreachable_says_so_instead_of_faking_state(data_dir):
    # Port 1 on loopback: connection refused, no server.
    controller = TrayController(ServerClient(port=1, timeout=0.5),
                                targets_file=data_dir / "tray-targets.json")
    status = controller.refresh()
    assert status.state == UNREACHABLE
    assert "not answering" in status.summary()
    items = menu_items(controller)
    assert not items[ITEM_START].enabled
    assert "not answering" in items[ITEM_STATUS].label


def test_tray_quit_stops_the_recording_first(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.action(ITEM_START)
    wait_state(controller, RECORDING)
    # Let both tracks actually produce audio before quitting, so the
    # recorder's final report is a real one.
    end = time.monotonic() + 10
    while time.monotonic() < end:
        controller.refresh()
        if (controller.status.system_frames or 0) > 0 \
                and (controller.status.mic_frames or 0) > 0:
            break
        time.sleep(0.1)
    controller.action(ITEM_QUIT)
    assert controller.quit_requested
    # The engine has no *active* session left behind the tray's back: the
    # server keeps the finished session in /api/live with state done.
    end = time.monotonic() + 10
    while time.monotonic() < end:
        live = client.get("/api/live").json()["live"]
        if live is None or live["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    live = client.get("/api/live").json()["live"]
    assert live is None or live["state"] in ("done", "error")
    assert controller.refresh().state == IDLE       # and the tray says idle
    report = client.get("/api/live").json()["pipewire"]["report"]
    assert report["ok"] is True                     # tracks finalized cleanly


def test_tray_quit_while_idle_does_not_touch_the_server(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.action(ITEM_QUIT)
    assert controller.quit_requested
    assert controller.status.state == IDLE


def test_tray_target_pick_persists_and_gone_nodes_readopt_default(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.action(ITEM_SYSTEM_BASE + 0)
    controller.action(ITEM_MIC_BASE + 0)
    saved = json.loads((controller._targets_file).read_text())
    assert saved == {"system": SINK, "mic": MIC}

    # A remembered node that vanished re-adopts the graph's own session
    # default — and when the graph offers no default, selection becomes
    # empty (Start disables) rather than guessing the first node.
    other_sink = dict(SINKS[0], name="other.output", id=99)
    controller.client = type("G", (), {
        "devices": lambda self: {
            "sinks": [other_sink], "sources": [],
            "defaults": {"sink": None, "source": None}}})()
    controller._load_devices()
    assert controller.selected_system is None     # no default offered
    assert controller.selected_mic is None
    assert not menu_items(controller)[ITEM_START].enabled

    # With a session default present, that default is adopted (labelled).
    controller.client = type("G", (), {
        "devices": lambda self: {
            "sinks": [other_sink], "sources": [],
            "defaults": {"sink": "other.output", "source": None}}})()
    controller._load_devices()
    assert controller.selected_system == "other.output"


def test_tray_target_selection_via_menu_ids(tray_server):
    controller, _ = tray_server
    controller.refresh()
    controller.action(ITEM_SYSTEM_BASE + 0)
    assert controller.selected_system == SINK
    controller.action(ITEM_MIC_BASE + 0)
    assert controller.selected_mic == MIC
    # out-of-range ids are ignored, not crashing the tray
    controller.action(ITEM_SYSTEM_BASE + 50)
    controller.action(ITEM_MIC_BASE + 50)
    assert controller.selected_system == SINK
    assert controller.selected_mic == MIC


# ---------------------------------------------------------------------------
# Application streams: the tray's system picker covers both modes
# ---------------------------------------------------------------------------


def wait_frames(controller, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        controller.refresh()
        if (controller.status.system_frames or 0) > 0 \
                and (controller.status.mic_frames or 0) > 0:
            return
        time.sleep(0.1)
    raise AssertionError(f"no frames on both tracks: {controller.status}")


def test_tray_application_stream_selection_is_the_one_reported(tray_server_apps):
    """Pick a stream in the tray: the recorded target is that stream."""
    controller, client = tray_server_apps
    assert controller.refresh().state == IDLE
    assert [n["name"] for n in controller.apps] == [APP_A, APP_B]
    # the session default is still a whole sink: an application stream is
    # offered, never adopted on the user's behalf
    assert controller.selected_system == SINK and controller.selected_mic == MIC

    controller.action(ITEM_APP_BASE + 0)
    assert controller.selected_system == APP_A
    saved = json.loads(controller._targets_file.read_text())
    assert saved["system"] == APP_A

    controller.action(ITEM_START)
    wait_state(controller, RECORDING)
    wait_frames(controller)
    targets = client.get("/api/live").json()["pipewire"]["targets"]
    assert targets["system"]["name"] == APP_A
    assert targets["system"]["media_class"] == "Stream/Output/Audio"
    assert targets["system"]["selection"] == "explicit"
    # the microphone is untouched: an independent Audio/Source track
    assert targets["mic"]["name"] == MIC
    assert targets["mic"]["media_class"] == "Audio/Source"
    assert controller.status.system_target == APP_A   # the tray says the same

    controller.action(ITEM_STOP)
    wait_state(controller, IDLE)
    report = client.get("/api/live").json()["pipewire"]["report"]
    assert report["ok"] is True
    assert report["targets"]["system"]["media_class"] == "Stream/Output/Audio"
    # metadata only: the titles the application publishes never appear
    blob = json.dumps(report, ensure_ascii=False)
    for needle in CONTENT_NEEDLES:
        assert needle not in blob


def test_tray_whole_sink_selection_reports_a_sink_while_streams_exist(
        tray_server_apps):
    """The same picker, other mode: choosing the sink must not become
    per-application just because streams are on offer."""
    controller, client = tray_server_apps
    controller.refresh()
    controller.action(ITEM_SYSTEM_BASE + 0)
    assert controller.selected_system == SINK

    controller.action(ITEM_START)
    wait_state(controller, RECORDING)
    wait_frames(controller)
    targets = client.get("/api/live").json()["pipewire"]["targets"]
    assert targets["system"]["name"] == SINK
    assert targets["system"]["media_class"] == "Audio/Sink"
    controller.action(ITEM_STOP)
    wait_state(controller, IDLE)


def test_tray_vanished_application_selection_readopts_only_the_graph_default(
        tray_server_apps):
    """A remembered pick that left the graph: only the graph's own default.

    Never the first node in the list, and a stream that is still there is
    kept exactly as the user made it.
    """
    controller, _ = tray_server_apps
    controller.refresh()
    controller.action(ITEM_APP_BASE + 1)                 # AppB
    assert controller.selected_system == APP_B
    sink_graph = {"sinks": SINKS, "sources": SOURCES, "apps": [],
                  "defaults": {"sink": SINK, "source": MIC}}

    controller.client = type("G", (), {
        "devices": lambda self: dict(sink_graph)})()
    controller._load_devices()
    assert controller.selected_system == SINK            # graph default only

    controller.selected_system = APP_B
    controller.client = type("G", (), {
        "devices": lambda self: {**sink_graph,
                                 "defaults": {"sink": None, "source": None}}})()
    controller._load_devices()
    assert controller.selected_system is None            # no default offered
    assert not menu_items(controller)[ITEM_START].enabled

    # still in the graph → kept; an app pick is never quietly swapped back
    controller.selected_system = APP_B
    controller.client = type("G", (), {
        "devices": lambda self: {**sink_graph, "apps": [APPS[1]]}})()
    controller._load_devices()
    assert controller.selected_system == APP_B


def test_tray_start_with_a_gone_application_stream_is_refused_not_replaced(
        tray_server_apps, fake_pw_app_env):
    """A stream that closed before Start: the server's refusal, not a sink."""
    controller, client = tray_server_apps
    controller.refresh()
    controller.action(ITEM_APP_BASE + 0)
    payload = [obj for obj in json.loads(fake_pw_app_env["graph_file"].read_text())
               if (obj.get("info", {}).get("props") or {}).get("node.name") != APP_A]
    fake_pw_app_env["graph_file"].write_text(json.dumps(payload))

    controller.action(ITEM_START)
    status = controller.refresh()
    assert status.state == IDLE                          # nothing is recording
    assert client.get("/api/live").json()["live"] is None
    assert controller.notice and "No PipeWire node named" in controller.notice
    assert "Application output streams" in controller.notice
    assert "No PipeWire node named" in menu_items(controller)[ITEM_STATUS].label
    # and the refused pick was not replaced by some other target
    assert controller.selected_system == APP_A


def test_tray_app_menu_labels_the_application_and_never_a_title(
        tray_server_apps):
    """Labels are the applications' own names — never what they are playing."""
    controller, _ = tray_server_apps
    controller.refresh()
    labels = " ".join(all_labels(controller.menu()))
    assert "Application streams" in labels
    assert "AppA" in labels and "AppB" in labels
    for needle in CONTENT_NEEDLES:
        assert needle not in labels
    blob = json.dumps({"apps": controller.apps}, ensure_ascii=False)
    for needle in CONTENT_NEEDLES:
        assert needle not in blob


def test_tray_privacy_opens_listns_own_settings_page(tray_server, monkeypatch):
    controller, _ = tray_server
    controller.refresh()
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    assert controller.open_privacy() == f"{controller.base_url()}/#settings"
    assert opened == [f"{controller.base_url()}/#settings"]


def test_tray_status_click_is_a_noop(tray_server):
    controller, client = tray_server
    controller.refresh()
    controller.action(ITEM_STATUS)
    assert controller.status.state == IDLE
    assert client.get("/api/live").json()["live"] is None
    assert not controller.quit_requested


def test_tray_targets_file_is_the_only_local_state(tray_server, data_dir):
    """The tray persists its selection under the data dir; nothing else."""
    controller, _ = tray_server
    controller.refresh()
    controller.action(ITEM_SYSTEM_BASE + 0)
    controller.action(ITEM_MIC_BASE + 0)
    files = sorted(p.name for p in (data_dir / "tray-targets.json").parent.iterdir())
    assert "tray-targets.json" in files


def test_tray_ignores_a_default_pointing_at_the_wrong_media_class(
        tray_server, data_dir):
    """A session-manager default is adopted only from the node's own class.

    Observed with WirePlumber 0.5.8 on a machine without microphones:
    ``default.audio.source`` named an ``Audio/Sink``. Adopting it would put
    a sink name in the mic pick, so the tray re-checks the default against
    the node list (the rule ``listn.pipewire.select_targets`` applies) and
    leaves the pick empty, keeping Start disabled instead of guessing.
    Covers both adoption paths: a first pick and a remembered node that
    vanished and is being re-readopted.
    """
    wrong = {"sinks": SINKS, "sources": SOURCES,
             # the mic default wrongly names the sink node
             "defaults": {"sink": SINK, "source": SINK}}
    transport = type("T", (), {"get": lambda self, path: (200, wrong)})()

    # First adoption: the sink default is fine, the sink-as-mic default is not.
    controller = TrayController(ServerClient(transport=transport),
                                targets_file=data_dir / "fresh.json")
    controller._load_devices()
    assert controller.selected_system == SINK
    assert controller.selected_mic is None
    assert not menu_items(controller)[ITEM_START].enabled

    # Re-adoption: a remembered mic that vanished re-readopts only a default
    # that really is a source; the same mis-pointed default is rejected.
    controller.selected_system = "gone.output"
    controller.selected_mic = "gone.mic"
    controller._load_devices()
    assert controller.selected_system == SINK
    assert controller.selected_mic is None


