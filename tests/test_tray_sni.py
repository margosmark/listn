"""StatusNotifierItem renderer tests: a real session bus, a real engine.

`listn/tray_sni.py` is the only module that talks D-Bus, so these tests
give it the real thing: a private ``dbus-daemon --session``, a uvicorn
thread with the stand-in PipeWire tools (real HTTP, real engine, real
recorder subprocesses), and a *client* bus that drives the exported item
exactly like Plasma would — property reads, ``com.canonical.dbusmenu``
layout + ``Event(clicked)``, and ``org.kde.StatusNotifierWatcher``
registration.

Honest limitation (documented in the module): a running plasmashell panel
cannot be exercised on a headless machine; everything below the panel —
the whole D-Bus contract — is verified here.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("dbus_next")

import uvicorn  # noqa: E402
from dbus_next import Message, Variant  # noqa: E402
from dbus_next.aio import MessageBus  # noqa: E402
from dbus_next.service import ServiceInterface, method  # noqa: E402

from listn.engine import Engine  # noqa: E402
from listn.server import create_app  # noqa: E402
from listn.tray import (  # noqa: E402
    ITEM_PAUSE, ITEM_QUIT, ITEM_RESUME, ITEM_START, ITEM_STATUS, ITEM_STOP,
    ServerClient, TrayController,
)
from listn.tray_sni import (  # noqa: E402
    StatusNotifierItem, WATCHER_NAME, WATCHER_PATH, service_name,
)

from .conftest import MAC_DEVICES, FakeSD  # noqa: E402
from .test_pipewire import sample_dump  # noqa: E402
from .test_pipewire_wiring import MIC, SINK, fake_pw_env  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("dbus-daemon") is None,
    reason="dbus-daemon is not installed")

SNI = "org.kde.StatusNotifierItem"
DBUSMENU = "com.canonical.dbusmenu"
PROPS = "org.freedesktop.DBus.Properties"


# ---------------------------------------------------------------------------
# Fixtures: private session bus + real server on loopback
# ---------------------------------------------------------------------------


@pytest.fixture
def session_bus(monkeypatch):
    proc = subprocess.Popen(
        ["dbus-daemon", "--session", "--print-address", "--nofork"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    address = proc.stdout.readline().decode().strip()
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", address)
    try:
        yield address
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture
def live_server(data_dir, fake_factory, fake_pw_env):
    """Real uvicorn + engine + stand-in PipeWire tools on a free port."""
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning", timeout_graceful_shutdown=3)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="tray-sni-server",
                              daemon=True)
    thread.start()
    end = time.monotonic() + 15
    while time.monotonic() < end and not server.started:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    try:
        yield port, engine, app
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if engine.live is not None:
            end = time.monotonic() + 15
            while time.monotonic() < end \
                    and engine.live.state not in ("done", "error"):
                time.sleep(0.05)
            time.sleep(0.2)   # trailing retention/notes writes


class FakeWatcher(ServiceInterface):
    """Minimal org.kde.StatusNotifierWatcher recording registrations."""

    def __init__(self):
        super().__init__(WATCHER_NAME)
        self.registrations: list[str] = []

    @method()
    def RegisterStatusNotifierItem(self, service: "s"):  # noqa: N802
        self.registrations.append(service)


# ---------------------------------------------------------------------------
# Client-side helpers (what a Plasma host does)
# ---------------------------------------------------------------------------


async def get_prop(bus, name, prop, path="/StatusNotifierItem",
                   interface=SNI):
    reply = await bus.call(Message(
        destination=name, path=path, interface=PROPS, member="Get",
        signature="ss", body=[interface, prop]))
    assert reply is not None and reply.message_type != 3, \
        f"Properties.Get({prop}) failed: {reply}"
    return reply.body[0].value


async def get_layout(bus, name):
    reply = await bus.call(Message(
        destination=name, path="/Menu", interface=DBUSMENU,
        member="GetLayout", signature="iias", body=[0, -1, []]))
    assert reply is not None and reply.message_type != 3, \
        f"GetLayout failed: {reply}"
    return reply.body[1]      # [id, props, [Variant children...]]


async def click(bus, name, item_id):
    await bus.call(Message(
        destination=name, path="/Menu", interface=DBUSMENU, member="Event",
        signature="isvu", body=[item_id, "clicked", Variant("i", 1), 0]))


def node_labels(root) -> list[str]:
    """Labels of a whole dbusmenu layout tree."""
    _, props, children = root
    out = []
    if "label" in props:
        out.append(props["label"].value)
    for child in children:
        out.extend(node_labels(child.value))
    return out


async def wait_prop(bus, name, prop, want, timeout=20, path="/StatusNotifierItem",
                    interface=SNI):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        last = await get_prop(bus, name, prop, path=path, interface=interface)
        if last == want:
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"property {prop} never became {want!r}; last={last!r}")


def api(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=5) as resp:
        return json.loads(resp.read())


async def wait_frames(port, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        pw = api(port, "/api/live").get("pipewire") or {}
        tracks = pw.get("tracks") or {}
        if all((tracks.get(n) or {}).get("frames", 0) > 0
               for n in ("system", "mic")):
            return tracks
        await asyncio.sleep(0.1)
    raise AssertionError("both tracks never produced frames")


async def run_tray(port, controller_ready=None):
    """Start the SNI run loop; return (sni, task, watcher, watcher_bus)."""
    watcher_bus = await MessageBus().connect()
    watcher = FakeWatcher()
    watcher_bus.export(WATCHER_PATH, watcher)
    await watcher_bus.request_name(WATCHER_NAME)

    controller = TrayController(ServerClient(port=port))
    if controller_ready is not None:
        controller_ready(controller)
    sni = StatusNotifierItem(controller)
    task = asyncio.create_task(sni.run())
    name = service_name()
    # wait for the item to own its bus name
    end = time.monotonic() + 10
    while time.monotonic() < end:
        reply = await watcher_bus.call(Message(
            destination="org.freedesktop.DBus", path="/org/freedesktop/DBus",
            interface="org.freedesktop.DBus", member="GetNameOwner",
            signature="s", body=[name]))
        if reply is not None and reply.body and reply.body[0]:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("tray never acquired its bus name")
    return sni, task, watcher, watcher_bus


async def shutdown(sni, task, watcher_bus, timeout=20):
    sni.request_quit()
    try:
        await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        task.cancel()
        raise AssertionError("tray run loop did not exit after quit")
    watcher_bus.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sni_registers_with_the_watcher(session_bus, live_server):
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        try:
            end = time.monotonic() + 10
            while time.monotonic() < end and not watcher.registrations:
                await asyncio.sleep(0.05)
            assert watcher.registrations == [service_name()]
        finally:
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_idle_properties_and_menu(session_bus, live_server):
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            assert await get_prop(client, name, "Title") == "listn"
            assert await get_prop(client, name, "Id") == "listn"
            assert await get_prop(client, name, "Status") == "Active"
            assert await get_prop(client, name, "ItemIsMenu") is True
            tooltip = await get_prop(client, name, "ToolTip")
            icon, _pixmaps, title, description = tooltip
            assert icon == "audio-input-microphone"
            assert "Idle" in description and "Nothing is recording" in description

            labels = node_labels(await get_layout(client, name))
            for wanted in ("Start recording", "Stop", "Pause", "Resume",
                           "System output", "Microphone",
                           "Open privacy settings", "Quit listn"):
                assert wanted in labels, labels
            joined = " ".join(labels).lower()
            for forbidden in ("meeting", "auto-record", "launch at login",
                              "per-app", "per app"):
                assert forbidden not in joined, labels
            # the status line is a disabled label, not a control
            reply = await client.call(Message(
                destination=name, path="/Menu", interface=DBUSMENU,
                member="GetGroupProperties", signature="aias",
                body=[[ITEM_STATUS], []]))
            props = reply.body[0][0][1]
            assert props["enabled"].value is False
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_starts_a_real_two_track_recording_and_stops(session_bus,
                                                         live_server):
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")

            await click(client, name, ITEM_START)
            await wait_prop(client, name, "IconName", "media-record")
            tooltip = await get_prop(client, name, "ToolTip")
            assert "Recording" in tooltip[3], tooltip

            # the click really started the recorder — frames on both tracks
            tracks = await wait_frames(port)
            assert tracks["system"]["frames"] > 0
            assert tracks["mic"]["frames"] > 0
            assert engine.live_active()

            await click(client, name, ITEM_PAUSE)
            await wait_prop(client, name, "IconName", "media-playback-pause")
            tooltip = await get_prop(client, name, "ToolTip")
            assert "Paused" in tooltip[3], tooltip

            await click(client, name, ITEM_RESUME)
            await wait_prop(client, name, "IconName", "media-record")

            await click(client, name, ITEM_STOP)
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            report = api(port, "/api/live")["pipewire"]["report"]
            assert report["ok"] is True, report
            # stop is asynchronous (wait=False on the HTTP path): the
            # session finalizes on its own thread — wait for it
            end = time.monotonic() + 10
            while time.monotonic() < end and engine.live_active():
                await asyncio.sleep(0.1)
            assert not engine.live_active()
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_menu_layout_signals_on_state_change(session_bus, live_server):
    """Plasma is told to re-fetch the menu when the state changes."""
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            await client.call(Message(
                destination="org.freedesktop.DBus",
                path="/org/freedesktop/DBus", interface="org.freedesktop.DBus",
                member="AddMatch", signature="s",
                body=[f"type='signal',interface='{DBUSMENU}',"
                      f"member='LayoutUpdated',sender='{name}'"]))
            seen = []
            done = asyncio.Event()

            def handler(msg):
                if msg.member == "LayoutUpdated":
                    seen.append(msg.body)
                    done.set()

            client.add_message_handler(handler)
            await click(client, name, ITEM_START)
            await wait_prop(client, name, "IconName", "media-record")
            try:
                await asyncio.wait_for(done.wait(), 10)
            except asyncio.TimeoutError:
                raise AssertionError("no LayoutUpdated after a state change")
            assert seen[0][1] == 0              # parent id is the root
            labels = node_labels(await get_layout(client, name))
            # the menu always lists the four controls; the state change
            # must be visible in their enabled flags (Pause on, Resume off)
            reply = await client.call(Message(
                destination=name, path="/Menu", interface=DBUSMENU,
                member="GetGroupProperties", signature="aias",
                body=[[ITEM_PAUSE, ITEM_RESUME, ITEM_STOP], []]))
            props = {row[0]: row[1] for row in reply.body[0]}
            assert props[ITEM_PAUSE]["enabled"].value is True
            assert props[ITEM_RESUME]["enabled"].value is False
            assert props[ITEM_STOP]["enabled"].value is True
            await click(client, name, ITEM_STOP)
            await wait_prop(client, name, "IconName", "audio-input-microphone")
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_failed_capture_shows_the_recorders_cause(session_bus,
                                                      live_server,
                                                      fake_pw_env):
    port, engine, app = live_server
    payload = sample_dump()
    for obj in payload:
        props = obj.get("info", {}).get("props") \
            if isinstance(obj.get("info"), dict) else None
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

    async def scenario():
        # discovery must see the rewritten graph before the click
        def ready(controller):
            pass

        sni, task, watcher, watcher_bus = await run_tray(port, ready)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            # the tray re-discovers on its own poll; wait for the die.* nodes
            end = time.monotonic() + 10
            while time.monotonic() < end:
                if sni.controller.selected_system == "die.system.out":
                    break
                await asyncio.sleep(0.1)
            assert sni.controller.selected_system == "die.system.out", \
                sni.controller.selected_system

            await click(client, name, ITEM_START)
            # both pw-record processes die; the monitor fails the capture
            # and the tray must show the failure, not silence or staleness
            await wait_prop(client, name, "IconName", "dialog-error",
                            timeout=25)
            await wait_prop(client, name, "Status", "NeedsAttention",
                            timeout=5)
            tooltip = await get_prop(client, name, "ToolTip")
            assert tooltip[0] == "dialog-error"
            assert "Failed" in tooltip[3], tooltip
            assert "system" in tooltip[3], tooltip
            live = api(port, "/api/live")
            assert live["pipewire"]["active"] is False
            assert live["pipewire"]["state"] == "failed"
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_quit_while_recording_stops_the_capture(session_bus,
                                                    live_server):
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            await click(client, name, ITEM_START)
            await wait_prop(client, name, "IconName", "media-record")
            await wait_frames(port)          # a real capture before quitting

            await click(client, name, ITEM_QUIT)
            await asyncio.wait_for(task, 20)  # the run loop exits on quit
            watcher_bus.disconnect()
            client.disconnect()

            # quitting left no recording running behind the tray
            end = time.monotonic() + 10
            while time.monotonic() < end:
                live = api(port, "/api/live").get("live")
                if live is None or live["state"] in ("done", "error"):
                    break
                await asyncio.sleep(0.1)
            live = api(port, "/api/live").get("live")
            assert live is None or live["state"] in ("done", "error")
            report = api(port, "/api/live")["pipewire"]["report"]
            assert report["ok"] is True, report
            return
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
            if not task.done():
                await shutdown(sni, task, watcher_bus)
            else:
                watcher_bus.disconnect()

    asyncio.run(scenario())


def test_sni_status_item_click_is_ignored(session_bus, live_server):
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            await click(client, name, ITEM_STATUS)
            await asyncio.sleep(1.0)          # one poll cycle
            assert api(port, "/api/live")["live"] is None
            assert await get_prop(client, name, "IconName") == \
                "audio-input-microphone"
            assert not sni.controller.quit_requested
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())


def test_sni_survives_a_dead_server_and_says_so(session_bus, live_server):
    """A dead engine must flip the tray to unreachable, not stale state."""
    port, engine, app = live_server

    async def scenario():
        sni, task, watcher, watcher_bus = await run_tray(port)
        client = await MessageBus().connect()
        try:
            name = service_name()
            await wait_prop(client, name, "IconName", "audio-input-microphone")
            # the engine goes away (point the client at a dead port)
            sni.controller.client = ServerClient(port=1, timeout=0.5)
            await wait_prop(client, name, "IconName", "dialog-warning")
            tooltip = await get_prop(client, name, "ToolTip")
            assert "not answering" in tooltip[3], tooltip
        finally:
            client.disconnect()
            await shutdown(sni, task, watcher_bus)

    asyncio.run(scenario())
