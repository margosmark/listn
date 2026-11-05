"""The real global-shortcut backend, against a fake portal on a private bus.

``listn/shortcut_portal.py`` talks to ``org.freedesktop.portal.Desktop``.
There is no portal on this machine, so these tests provide one: a private
``dbus-daemon --session`` and a ``ServiceInterface`` that answers
``CreateSession``/``BindShortcuts`` exactly as the portal reference says and
emits ``Activated`` back. That is a real D-Bus round trip through the real
client code — it proves the wiring, the signatures and the mapping of the
desktop's answer onto a status. It does **not** prove a Plasma panel
accepts a key press; that stays a manual test (docs/LINUX_DESKTOP.md).
"""

from __future__ import annotations

import asyncio
import shutil
import time

import pytest
from dbus_next import Message, Variant
from dbus_next.service import ServiceInterface, method

from listn.shortcut_portal import (DEST, IFACE, PATH, REQUEST_IFACE,
                                   SESSION_IFACE, PortalBackend,
                                   PortalUnavailable)
from listn.shortcuts import (ACTIVE, DECLINED, DEFAULT_BOOKMARK,
                             DEFAULT_START_STOP, PARTIAL, HotkeyActions,
                             ShortcutManager)
from listn.tray import (IDLE, ITEM_SHORTCUTS, ITEM_START, RECORDING,
                        TrayStatus, build_menu)

from .test_tray_sni import session_bus

pytestmark = pytest.mark.skipif(
    shutil.which("dbus-daemon") is None,
    reason="dbus-daemon is not installed")

SESSION_PATH = "/session/listn"


def emit(bus, path: str, interface: str, member: str, signature: str, body: list):
    """Send a raw signal: dbus-next has no MessageBus.emit()."""
    bus.send(Message.new_signal(path=path, interface=interface, member=member,
                                signature=signature, body=body))


class FakePortal(ServiceInterface):
    """Minimal GlobalShortcuts portal: grant, decline or drop bindings."""

    def __init__(self, bus, grant=None, decline=False):
        super().__init__(IFACE)
        self.bus = bus
        self.grant = grant            # None = grant every requested binding
        self.decline = decline
        self.bind_calls: list = []
        self.closed = False
        self._n = 0

    def _request_path(self) -> str:
        self._n += 1
        return f"{PATH}/request/test{self._n}/tok{self._n}"

    @method()
    def CreateSession(self, options: "a{sv}") -> "o":  # noqa: N802
        path = self._request_path()
        emit(self.bus, path, REQUEST_IFACE, "Response", "ua{sv}",
             [0, {"session_handle": Variant("s", SESSION_PATH)}])
        return path

    @method()
    def BindShortcuts(self, session_handle: "o", shortcuts: "a(sa{sv})",  # noqa: N802
                      parent_window: "s", options: "a{sv}") -> "o":
        path = self._request_path()
        self.bind_calls.append([(sid, dict(props)) for sid, props in shortcuts])
        granted = []
        if not self.decline:
            for sid, props in shortcuts:
                if self.grant is not None and sid not in self.grant:
                    continue
                trigger = props.get("preferred_trigger")
                granted.append([sid, {
                    "trigger_description": Variant(
                        "s", str(getattr(trigger, "value", trigger)))}])
        emit(self.bus, path, REQUEST_IFACE, "Response", "ua{sv}",
             [0, {"shortcuts": Variant("a(sa{sv})", granted)}])
        return path


class FakeSession(ServiceInterface):
    def __init__(self):
        super().__init__(SESSION_IFACE)
        self.closed = False

    @method()
    def Close(self):  # noqa: N802
        self.closed = True


def run(coro, timeout=15.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


async def _with_portal(grant=None, decline=False):
    """Export the fake portal and return (backend, fake, session)."""
    from dbus_next.aio import MessageBus

    bus = await MessageBus().connect()
    portal = FakePortal(bus, grant=grant, decline=decline)
    session = FakeSession()
    bus.export(PATH, portal)
    bus.export(SESSION_PATH, session)
    await bus.request_name(DEST)
    backend = PortalBackend()
    return backend, portal, session, bus


async def _teardown(backend, bus):
    try:
        await backend.close()
    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


def wait_for(predicate, timeout=5.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# Real client, real bus, fake portal
# ---------------------------------------------------------------------------

def test_registers_both_shortcuts_and_uses_the_desktops_triggers(session_bus):
    async def scenario():
        backend, portal, session, bus = await _with_portal()
        try:
            got = await backend.bind(
                {"start_stop": DEFAULT_START_STOP, "bookmark": DEFAULT_BOOKMARK},
                lambda action: None)
            return got, portal
        finally:
            await _teardown(backend, bus)

    got, portal = run(scenario())
    assert got["start_stop"] == (True, "Ctrl+Alt+R")
    assert got["bookmark"] == (True, "Ctrl+Alt+B")
    # The preferred triggers travelled over the wire, not a hard-coded guess.
    preferred = {sid: props["preferred_trigger"].value
                 for sid, props in portal.bind_calls[-1]}
    assert preferred == {"start_stop": DEFAULT_START_STOP,
                         "bookmark": DEFAULT_BOOKMARK}


def test_an_activated_signal_runs_the_action(session_bus):
    pressed: list[str] = []

    async def scenario():
        backend, portal, session, bus = await _with_portal()
        try:
            # The backend hands the shortcut id to its callback; the manager
            # maps id -> action, so receiving the id here is the contract.
            await backend.bind({"start_stop": DEFAULT_START_STOP,
                                "bookmark": DEFAULT_BOOKMARK},
                               pressed.append)
            # Activated is (o session, s shortcut_id, t timestamp, a{sv}
            # options) per the portal reference XML.
            emit(portal.bus, PATH, IFACE, "Activated", "osta{sv}",
                 [SESSION_PATH, "start_stop", 1234, {}])
            await asyncio.sleep(0.1)
        finally:
            await _teardown(backend, bus)

    run(scenario())
    assert pressed == ["start_stop"]


def test_a_declined_bind_is_reported_not_assumed(session_bus):
    async def scenario():
        backend, portal, session, bus = await _with_portal(decline=True)
        try:
            return await backend.bind(
                {"start_stop": DEFAULT_START_STOP, "bookmark": DEFAULT_BOOKMARK},
                lambda action: None)
        finally:
            await _teardown(backend, bus)

    got = run(scenario())
    assert got["start_stop"][0] is False
    assert got["bookmark"][0] is False


def test_a_partial_grant_keeps_the_two_answers_apart(session_bus):
    async def scenario():
        backend, portal, session, bus = await _with_portal(grant={"start_stop"})
        try:
            return await backend.bind(
                {"start_stop": DEFAULT_START_STOP, "bookmark": DEFAULT_BOOKMARK},
                lambda action: None)
        finally:
            await _teardown(backend, bus)

    got = run(scenario())
    assert got["start_stop"][0] is True
    assert got["bookmark"][0] is False


def test_closing_the_backend_closes_the_portal_session(session_bus):
    async def scenario():
        backend, portal, session, bus = await _with_portal()
        await backend.bind({"start_stop": DEFAULT_START_STOP,
                            "bookmark": DEFAULT_BOOKMARK}, lambda a: None)
        await backend.close()
        closed = session.closed
        await asyncio.sleep(0.05)
        bus.disconnect()
        return closed

    assert run(scenario()) is True


# ---------------------------------------------------------------------------
# No portal at all: the reason, not a false success
# ---------------------------------------------------------------------------

def test_no_portal_is_an_unavailable_backend_with_a_reason(session_bus):
    async def scenario():
        backend = PortalBackend()
        try:
            await backend.bind({"start_stop": DEFAULT_START_STOP,
                                "bookmark": DEFAULT_BOOKMARK}, lambda a: None)
        except PortalUnavailable as exc:
            return str(exc)
        finally:
            await backend.close()
        return None

    reason = run(scenario())
    assert reason, "a bus without the portal must fail loudly"
    assert "xdg-desktop-portal" in reason
    assert "GlobalShortcuts" in reason or "session bus" in reason


def test_manager_turns_a_missing_portal_into_a_visible_status(session_bus):
    async def scenario():
        shortcuts = ShortcutManager(HotkeyActions(_Controller()),
                                    {"start_stop": DEFAULT_START_STOP,
                                     "bookmark": DEFAULT_BOOKMARK},
                                    backend_factory=PortalBackend)
        status = await shortcuts.start()
        await shortcuts.close()
        return status

    status = run(scenario())
    assert status.state == "unavailable"
    assert "xdg-desktop-portal" in status.reason
    line = status.summary()
    assert "unavailable" in line and "xdg-desktop-portal" in line
    # And the tray still offers its own controls.
    items = {i.id: i for i in build_menu(TrayStatus(state=RECORDING), [], [],
                                         None, None, shortcut_status=status)}
    assert items[ITEM_START].enabled is False
    assert items[ITEM_SHORTCUTS].enabled is False
    assert "xdg-desktop-portal" in items[ITEM_SHORTCUTS].label


class _Controller:
    """Only what HotkeyActions touches."""

    def __init__(self):
        self.status = TrayStatus(state=IDLE)
        self.calls: list = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def bookmark(self, note="", source="hotkey"):
        self.calls.append(("bookmark", source))


def test_manager_end_to_end_through_the_real_backend(session_bus):
    controller = _Controller()

    async def scenario():
        # The manager builds its own PortalBackend, so the fake portal must
        # already own org.freedesktop.portal.Desktop on this session bus.
        _backend, _portal, _session, bus = await _with_portal()
        try:
            shortcuts = ShortcutManager(HotkeyActions(controller),
                                        {"start_stop": DEFAULT_START_STOP,
                                         "bookmark": DEFAULT_BOOKMARK},
                                        backend_factory=PortalBackend)
            status = await shortcuts.start()
            await shortcuts.close()
            return status
        finally:
            try:
                bus.disconnect()
            except Exception:
                pass

    status = run(scenario())
    assert status.state == ACTIVE
    assert status.backend == "freedesktop-globalshortcuts"
    assert status.ok
