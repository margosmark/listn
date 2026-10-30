"""KDE Plasma tray renderer: StatusNotifierItem + dbusmenu over the session bus.

Plasma's tray protocol is D-Bus, not Qt: an app exports
``org.kde.StatusNotifierItem`` (icon, title, tooltip) and
``com.canonical.dbusmenu`` (the menu), then registers with
``org.kde.StatusNotifierWatcher``. This module does exactly that with
``dbus-next`` — no Qt dependency — and is a *renderer* of
:class:`listn.tray.TrayController`: every label, icon and enabled flag comes
from the controller's derived status, so what the tray shows is only ever
what the server reported.

Local-first: everything here talks to the session bus and loopback HTTP.
No sockets leave this machine.

Honest limitations (also in the docs): properties, menu layout and item
clicks are verified against a real session bus; interaction with a running
plasmashell panel cannot be exercised on a headless machine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from listn.tray import (
    FAILED, IDLE, ITEM_STATUS, PAUSED, POLL_SECONDS, RECORDING, STARTING,
    STOPPING, TrayController, MenuItem, UNREACHABLE,
)

log = logging.getLogger("listn.tray")

# Icon names from the freedesktop icon theme (Breeze ships all of these).
ICONS = {
    IDLE: "audio-input-microphone",
    RECORDING: "media-record",
    PAUSED: "media-playback-pause",
    STARTING: "view-refresh",
    STOPPING: "view-refresh",
    FAILED: "dialog-error",
    UNREACHABLE: "dialog-warning",
}
WATCHER_NAME = "org.kde.StatusNotifierWatcher"
WATCHER_PATH = "/StatusNotifierWatcher"
RETRY_REGISTER_SECONDS = 10.0

# dbusmenu property name -> D-Bus signature (everything else is "s").
_SIG_OF = {"enabled": "b", "toggle-state": "i"}


def service_name() -> str:
    return f"org.kde.StatusNotifierItem-{os.getpid()}-1"


def _menu_node(item: MenuItem, names: dict[int, str]) -> list:
    """One dbusmenu node: [id, props, children] with Variant-typed props."""
    props: dict = {}
    if item.separator:
        props["type"] = "separator"
    else:
        props["label"] = item.label
        props["enabled"] = bool(item.enabled)
        if item.toggle:
            props["toggle-type"] = item.toggle
            props["toggle-state"] = 1 if item.toggled else 0
        if item.children:
            props["children-display"] = "submenu"
    children = [_menu_node(child, names) for child in item.children]
    names[item.id] = item.label
    return [item.id, props, children]


def build_layout(items: list[MenuItem]) -> tuple[dict[int, str], list]:
    """(id -> label map, root node) for the menu; root id is 0 per spec."""
    names: dict[int, str] = {}
    root = [0, {"children-display": "submenu"},
            [_menu_node(item, names) for item in items]]
    return names, root


def _typed_props(props: dict) -> dict:
    from dbus_next import Variant

    return {key: Variant(_SIG_OF.get(key, "s"), value)
            for key, value in props.items()}


def _struct_variants(node: list) -> list:
    """(ia{sv}av) struct body: the children slot is an array of Variants,
    and every child is the same shape again — at every nesting level."""
    from dbus_next import Variant

    node_id, props, children = node
    return [node_id, _typed_props(props),
            [Variant("(ia{sv}av)", _struct_variants(c)) for c in children]]


def _walk(items: list[MenuItem]):
    for item in items:
        yield item
        yield from _walk(item.children)


def _flatten(node: list) -> list:
    """Every node of a layout tree, depth-first."""
    out = [node]
    for child in node[2]:
        out.extend(_flatten(child))
    return out


class StatusNotifierItem:
    """Owns the bus name, the two exported interfaces and the poll loop."""

    def __init__(self, controller: TrayController):
        self.controller = controller
        self.revision = 1
        self.registered = False
        self._last_render: tuple | None = None
        self._last_register_attempt = 0.0
        self._quit = asyncio.Event()
        self._sni = None
        self._menu = None
        self._bus = None

    # -- interface definitions (built lazily so import needs dbus-next) ----

    def _build_interfaces(self):
        from dbus_next import Variant
        from dbus_next.constants import PropertyAccess
        from dbus_next.service import (
            ServiceInterface, dbus_property, method, signal,
        )

        controller = self.controller
        item = self

        class SNI(ServiceInterface):
            def __init__(self):
                super().__init__("org.kde.StatusNotifierItem")

            @dbus_property(access=PropertyAccess.READ)
            def Category(self) -> "s":  # noqa: N802
                return "ApplicationStatus"

            @dbus_property(access=PropertyAccess.READ)
            def Id(self) -> "s":  # noqa: N802
                return "listn"

            @dbus_property(access=PropertyAccess.READ)
            def Title(self) -> "s":  # noqa: N802
                return "listn"

            @dbus_property(access=PropertyAccess.READ)
            def Status(self) -> "s":  # noqa: N802
                return "NeedsAttention" if controller.status.state == FAILED else "Active"

            @dbus_property(access=PropertyAccess.READ)
            def IconName(self) -> "s":  # noqa: N802
                return ICONS.get(controller.status.state, "audio-input-microphone")

            @dbus_property(access=PropertyAccess.READ)
            def AttentionIconName(self) -> "s":  # noqa: N802
                return "dialog-error"

            @dbus_property(access=PropertyAccess.READ)
            def IconThemePath(self) -> "s":  # noqa: N802
                return ""

            @dbus_property(access=PropertyAccess.READ)
            def Menu(self) -> "o":  # noqa: N802
                return "/Menu"

            @dbus_property(access=PropertyAccess.READ)
            def ItemIsMenu(self) -> "b":  # noqa: N802
                # True: hosts open the exported menu on activation instead of
                # expecting this process to pop its own widget.
                return True

            @dbus_property(access=PropertyAccess.READ)
            def ToolTip(self) -> "(sa(iiay)ss)":  # noqa: N802
                state = controller.status.state
                title = "listn — " + state.replace("-", " ").title()
                # (icon name, pixmap array, title, description): no custom
                # pixmap — hosts resolve IconName from the icon theme.
                return [ICONS.get(state, "audio-input-microphone"),
                        [], title, controller.status.summary()]

            @method()
            def Activate(self, x: "i", y: "i"):  # noqa: N802
                return None

            @method()
            def ContextMenu(self, x: "i", y: "i"):  # noqa: N802
                return None

            @method()
            def SecondaryActivate(self, x: "i", y: "i"):  # noqa: N802
                return None

            @method()
            def Scroll(self, delta: "i", orientation: "s"):  # noqa: N802
                return None

            @signal()
            def NewTitle(self):  # noqa: N802
                return None

            @signal()
            def NewIcon(self):  # noqa: N802
                return None

            @signal()
            def NewStatus(self, status: "s") -> "s":  # noqa: N802
                return status

            @signal()
            def NewToolTip(self):  # noqa: N802
                return None

        class DBusMenu(ServiceInterface):
            def __init__(self):
                super().__init__("com.canonical.dbusmenu")

            @dbus_property(access=PropertyAccess.READ)
            def Version(self) -> "u":  # noqa: N802
                return 3

            @dbus_property(access=PropertyAccess.READ)
            def TextDirection(self) -> "s":  # noqa: N802
                return "ltr"

            @dbus_property(access=PropertyAccess.READ)
            def Status(self) -> "s":  # noqa: N802
                return "normal"

            @dbus_property(access=PropertyAccess.READ)
            def IconThemePath(self) -> "as":  # noqa: N802
                return []

            @method()
            def GetLayout(self, parent_id: "i", recursion_depth: "i",
                          property_names: "as") -> "u(ia{sv}av)":
                # (revision, root struct): the struct's children slot is av,
                # so each child is a Variant; the root struct itself is not.
                return [item.revision, _struct_variants(item.current_root())]

            @method()
            def GetGroupProperties(self, ids: "ai",
                                   property_names: "as") -> "a(ia{sv})":
                found = {node[0]: node for node in _flatten(item.current_root())}
                out = []
                for wanted in ids:
                    node = found.get(wanted)
                    if node is None:
                        continue
                    out.append([wanted, _typed_props(node[1])])
                return out

            @method()
            def GetProperty(self, id: "i", name: "s") -> "v":
                return Variant("s", "")

            @method()
            def Event(self, id: "i", event_id: "s", data: "v",
                      timestamp: "u"):
                if event_id == "clicked" and id != ITEM_STATUS:
                    item.on_click(id)

            @method()
            def EventGroup(self, events: "a(isvu)") -> "ai":
                handled = []
                for entry, event_id, data, stamp in events:
                    if event_id == "clicked" and entry != ITEM_STATUS:
                        item.on_click(entry)
                        handled.append(entry)
                return handled

            @method()
            def AboutToShow(self, id: "i") -> "b":
                return False

            @method()
            def AboutToShowGroup(self, ids: "ai") -> "aiai":
                return [[], []]

            @signal()
            def LayoutUpdated(self, revision: "u", parent: "i") -> "ui":
                return [revision, parent]

        def current_root():
            names, root = build_layout(controller.menu())
            self.names = names
            return root

        item.names = {}
        item.current_root = current_root
        item._SNI, item._Menu = SNI, DBusMenu
        return SNI, DBusMenu

    # -- callbacks from the menu -------------------------------------------

    def on_click(self, item_id: int) -> None:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._safe_action, item_id)

    def _safe_action(self, item_id: int) -> None:
        try:
            self.controller.action(item_id)
        except Exception:  # a tray click must never take the app down
            log.exception("tray action %s failed", item_id)
        if self.controller.quit_requested:
            self._quit.set()

    # -- rendering ---------------------------------------------------------

    def render(self) -> bool:
        """Push controller state to the bus. Returns True if anything changed."""
        status = self.controller.status
        menu = self.controller.menu()
        snapshot = (
            status.state, status.summary(),
            status.elapsed and int(status.elapsed),
            tuple((i.id, i.label, i.enabled, i.toggled) for i in _walk(menu)),
        )
        if snapshot == self._last_render:
            return False
        had = self._last_render is not None
        self._last_render = snapshot
        if self._menu is None:
            return True
        if had:
            self.revision += 1
            self._menu.LayoutUpdated(self.revision, 0)
        self._sni.NewTitle()
        self._sni.NewIcon()
        self._sni.NewToolTip()
        self._sni.NewStatus(self._sni.Status)
        self._sni.emit_properties_changed({
            "Title": "listn",
            "IconName": self._sni.IconName,
            "Status": self._sni.Status,
            "ToolTip": list(self._sni.ToolTip),
        })
        return True

    # -- registration ------------------------------------------------------

    async def _register(self, bus) -> None:
        if self.registered:
            return
        now = time.monotonic()
        if now - self._last_register_attempt < RETRY_REGISTER_SECONDS:
            return
        self._last_register_attempt = now
        from dbus_next import DBusError, Message

        try:
            reply = await bus.call(Message(
                destination=WATCHER_NAME, path=WATCHER_PATH,
                interface=WATCHER_NAME, member="RegisterStatusNotifierItem",
                signature="s", body=[service_name()]))
        except DBusError as exc:
            # No plasmashell yet (headless, or tray started before the
            # desktop): say so once and keep retrying, never crash.
            log.warning("StatusNotifierWatcher not available: %s", exc.text)
            return
        if reply is None:
            log.warning("StatusNotifierWatcher did not answer; "
                        "will retry (is Plasma's system tray running?)")
            return
        self.registered = True
        log.info("tray registered with the Plasma system tray")

    # -- main loop ---------------------------------------------------------

    async def run(self) -> None:
        from dbus_next.aio import MessageBus

        self._build_interfaces()
        self._bus = await MessageBus().connect()
        self._sni = self._SNI()
        self._menu = self._Menu()
        self._bus.export("/StatusNotifierItem", self._sni)
        self._bus.export("/Menu", self._menu)
        await self._bus.request_name(service_name())
        await self._register(self._bus)
        self.render()

        loop = asyncio.get_running_loop()

        async def tick():
            while not self._quit.is_set():
                await loop.run_in_executor(None, self._safe_refresh)
                await self._register(self._bus)
                self.render()
                if self.controller.quit_requested:
                    self._quit.set()
                    return
                try:
                    await asyncio.wait_for(self._quit.wait(),
                                           timeout=POLL_SECONDS)
                except asyncio.TimeoutError:
                    continue

        poller = asyncio.ensure_future(tick())
        await self._quit.wait()
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass
        self._bus.disconnect()

    def _safe_refresh(self) -> None:
        try:
            self.controller.refresh()
        except Exception:  # polling must survive a misbehaving server
            log.exception("tray refresh failed")

    def request_quit(self) -> None:
        self._quit.set()
