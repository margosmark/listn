"""Registration backend: the freedesktop ``GlobalShortcuts`` portal.

Why this one: it is the only global-shortcut API that works the same way
on X11 and Wayland, needs no X server, no ``python-xlib`` and no
per-desktop guesswork, and it is the mechanism the desktop itself owns —
which is exactly what makes its answer trustworthy. The trade-off is
honest and visible: if the session has no ``xdg-desktopportal`` with this
portal (older Plasma, a minimal window manager, a headless box), nothing
can be registered, and the caller reports that reason instead of pretending
a key works.

The flow, per the portal reference
(``org.freedesktop.portal.GlobalShortcuts``, interface version 2):

1. ``CreateSession`` -> a request handle; the session handle arrives on
   ``org.freedesktop.portal.Request::Response``.
2. ``BindShortcuts`` with the preferred triggers; the granted set arrives
   on another ``Response``. The result may be a *subset* — that is the
   desktop's conflict/decline answer, and it is reported per binding.
3. ``Activated`` carries the key presses back.

Everything here talks to the session bus only.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

log = logging.getLogger("listn.shortcuts")

DEST = "org.freedesktop.portal.Desktop"
PATH = "/org/freedesktop/portal/desktop"
IFACE = "org.freedesktop.portal.GlobalShortcuts"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"
REQUEST_PREFIX = "/org/freedesktop/portal/desktop/request/"
RESPONSE_TIMEOUT = 60.0   # a dialog may wait for the user

# The session bus only delivers a signal to a connection that asked for
# it, so the two signal families the portal uses must be matched explicitly
# before CreateSession is sent — otherwise every Response arrives at a bus
# that is not listening and the caller waits forever for an answer that was
# already given.
MATCH_RULES = (
    "type='signal',interface='org.freedesktop.portal.Request'",
    "type='signal',interface='org.freedesktop.portal.GlobalShortcuts'",
)

DESCRIPTIONS = {
    "start_stop": "listn: start or stop recording",
    "bookmark": "listn: bookmark this moment",
}


class PortalUnavailable(RuntimeError):
    """The portal cannot register shortcuts; ``str`` is the user-facing reason."""


def _reason_for(exc: BaseException) -> str:
    """Turn a D-Bus failure into a sentence that says what to do about it."""
    return _reason(getattr(exc, "name", "") or "",
                   str(getattr(exc, "text", "") or exc))


def _reason(name: str, text: str) -> str:
    """The same sentence from an error *name*, for error replies.

    dbus-next returns a D-Bus error reply as a Message instead of raising,
    so the name has to be read off the message itself — without this the
    caller would wait forever for a request that was already refused.
    """
    if "ServiceUnknown" in name or "NameHasNoOwner" in name:
        return ("xdg-desktop-portal is not running on this session bus, so "
                "global shortcuts cannot be registered.")
    if "UnknownMethod" in name or "UnknownInterface" in name or "UnknownObject" in name:
        return ("this desktop's xdg-desktop-portal does not provide the "
                "GlobalShortcuts portal (it needs xdg-desktop-portal 1.28 or "
                "newer with a desktop implementation, e.g. Plasma 6).")
    if "NoReply" in name or "Timeout" in name or "timed out" in text.lower():
        return "the desktop did not answer the shortcut request in time."
    if "Disconnected" in name or "not connected" in text.lower():
        return "the session bus went away, so global shortcuts cannot be registered."
    return f"the desktop refused the shortcut request ({name or text})."


class PortalBackend:
    """Async backend for :class:`listn.shortcuts.ShortcutManager`."""

    name = "freedesktop-globalshortcuts"

    def __init__(self, bus=None, timeout: float = RESPONSE_TIMEOUT):
        self._bus = bus
        self._owns_bus = bus is None
        self._session: str | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._responses: dict[str, tuple[int, dict]] = {}
        self._on_activate = None
        self.timeout = timeout
        self._handler_installed = False
        self._matches_added = False

    # -- bus --------------------------------------------------------------

    async def _connect(self):
        if self._bus is not None:
            return self._bus
        try:
            from dbus_next.aio import MessageBus
        except Exception as exc:
            raise PortalUnavailable(
                "dbus-next is not installed — install listn's tray extra "
                f"(uv sync --extra tray). ({type(exc).__name__})") from exc
        try:
            self._bus = await MessageBus().connect()
        except Exception as exc:
            raise PortalUnavailable(
                "there is no D-Bus session bus here, so global shortcuts "
                f"cannot be registered. ({type(exc).__name__})") from exc
        return self._bus

    async def _ensure_matches(self) -> None:
        """Ask the bus daemon for the portal's signals (see MATCH_RULES)."""
        if self._matches_added:
            return
        from dbus_next import Message

        bus = await self._connect()
        for rule in MATCH_RULES:
            reply = await bus.call(Message(
                destination="org.freedesktop.DBus", path="/org/freedesktop/DBus",
                interface="org.freedesktop.DBus", member="AddMatch",
                signature="s", body=[rule]))
            error = getattr(reply, "error_name", None) if reply else None
            if error:
                raise PortalUnavailable(
                    "the session bus refused the portal's signal "
                    f"subscription ({error}), so shortcut answers could "
                    "never arrive.")
        self._matches_added = True

    def _handle_message(self, msg) -> bool:
        if msg.interface == REQUEST_IFACE and msg.member == "Response" \
                and str(msg.path).startswith(REQUEST_PREFIX):
            try:
                code, results = msg.body
            except (TypeError, ValueError):
                return True
            path = str(msg.path)
            self._responses[path] = (int(code), results or {})
            future = self._pending.pop(path, None)
            if future is not None and not future.done():
                future.set_result((int(code), results or {}))
            return True
        if msg.interface == IFACE and msg.member == "Activated":
            try:
                _session, shortcut_id = msg.body[0], msg.body[1]
            except (IndexError, TypeError):
                return True
            callback = self._on_activate
            if callback is not None:
                # Key presses arrive on the bus thread; the manager decides
                # what to do (and swallows its own errors).
                asyncio.get_running_loop().call_soon(
                    _safe_call, callback, str(shortcut_id))
            return True
        if msg.interface == IFACE and msg.member == "ShortcutsChanged":
            return True
        return False

    # -- calls ------------------------------------------------------------

    async def _call(self, member: str, signature: str, body: list):
        from dbus_next import Message

        bus = await self._connect()
        reply = await bus.call(Message(destination=DEST, path=PATH,
                                       interface=IFACE, member=member,
                                       signature=signature, body=body))
        if reply is None:
            raise PortalUnavailable("the portal did not answer.")
        error = getattr(reply, "error_name", None)
        if error:
            detail = str(reply.body[0]) if getattr(reply, "body", None) else ""
            raise PortalUnavailable(_reason(str(error), detail))
        return reply.body

    async def _wait_response(self, request_path: str) -> tuple[int, dict]:
        if request_path in self._responses:
            return self._responses.pop(request_path)
        future = asyncio.get_running_loop().create_future()
        self._pending[request_path] = future
        try:
            return await asyncio.wait_for(future, self.timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_path, None)
            raise PortalUnavailable("the desktop did not answer the shortcut "
                                    "request in time.") from exc

    @staticmethod
    def _unwrap(value):
        return getattr(value, "value", value)

    # -- API used by the manager -----------------------------------------

    async def bind(self, chords: dict[str, str], on_activate) -> dict[str, tuple[bool, str]]:
        """Register every action; return ``{action: (granted, trigger)}``.

        Raises :class:`PortalUnavailable` when nothing can be registered —
        the manager turns that into a visible "unavailable" status.
        """
        from dbus_next import Variant

        bus = await self._connect()
        if not self._handler_installed:
            bus.add_message_handler(self._handle_message)
            self._handler_installed = True
        await self._ensure_matches()
        self._on_activate = on_activate

        token = _token()
        _body = await self._call("CreateSession", "a{sv}", [{
            "handle_token": Variant("s", token),
            "session_handle_token": Variant("s", token),
        }])
        request_path = str(_body[0])
        code, results = await self._wait_response(request_path)
        if code != 0:
            raise PortalUnavailable(_decline_reason(code))
        session = self._unwrap(results.get("session_handle"))
        if not session:
            raise PortalUnavailable("the desktop did not create a shortcut session.")
        self._session = str(session)

        entries = []
        for action, chord in chords.items():
            # dbus-next verifies STRUCTs as Python lists, not tuples.
            entries.append([action, {
                "description": Variant("s", DESCRIPTIONS.get(action, f"listn: {action}")),
                "preferred_trigger": Variant("s", str(chord)),
            }])
        bind_token = _token()
        reply = await self._call(
            "BindShortcuts", "oa(sa{sv})sa{sv}",
            [self._session, entries, "", {"handle_token": Variant("s", bind_token)}])
        code, results = await self._wait_response(str(reply[0]))
        if code != 0:
            raise PortalUnavailable(_decline_reason(code))

        granted: dict[str, tuple[bool, str]] = {}
        raw = self._unwrap(results.get("shortcuts")) or []
        triggers: dict[str, str] = {}
        for item in raw:
            try:
                shortcut_id, props = item[0], item[1]
            except (TypeError, IndexError, KeyError):
                continue
            trigger = ""
            for key in ("trigger_description", "description"):
                if key in props:
                    trigger = str(self._unwrap(props[key]))
                    break
            triggers[str(shortcut_id)] = trigger
        for action, chord in chords.items():
            ok = action in triggers
            granted[action] = (ok, triggers.get(action) or str(chord))
        return granted

    async def close(self) -> None:
        session, self._session = self._session, None
        if session:
            try:
                from dbus_next import Message

                bus = await self._connect()
                await bus.call(Message(destination=DEST, path=session,
                                       interface=SESSION_IFACE, member="Close"))
            except Exception:
                pass  # the session dies with the process anyway
        bus, self._bus = self._bus, None
        if bus is not None and self._owns_bus:
            try:
                bus.disconnect()
            except Exception:
                pass


def _safe_call(callback, *args) -> None:
    try:
        callback(*args)
    except Exception:  # a key press must never take the tray down
        log.exception("shortcut callback failed")


def _decline_reason(code: int) -> str:
    if code == 1:
        return ("the shortcut dialog was declined, so nothing was registered.")
    if code == 2:
        return ("the desktop reported an error while registering the shortcuts.")
    return f"the desktop returned response code {code} instead of registering the shortcuts."


def _token() -> str:
    return "listn" + uuid.uuid4().hex[:8]
