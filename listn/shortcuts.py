"""Configurable global shortcuts for the Linux desktop (start/stop, bookmark).

Two jobs, deliberately separated:

* :func:`parse_chord` / :func:`normalize_chord` / :func:`conflict` — pure
  parsing and local conflict detection. No D-Bus, no display, fully unit
  tested. A bad combination raises :class:`ShortcutError`, which
  ``Settings.validated`` turns into a readable ``SettingsError`` so a typo
  in ``settings.json`` fails loudly instead of silently doing nothing.
* :class:`ShortcutManager` — registers the pair with a desktop backend and
  turns the outcome into a :class:`ShortcutStatus` the tray can show.
  "Registered" is only ever claimed when the desktop says it granted the
  binding; everything else (no portal, no session bus, a combination the
  desktop refused or the user declined) is a distinct, visible state with
  its own reason. Tray controls keep working regardless.

Defaults are Linux choices, not a copy of the macOS modifiers: there is no
Option key on a PC keyboard, and ``Meta``/``Alt+Space``/``Alt+F2`` are
already spoken for by Plasma (KRunner, the run command), so the pair uses
``Ctrl+Alt`` with the same letters macOS uses. The desktop may still refuse
or reassign a combination — the effective trigger it reports is what gets
displayed, not what was asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("listn.shortcuts")

DEFAULT_START_STOP = "Ctrl+Alt+R"
DEFAULT_BOOKMARK = "Ctrl+Alt+B"

# action -> human label used in menu lines and portal descriptions
LABELS = {"start_stop": "start/stop", "bookmark": "bookmark"}

MODIFIER_ALIASES = {
    "ctrl": "Ctrl", "control": "Ctrl",
    "alt": "Alt", "opt": "Alt", "option": "Alt",
    "shift": "Shift",
    "meta": "Meta", "super": "Meta", "win": "Meta", "mod4": "Meta", "cmd": "Meta",
}
MODIFIERS = ("Ctrl", "Alt", "Shift", "Meta")

# Keys the freedesktop shortcuts spec spells this way; aliases map to them.
KEY_ALIASES = {
    "esc": "Escape", "escape": "Escape",
    "enter": "Return", "return": "Return",
    "space": "Space", "tab": "Tab", "backspace": "BackSpace",
    "delete": "Delete", "del": "Delete", "insert": "Insert", "ins": "Insert",
    "home": "Home", "end": "End",
    "pageup": "Page_Up", "pgup": "Page_Up", "pagedown": "Page_Down", "pgdn": "Page_Down",
    "left": "Left", "right": "Right", "up": "Up", "down": "Down",
    "print": "Print", "printscreen": "Print", "pause": "Pause",
    "plus": "plus", "minus": "minus", "comma": "comma", "period": "period",
    "slash": "slash", "semicolon": "semicolon", "apostrophe": "apostrophe",
    "bracketleft": "bracketleft", "bracketright": "bracketright",
    "backslash": "backslash", "equal": "equal", "grave": "grave",
}
_PUNCT = {
    ",": "comma", ".": "period", "/": "slash", ";": "semicolon",
    "'": "apostrophe", "[": "bracketleft", "]": "bracketright",
    "\\": "backslash", "=": "equal", "`": "grave", "-": "minus",
}


class ShortcutError(ValueError):
    """A shortcut string the user can fix in settings."""


@dataclass(frozen=True)
class Chord:
    """One parsed combination: modifiers + a single key."""

    modifiers: frozenset[str]
    key: str

    def __str__(self) -> str:
        ordered = [m for m in MODIFIERS if m in self.modifiers]
        return "+".join(ordered + [self.key])


def parse_chord(text: str) -> Chord:
    """``"ctrl+alt+r"`` -> ``Chord(frozenset({'Ctrl','Alt'}), 'R')``.

    Raises :class:`ShortcutError` with a message a person can act on.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ShortcutError("The shortcut is empty.")
    parts = [p.strip() for p in raw.split("+")]
    if any(not p for p in parts):
        raise ShortcutError(f"'{raw}' has an empty part — write it as Ctrl+Alt+R.")
    modifiers: set[str] = set()
    key: str | None = None
    for index, part in enumerate(parts):
        low = part.lower()
        if low in MODIFIER_ALIASES:
            modifiers.add(MODIFIER_ALIASES[low])
            continue
        if key is not None:
            raise ShortcutError(
                f"'{raw}' has more than one non-modifier key. Use Ctrl+Alt+R.")
        if index != len(parts) - 1:
            raise ShortcutError(
                f"'{raw}' puts '{part}' before the key. Modifiers come first: Ctrl+Alt+R.")
        key = _canonical_key(part, raw)
    if key is None:
        raise ShortcutError(f"'{raw}' has no key. Add one, e.g. Ctrl+Alt+R.")
    if not modifiers:
        raise ShortcutError(
            f"'{raw}' has no modifier. A global shortcut needs at least one "
            "(Ctrl, Alt, Shift or Meta), otherwise it would swallow normal typing.")
    return Chord(frozenset(modifiers), key)


def _canonical_key(part: str, raw: str) -> str:
    if len(part) == 1:
        if part.isalpha():
            return part.upper()
        if part.isdigit():
            return part
        if part in _PUNCT:
            return _PUNCT[part]
        raise ShortcutError(f"'{raw}' has an unsupported key '{part}'.")
    low = part.lower()
    if low in KEY_ALIASES:
        return KEY_ALIASES[low]
    if low.startswith("f") and low[1:].isdigit() and 1 <= int(low[1:]) <= 24:
        return f"F{int(low[1:])}"
    if part.isalpha() and len(part) <= 12:
        # A named key the spec spells as-is (Page_Up style names pass through
        # only when they are a known alias above); anything else is a typo.
        raise ShortcutError(
            f"'{raw}' has an unknown key '{part}'. Try a letter, F1–F12, "
            "Escape, Return, Space or an arrow key.")
    raise ShortcutError(f"'{raw}' has an unsupported key '{part}'.")


def normalize_chord(text: str) -> str:
    """Canonical spelling for settings storage; raises on nonsense."""
    return str(parse_chord(text))


def conflict(chords: dict[str, str]) -> str | None:
    """Local conflicts: two actions bound to the same combination.

    Returns a readable reason, or ``None`` when the pair is distinct. The
    desktop's own conflicts are only known after registration, so this runs
    *before* any D-Bus call and refuses to register a self-conflicting pair.
    """
    seen: dict[str, str] = {}
    for action, text in chords.items():
        try:
            chord = str(parse_chord(text))
        except ShortcutError as exc:
            return str(exc)
        if chord in seen:
            other = seen[chord]
            return (f"{LABELS.get(other, other)} and {LABELS.get(action, action)} "
                    f"are both {chord} — give them different keys.")
        seen[chord] = action
    return None


# ---------------------------------------------------------------------------
# Status: what the desktop actually granted
# ---------------------------------------------------------------------------

PENDING = "pending"
ACTIVE = "active"
PARTIAL = "partial"
DECLINED = "declined"
CONFLICT = "conflict"
UNAVAILABLE = "unavailable"


@dataclass
class ShortcutBinding:
    action: str
    chord: str
    state: str = PENDING
    reason: str = ""
    trigger: str = ""          # effective trigger as the desktop reports it

    @property
    def label(self) -> str:
        return LABELS.get(self.action, self.action)

    def line(self) -> str:
        shown = self.trigger or self.chord
        if self.state == ACTIVE:
            return f"{self.label} {shown}"
        return f"{self.label} {shown} — {self.state}: {self.reason}" if self.reason \
            else f"{self.label} {shown} — {self.state}"


@dataclass
class ShortcutStatus:
    """Backend-level truth for one registration attempt."""

    backend: str = ""
    state: str = PENDING
    reason: str = ""
    bindings: dict[str, ShortcutBinding] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == ACTIVE

    def summary(self) -> str:
        """One line for the tray menu. Never claims success it lacks."""
        if self.state == PENDING:
            return "Shortcuts: checking with the desktop…"
        if self.state == ACTIVE:
            parts = " · ".join(b.line() for b in self.bindings.values())
            return f"Shortcuts: {parts} — active"
        reason = self.reason or "the desktop did not grant them"
        return f"Shortcuts: unavailable — {reason}" if self.state == UNAVAILABLE \
            else f"Shortcuts: {self.state} — {reason}"

    def detail(self) -> str:
        """Longer form for the status tooltip: every binding's own state."""
        lines = [self.summary()]
        lines += [b.line() for b in self.bindings.values() if b.state != ACTIVE]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Actions: what a key press does
# ---------------------------------------------------------------------------

class HotkeyActions:
    """Turns a shortcut press into the tray's own actions.

    Both calls go through the controller, so a refused start or a bookmark
    with nothing recording surfaces as the controller's visible notice —
    the same truthful failure a menu click produces. Nothing here reports
    success on its own.
    """

    def __init__(self, controller):
        self.controller = controller

    def start_stop(self) -> None:
        controller = self.controller
        if controller.status.active:
            controller.stop()
        else:
            controller.start()

    def bookmark(self) -> None:
        self.controller.bookmark(source="hotkey")


# ---------------------------------------------------------------------------
# Manager: register with the desktop, or say why not
# ---------------------------------------------------------------------------

class ShortcutManager:
    """Owns one registration attempt and its status.

    ``backend`` is anything with ``name`` and
    ``async bind(bindings, on_activate) -> dict`` returning
    ``{action: (granted: bool, trigger: str)}``, plus an optional
    ``async close()``. ``listn.shortcut_portal.PortalBackend`` is the real
    one; tests inject fakes to exercise every outcome without a desktop.
    """

    def __init__(self, actions: HotkeyActions, chords: dict[str, str],
                 backend=None, backend_factory=None):
        self.actions = actions
        self._chords = dict(chords)
        self._backend = backend
        self._backend_factory = backend_factory
        self._status = ShortcutStatus(state=PENDING,
                                      bindings={a: ShortcutBinding(a, c)
                                                for a, c in chords.items()})
        self._bound: dict[str, str] = {}

    @property
    def status(self) -> ShortcutStatus:
        return self._status

    @property
    def chords(self) -> dict[str, str]:
        return dict(self._chords)

    def _on_activate(self, action: str) -> None:
        handler = {"start_stop": self.actions.start_stop,
                   "bookmark": self.actions.bookmark}.get(action)
        if handler is None:
            log.debug("shortcut %s has no action", action)
            return
        try:
            handler()
        except Exception:  # a key press must never take the tray down
            log.exception("shortcut action %s failed", action)

    async def start(self) -> ShortcutStatus:
        return await self.rebind(self._chords)

    async def rebind(self, chords: dict[str, str]) -> ShortcutStatus:
        """(Re)register a possibly changed pair. Safe to call repeatedly."""
        self._chords = dict(chords)
        bindings = {a: ShortcutBinding(a, c) for a, c in self._chords.items()}
        self._status = ShortcutStatus(state=PENDING, bindings=bindings)
        reason = conflict(self._chords)
        if reason:
            # Local conflict: register nothing, claim nothing.
            self._status.state = CONFLICT
            self._status.reason = reason
            for binding in self._status.bindings.values():
                binding.state = CONFLICT
                binding.reason = reason
            return self._status

        backend = self._backend
        if backend is None:
            factory = self._backend_factory or _default_backend
            try:
                backend = factory()
            except Exception as exc:
                return self._unavailable(f"{type(exc).__name__}: {exc}")
            self._backend = backend
        if backend is None:
            return self._unavailable(_no_backend_reason())

        try:
            outcome = await backend.bind(self._chords, self._on_activate)
        except Exception as exc:
            return self._unavailable(str(exc) or type(exc).__name__)

        granted = {a: outcome.get(a, (False, "")) for a in self._chords}
        for action, binding in self._status.bindings.items():
            ok, trigger = granted.get(action, (False, ""))
            binding.trigger = trigger or binding.chord
            binding.state = ACTIVE if ok else DECLINED
        active = [b for b in self._status.bindings.values() if b.state == ACTIVE]
        missing = [b for b in self._status.bindings.values() if b.state != ACTIVE]
        self._status.backend = getattr(backend, "name", "")
        if not missing:
            self._status.state = ACTIVE
            self._status.reason = ""
        elif not active:
            self._status.state = DECLINED
            self._status.reason = (
                "the desktop did not grant " +
                ", ".join(f"{b.label} ({b.trigger or b.chord})" for b in missing) +
                " — it may already be in use by another app, or the "
                "confirmation was declined.")
        else:
            self._status.state = PARTIAL
            self._status.reason = (
                "only " + ", ".join(b.label for b in active) +
                " was granted; " + ", ".join(b.label for b in missing) +
                " was not.")
        self._bound = {a: str(parse_chord(c)) for a, c in self._chords.items()}
        return self._status

    def _unavailable(self, reason: str) -> ShortcutStatus:
        self._status.state = UNAVAILABLE
        self._status.reason = reason
        for binding in self._status.bindings.values():
            binding.state = UNAVAILABLE
            binding.reason = reason
        log.info("global shortcuts unavailable: %s", reason)
        return self._status

    async def close(self) -> None:
        backend, self._backend = self._backend, None
        close = getattr(backend, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                log.debug("shortcut backend close failed", exc_info=True)


def _default_backend():
    """The freedesktop portal backend, or None when it cannot be used.

    Kept as a function so importing ``listn.shortcuts`` never needs
    dbus-next or a session bus.
    """
    try:
        from listn.shortcut_portal import PortalBackend
    except Exception:
        return None
    return PortalBackend()


def _no_backend_reason() -> str:
    try:
        import dbus_next  # noqa: F401
    except Exception:
        return ("dbus-next is not installed — install the tray extra "
                "(uv sync --extra tray) and run listn from a desktop session.")
    return ("no global-shortcut backend could be created; start listn from a "
            "desktop session with a session bus.")
