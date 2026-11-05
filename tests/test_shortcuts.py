"""Global shortcuts: parsing, conflicts, actions, and honest status.

The interesting rules in ``listn/shortcuts.py`` need no desktop:

* a combination is parsed and normalised (or rejected with something a
  person can fix),
* a self-conflicting pair registers nothing and says so,
* a key press runs the tray's own actions, so failures surface as the
  controller's visible notice instead of a fake success,
* every outcome the desktop can give (all granted / some / none / no
  portal at all) has its own status line, and the start/stop controls stay
  in the menu regardless.

``tests/test_shortcut_portal.py`` then drives the real D-Bus backend
against a fake portal; this file stays pure.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from listn.config import Settings, SettingsError
from listn.engine import Engine
from listn.server import create_app
from listn.shortcuts import (ACTIVE, DEFAULT_BOOKMARK, DEFAULT_START_STOP,
                             DECLINED, HotkeyActions, PARTIAL, ShortcutError,
                             ShortcutManager, ShortcutStatus, UNAVAILABLE,
                             conflict, normalize_chord, parse_chord)
from listn.tray import (IDLE, ITEM_BOOKMARK, ITEM_CAPTIONS, ITEM_SHORTCUTS,
                        ITEM_START, RECORDING, ServerClient, TrayController,
                        TrayStatus, build_menu)

from .conftest import MAC_DEVICES, FakeSD


# ---------------------------------------------------------------------------
# Parsing and local conflicts
# ---------------------------------------------------------------------------

def test_normalises_aliases_to_the_spec_spelling():
    assert normalize_chord("ctrl+alt+r") == "Ctrl+Alt+R"
    assert normalize_chord("Control+Alt+R") == "Ctrl+Alt+R"
    assert normalize_chord("meta+shift+b") == "Shift+Meta+B"
    assert normalize_chord("alt+f5") == "Alt+F5"
    assert str(parse_chord("Super+Return")) == "Meta+Return"


@pytest.mark.parametrize("bad", [
    "", "   ", "r", "Ctrl+", "+Alt", "Ctrl+Alt", "Ctrl+Alt+R+X",
    "Ctrl+Alt+NotAKey", "Alt+Space+R",
])
def test_rejects_combinations_a_person_could_fix(bad):
    with pytest.raises(ShortcutError):
        normalize_chord(bad)


def test_conflict_names_both_actions():
    reason = conflict({"start_stop": "Ctrl+Alt+R", "bookmark": "ctrl+alt+r"})
    assert reason is not None
    assert "start/stop" in reason and "bookmark" in reason
    assert conflict({"start_stop": "Ctrl+Alt+R",
                     "bookmark": DEFAULT_BOOKMARK}) is None


def test_defaults_are_distinct_and_documented_linux_choices():
    assert normalize_chord(DEFAULT_START_STOP) == "Ctrl+Alt+R"
    assert normalize_chord(DEFAULT_BOOKMARK) == "Ctrl+Alt+B"
    # Not a copy of the macOS Ctrl-Option pair: there is no Option key, and
    # the letters are kept so muscle memory transfers.
    assert "Option" not in DEFAULT_START_STOP
    assert conflict({"start_stop": DEFAULT_START_STOP,
                     "bookmark": DEFAULT_BOOKMARK}) is None


# ---------------------------------------------------------------------------
# Settings: a bad combination fails loudly
# ---------------------------------------------------------------------------

def test_settings_normalise_and_reject_bad_shortcuts(data_dir):
    s = Settings(hotkey_start_stop="ctrl+alt+r").validated()
    assert s.hotkey_start_stop == "Ctrl+Alt+R"
    with pytest.raises(SettingsError, match="hotkey_bookmark"):
        Settings(hotkey_bookmark="r").validated()
    with pytest.raises(SettingsError, match="hotkey_start_stop"):
        Settings(hotkey_start_stop="Ctrl+Alt").validated()


def test_captions_are_opt_in(data_dir):
    assert Settings().captions_enabled is False


def test_settings_file_round_trips_the_new_fields(data_dir):
    s = Settings(captions_enabled=True, hotkey_start_stop="Ctrl+Alt+K")
    s.validated().save()
    loaded = Settings.load()
    assert loaded.captions_enabled is True
    assert loaded.hotkey_start_stop == "Ctrl+Alt+K"
    assert Settings.path().stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# Actions: a key press is a tray action
# ---------------------------------------------------------------------------

class FakeController:
    def __init__(self, active=False):
        self.status = TrayStatus(state=RECORDING if active else IDLE)
        self.calls: list = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def bookmark(self, note="", source="tray"):
        self.calls.append(("bookmark", source))


def test_start_stop_toggles_on_the_current_state():
    idle = HotkeyActions(FakeController(active=False))
    idle.start_stop()
    assert idle.controller.calls == ["start"]
    recording = HotkeyActions(FakeController(active=True))
    recording.start_stop()
    assert recording.controller.calls == ["stop"]


def test_bookmark_keeps_its_provenance():
    actions = HotkeyActions(FakeController(active=True))
    actions.bookmark()
    assert actions.controller.calls == [("bookmark", "hotkey")]


# ---------------------------------------------------------------------------
# Manager: every desktop outcome has its own status
# ---------------------------------------------------------------------------

class FakeBackend:
    name = "fake-portal"

    def __init__(self, granted=None, error=None):
        self.granted = granted
        self.error = error
        self.bindings = None
        self.on_activate = None
        self.closed = False

    async def bind(self, chords, on_activate):
        self.bindings = dict(chords)
        self.on_activate = on_activate
        if self.error is not None:
            raise self.error
        if self.granted is None:
            return {action: (True, chord) for action, chord in chords.items()}
        return self.granted

    async def close(self):
        self.closed = True


def manager(backend, chords=None):
    return ShortcutManager(
        HotkeyActions(FakeController()),
        chords or {"start_stop": DEFAULT_START_STOP, "bookmark": DEFAULT_BOOKMARK},
        backend=backend)


def test_all_granted_is_active_with_the_desktops_own_triggers():
    backend = FakeBackend(granted={"start_stop": (True, "Ctrl+Alt+R"),
                                   "bookmark": (True, "Ctrl+Alt+B")})
    status = asyncio_run(manager(backend).start())
    assert status.state == ACTIVE
    assert status.backend == "fake-portal"
    assert status.ok
    assert "active" in status.summary()
    assert "Ctrl+Alt+R" in status.summary()


def test_partial_grant_names_what_is_missing():
    backend = FakeBackend(granted={"start_stop": (True, "Ctrl+Alt+R"),
                                   "bookmark": (False, "")})
    status = asyncio_run(manager(backend).start())
    assert status.state == PARTIAL
    assert "bookmark" in status.reason
    assert status.bindings["bookmark"].state == DECLINED


def test_nothing_granted_is_declined_not_active():
    backend = FakeBackend(granted={"start_stop": (False, ""), "bookmark": (False, "")})
    status = asyncio_run(manager(backend).start())
    assert status.state == DECLINED
    assert not status.ok
    assert "in use" in status.reason or "declined" in status.reason
    assert "declined" in status.summary() or "not" in status.summary()


def test_backend_failure_is_unavailable_with_the_reason():
    backend = FakeBackend(error=RuntimeError("xdg-desktop-portal is not running"))
    status = asyncio_run(manager(backend).start())
    assert status.state == UNAVAILABLE
    assert "xdg-desktop-portal is not running" in status.reason
    assert "unavailable" in status.summary()


def test_local_conflict_registers_nothing():
    backend = FakeBackend()
    status = asyncio_run(manager(
        backend, {"start_stop": "Ctrl+Alt+R", "bookmark": "ctrl+alt+r"}).start())
    assert backend.bindings is None          # never reached the desktop
    assert status.state != ACTIVE
    assert "both Ctrl+Alt+R" in status.reason


def test_rebind_picks_up_a_changed_pair():
    backend = FakeBackend()
    shortcuts = manager(backend)
    asyncio_run(shortcuts.start())
    assert backend.bindings["bookmark"] == DEFAULT_BOOKMARK
    new = {"start_stop": "Ctrl+Alt+K", "bookmark": DEFAULT_BOOKMARK}
    status = asyncio_run(shortcuts.rebind(new))
    assert status.state == ACTIVE
    assert backend.bindings["start_stop"] == "Ctrl+Alt+K"
    assert shortcuts.chords["start_stop"] == "Ctrl+Alt+K"


def test_close_releases_the_backend():
    backend = FakeBackend()
    shortcuts = manager(backend)
    asyncio_run(shortcuts.start())
    asyncio_run(shortcuts.close())
    assert backend.closed


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Reporting: the menu says what happened and keeps working
# ---------------------------------------------------------------------------

def menu(items):
    return {item.id: item for item in items}


def test_menu_reports_an_unavailable_shortcut_and_keeps_the_controls():
    status = asyncio_run(manager(
        FakeBackend(error=RuntimeError("no portal here"))).start())
    items = menu(build_menu(TrayStatus(state=RECORDING), [], [], None, None,
                            shortcut_status=status))
    line = items[ITEM_SHORTCUTS]
    assert line.enabled is False
    assert "no portal here" in line.label
    assert "unavailable" in line.label
    # The controls a person can still use are untouched.
    assert items[ITEM_START].enabled is False
    assert items[ITEM_BOOKMARK].enabled is True


def test_menu_omits_the_shortcut_line_without_a_status():
    items = menu(build_menu(TrayStatus(state=IDLE), [], [], None, None))
    assert ITEM_SHORTCUTS not in items
    assert ITEM_CAPTIONS in items


def test_caption_toggle_is_a_check_mark_and_reflects_the_setting():
    off = menu(build_menu(TrayStatus(state=IDLE), [], [], None, None,
                          captions_enabled=False))[ITEM_CAPTIONS]
    on = menu(build_menu(TrayStatus(state=IDLE), [], [], None, None,
                         captions_enabled=True))[ITEM_CAPTIONS]
    assert off.toggle == "checkmark" and off.toggled is False
    assert on.toggled is True


def test_caption_toggle_explains_itself_when_there_is_no_display():
    items = menu(build_menu(TrayStatus(state=IDLE), [], [], None, None,
                            captions_unavailable="no display is available to this process"))
    item = items[ITEM_CAPTIONS]
    assert item.enabled is False
    assert "no display" in item.label
    assert item.toggled is False


# ---------------------------------------------------------------------------
# End to end: a bookmark through the real server, with provenance
# ---------------------------------------------------------------------------

class Transport:
    """TestClient behind ServerClient's transport seam (GET/POST/PUT)."""

    def __init__(self, client):
        self.client = client

    def get(self, path):
        r = self.client.get(path)
        return r.status_code, (r.json() if r.content else {})

    def post(self, path, body):
        r = self.client.post(path, json=body or {})
        return r.status_code, (r.json() if r.content else {})

    def put(self, path, body):
        r = self.client.put(path, json=body or {})
        return r.status_code, (r.json() if r.content else {})


@pytest.fixture
def http_tray(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        controller = TrayController(ServerClient(transport=Transport(client)),
                                    targets_file=data_dir / "tray-targets.json")
        yield controller, client, engine
        if engine.live_active():
            engine.stop_live(wait=True)


def test_bookmark_route_is_absent_until_something_is_recording(http_tray):
    controller, client, engine = http_tray
    refused = client.post("/api/live/bookmark", json={"source": "hotkey"})
    assert refused.status_code == 409
    assert "recording" in refused.json()["detail"].lower()


def test_hotkey_bookmark_lands_in_the_session_with_its_source(http_tray):
    controller, client, engine = http_tray
    assert client.post("/api/live/start", json={"title": "Sync"}).status_code == 200
    actions = HotkeyActions(controller)
    actions.bookmark()
    marks = engine.library.bookmarks(engine.live.meta.id)
    assert len(marks) == 1
    assert marks[0]["source"] == "hotkey"
    assert marks[0]["note"] == "Bookmark"


def test_tray_menu_bookmark_writes_tray_provenance(http_tray):
    controller, client, engine = http_tray
    assert client.post("/api/live/start", json={"title": "Sync"}).status_code == 200
    controller.action(ITEM_BOOKMARK)
    marks = engine.library.bookmarks(engine.live.meta.id)
    assert marks and marks[-1]["source"] == "tray"


def test_unknown_bookmark_source_is_rejected(http_tray):
    _controller, client, _engine = http_tray
    assert client.post("/api/live/start", json={"title": "Sync"}).status_code == 200
    bad = client.post("/api/live/bookmark", json={"source": "malware"})
    assert bad.status_code == 400


def test_shortcut_settings_reject_a_bad_value_over_http(http_tray):
    controller, client, _engine = http_tray
    bad = client.put("/api/settings", json={"hotkey_bookmark": "r"})
    assert bad.status_code == 400
    assert "modifier" in bad.json()["detail"]
    # The stored setting is untouched: a rejected write changes nothing.
    assert client.get("/api/settings").json()["hotkey_bookmark"] == DEFAULT_BOOKMARK
    assert controller.settings == {}


def test_controller_adopts_settings_and_flips_captions(http_tray):
    controller, client, _engine = http_tray
    controller.refresh()
    assert controller.settings.get("captions_enabled") is False
    assert controller.captions_enabled is False
    assert controller.toggle_captions() is True
    assert controller.captions_enabled is True
    items = menu(controller.menu())
    assert items[ITEM_CAPTIONS].toggled is True
    assert controller.toggle_captions() is False


def test_controller_rebinds_when_the_shortcuts_change(http_tray):
    controller, client, _engine = http_tray
    seen: list[dict] = []
    controller.on_chords_changed = seen.append
    controller.refresh()
    assert seen == []
    assert client.put("/api/settings",
                      json={"hotkey_start_stop": "Ctrl+Alt+K"}).status_code == 200
    for _ in range(6):                     # SETTINGS_POLL_EVERY = 5
        controller.refresh()
    assert seen and seen[-1]["start_stop"] == "Ctrl+Alt+K"
    assert seen[-1]["bookmark"] == DEFAULT_BOOKMARK
