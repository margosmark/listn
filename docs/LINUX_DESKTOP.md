# Linux desktop: captions and global shortcuts (Plasma)

Status: **implemented and tested on a headless machine — real D-Bus session
bus with a fake portal, real Tk window under Xvfb. Not yet exercised inside
a running plasmashell panel or with a physical key press.** The manual
test [below](#target-plasma-manual-test-the-acceptance-gate) is the
acceptance gate; the automated suites do **not** substitute for it. This is
L2 of the Linux/macOS parity epic (synthesis on task `t_f35d2739`);
session/report semantics follow the L0 capture contract
(`docs/CAPTURE_CONTRACT.md`, landing from the L0 branch `wt/t_3db2e7fc`).

Everything here is local-first: the caption window is drawn by the tray
process from the loopback event stream, shortcuts are registered through
the desktop's own portal on the session bus, and nothing in this feature
opens a socket to anywhere else. `macos/` is untouched — the macOS menu bar
keeps its own native captions window and Control-Option hotkeys, and the
settings introduced here are inert there.

## Captions (opt-in, local only)

`listn/captions.py` renders a floating Tk window over the engine's existing
local transcript events (`GET /api/events`, the same SSE stream the web UI
uses). It is **off by default** (`captions_enabled: false`) and can be
switched on from the tray menu ("Show captions", a check mark) or via
`PUT /api/settings {"captions_enabled": true}`. The client subscribes to
the event stream *only while the setting is on*, so a recording made with
captions off never reaches this code path.

Privacy rules — each one is a test in `tests/test_captions.py`:

- **Local only.** Events are read from `127.0.0.1` and drawn by this
  process. No network destination other than the loopback API exists in the
  module.
- **Never logged.** Caption text is never passed to `log` (only counts and
  states), and tests walk real log records and the stream log for it.
- **Never copied.** There is no clipboard call anywhere in the module.
- **Clear on pause, close on stop.** `live.paused` empties the window and
  hides it; `live.stopped` empties it and destroys it (the widget is
  destroyed, so the text is gone with the window). Resume shows an empty
  window again; the `[paused]` marker is never rendered as speech.
- **Session-pinned.** Only events for the session captions were opened for
  are rendered; stale events from another recording are dropped.

Without tkinter or a display the supervisor reports why in the tray menu
("Show captions — <reason>") and the toggle is disabled instead of
silently doing nothing. The Tk window itself is tested under Xvfb; that
covers show/update/hide/close and text lifecycle, not compositing on a
real desktop (over-full-screen behaviour, focus stealing) — that is part of
the manual test.

## Global shortcuts (start/stop, bookmark)

`listn/shortcuts.py` parses and validates chords; `listn/shortcut_portal.py`
registers them with the freedesktop **GlobalShortcuts portal**
(`org.freedesktop.portal.GlobalShortcuts`, interface version 2), which is
the same mechanism on X11 and Wayland and needs no X server, no
`python-xlib` and no per-desktop guesswork.

- **Defaults:** `Ctrl+Alt+R` (start/stop) and `Ctrl+Alt+B` (bookmark) —
  Linux choices, not a copy of the macOS modifiers: there is no Option key
  on a PC keyboard, and `Meta`/`Alt+Space`/`Alt+F2` are already spoken for
  by Plasma (KRunner, the run command).
- **Configuration:** `hotkey_start_stop` / `hotkey_bookmark` in
  `settings.json`, or `PUT /api/settings`. Values are normalised on save
  (`ctrl+alt+r` → `Ctrl+Alt+R`) and a value a person can fix (empty part,
  no modifier, unknown key) is rejected with a readable `400` **and the
  previous settings stay in effect** — a typo fails loudly instead of
  silently doing nothing. The web settings page does not expose these two
  fields yet; the tray toggle covers captions, the API/settings file cover
  the chords.
- **Local conflict:** if both actions are set to the same combination, the
  pair is refused *before* any D-Bus call and the menu says which two
  actions collide.
- **Desktop answer, reported truthfully.** The portal may grant a subset or
  reassign a key. The status line shows the *effective* trigger the desktop
  reported, and each outcome is distinct in the tray menu:
  `active` (both granted), `partial` (names what is missing), `declined`
  (nothing granted — likely already in use by another app, or the
  confirmation was declined), `unavailable` (no portal/session bus, with
  the reason: e.g. `xdg-desktop-portal` not running, or older than 1.28 /
  without a desktop implementation). Tray start/stop/bookmark controls keep
  working in every state.
- **Rebinding.** The tray pulls `GET /api/settings` on a slow tick; when a
  chord changes it re-registers with the desktop and the status line
  updates. A press that the desktop never grants simply never arrives —
  nothing claims a shortcut works before the desktop says it did.

Wayland note: key capture *must* go through the portal — there is no
XGrabKey fallback, by design; with no portal the feature reports
`unavailable` rather than pretending.

## Tray and API contract (shared with L3a)

- New route: `POST /api/live/bookmark` with `{note?, source?}` where
  `source ∈ {api, tray, hotkey, web}`; `409` when nothing is recording.
  The source is stored with the bookmark (`{"t", "note", "source"}`);
  records without one keep their exact two-key shape.
- New settings (inert on macOS): `captions_enabled`, `hotkey_start_stop`,
  `hotkey_bookmark` — all returned by `GET /api/settings`.
- Tray menu ids added here: `7` Bookmark this moment, `8` Show captions,
  `9` shortcut status line (informational, never clickable).
  `10/11/100+/200+` (pickers) and L3a's meeting ids `40–43` are untouched.
- macOS: the menu-bar app's native captions and hotkeys are untouched;
  this document describes Linux only.

## What is verified where

| Claim | Evidence |
|---|---|
| Chord parsing, local conflicts, settings validation/rejection | `tests/test_shortcuts.py` (unit, no bus) |
| Status mapping: active/partial/declined/conflict/unavailable + menu reporting | `tests/test_shortcuts.py` |
| Real D-Bus round trip: CreateSession/BindShortcuts/Activated against a fake portal on a private `dbus-daemon` | `tests/test_shortcut_portal.py` (8 tests) |
| Bookmark route + provenance, settings 400-and-rollback | `tests/test_shortcuts.py` (`http_tray`) |
| Caption model/privacy rules, supervisor wiring, engine event stream | `tests/test_captions.py` |
| Tk window show/update/hide/close + text destroyed with the window | `tests/test_captions.py` under `xvfb-run` (18/18) |
| Tray D-Bus contract (StatusNotifierItem/dbusmenu) | `tests/test_tray_sni.py` (real session bus) |
| **A running plasmashell panel accepting these menu ids, a real key press, compositing over a full-screen app** | **Not automated — manual test below** |

## Target Plasma manual test (the acceptance gate)

Run on the target machine (CachyOS / KDE Plasma 6, Wayland or X11 —
record both versions plus `xdg-desktop-portal` and `pipewire` versions):

1. **Panel.** `uv run listn tray` — the icon appears in the panel; the
   menu shows the status line, Start/Stop, Pause/Resume, pickers,
   *Bookmark this moment*, *Show captions*, the shortcut status line,
   privacy settings and Quit. No error icon, no missing entries.
2. **Captions lifecycle.** Enable *Show captions*, start a recording: the
   window appears above other windows (test a full-screen video) without
   taking keyboard focus; lines update as segments arrive. Pause → text
   cleared and window hidden; resume → window back and empty; stop →
   window destroyed. Switch captions off mid-recording → subscription ends
   and the window closes.
3. **Caption privacy.** After a recording with captions on, confirm no
   caption text appears in the tray's log output or `journalctl --user`
   for this session, and that no clipboard content changed.
4. **Shortcuts.** With another application focused: `Ctrl+Alt+R` starts a
   recording, again stops it; `Ctrl+Alt+B` while recording adds a bookmark
   visible in the session with `source: "hotkey"`; pressing it with
   nothing recording shows the refusal as the tray's notice. The menu's
   shortcut line says *active* with the triggers Plasma actually bound.
5. **Conflict/unavailable reporting.** Set `hotkey_start_stop` to a
   combination another app already owns (or reassign it in Plasma's
   Shortcuts settings) → the menu shows *partial* or *declined* naming
   what is missing, and tray controls still work. Set both actions equal →
   *conflict* naming both. Set a malformed value over `PUT /api/settings`
   → `400` with the reason and settings unchanged. Stop
   `xdg-desktop-portal` (or run on a session without it) → *unavailable*
   with the reason, never a silent no-op.
6. **Rebind.** Change a chord in settings while the tray runs → within a
   few seconds the status line reflects the new registration and the new
   key works.
7. **macOS regression (separate machine).** The macOS menu bar shows its
   own native captions window and Control-Option hotkeys exactly as
   before; the three new settings do nothing there.

Automated D-Bus and Xvfb coverage reproduces everything below the panel
(the contract, the wiring, the window logic). It cannot prove that
plasmashell renders the menu, that the compositor stacks the window, or
that a physical key press reaches us — those claims belong to the run
above and must not be made from the test suite alone.
