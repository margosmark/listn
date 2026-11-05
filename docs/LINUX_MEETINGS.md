# Linux meeting detection and prompts (L3a)

Status: implemented with fixture tests; **not** verified on a real KDE
session with real meeting apps. The macOS watcher
(`macos/Sources/ListnMenu/MeetingWatch.swift`) is untouched and remains
the macOS implementation.

## What ships

`listn tray` polls for running meetings and, when one appears, offers to
record it from the tray menu:

| Prompt | Menu items |
|---|---|
| Meeting detected | **Record \<app\>?** / **Not now** / **Remind me in 5 minutes** |
| Meeting you recorded ended | **\<app\> ended — stop recording** |

* **Consent first.** With `offer_meetings` on (default) a detection only
  ever *offers*. Recording starts from an explicit click, or from
  `auto_record_meetings` — which is **off by default** and, when on, is
  the user's own setting. There is no other start path.
* **Settings are re-read every poll** (`GET /api/settings`, the same
  route macOS re-reads on every `menuWillOpen`), so turning
  `auto_record_meetings` off while the tray runs stops later
  auto-starts, and `offer_meetings=off` stops new prompts — no restart.
  A failed fetch keeps the **last known** values, never the defaults, so
  a dead server can never resurrect auto-record after it was revoked;
  before the first successful read the watcher's own defaults apply
  (offer on, auto-record off).
* **Dismiss** stays quiet for the rest of that meeting; if the same app
  starts a *new* meeting later, it asks again.
* **Snooze** asks again after five minutes while the meeting continues.
* A pending offer makes the tray icon `NeedsAttention` with the offer in
  the tooltip; it is not an error, so the error icon is not shown.

## Detection sources (and what is deliberately not used)

Rules are the shared macOS rules in `listn.meetings` (dedicated meeting
apps by id; browsers and Slack only with a matching window title). The
Linux sources feeding them:

| Source | Works on | Evidence | Limits |
|---|---|---|---|
| PipeWire app identity (`pw-dump`) | X11 **and** Wayland | A `Stream/Input/Audio` node — the app is *using the microphone*, the same "audio active" signal the macOS process tap gives. Identity from `application.id` / `application.process.binary` / `application.name`. | Playback-only streams (`Stream/Output/Audio`) are not evidence: an app merely playing audio is not in a call. |
| X11 window titles (`xprop`, argv only) | X11 / XWayland | `_NET_WM_NAME` + `WM_CLASS` per client, for the browser/huddle title rule. | Native Wayland windows are not exposed to clients. On Wayland, browsers and Slack huddles are **not detected** — listn reports no hit rather than guessing. `xprop` missing ⇒ source absent. |

Not used, by design: KWin scripting / private D-Bus window APIs,
foreign-toplevel Wayland protocols, accessibility APIs, or any per-app
permission scrape. Detection needs no permission the recorder does not
already have, and degrades to "no meeting seen" when a source is
unavailable — never to a guess.

**Privacy:** window titles and stream names are classification inputs
only. They are never logged, never persisted, and never appear in prompt
metadata: `MeetingPrompt.to_dict()` is a fixed five-key dict
(`kind`, `meeting_id`, `label`, `source`, `title`), where `title` is the
app label (e.g. `Zoom`), not window text. No calendar data is read.

## Contract shared with L2 (captions/hotkeys, `t_7c59bc5e`)

* Metadata shape: `listn.meeting_watch.MeetingPrompt.to_dict()` — the
  five keys above; `kind` ∈ `offer | auto_start | end | auto_end`.
* Menu ids: `ITEM_MEETING_RECORD=40`, `ITEM_MEETING_DISMISS=41`,
  `ITEM_MEETING_SNOOZE=42`, `ITEM_MEETING_END=43` (only present while a
  prompt is pending).
* L3a adds **no server routes**; settings stay server truth via the
  existing `GET /api/settings`. L2 should consume `MeetingPrompt` rather
  than adding meeting routes.

## Module map

* `listn/meeting_sources.py` — sources: PipeWire capture-stream
  evidence, X11 titles, `collect_hits()` (throttled, never raises).
* `listn/meeting_watch.py` — pure polling state machine + prompt
  contract; occurrence dedup, dismiss/snooze, end offer.
* `listn/meetings.py` — shared rules (now also matching Linux desktop
  ids, case-insensitively).
* `listn/tray.py` / `listn/tray_sni.py` — prompt menu, consent actions,
  `NeedsAttention` while an offer is pending.
* `listn/cli.py` — `listn tray` constructs the watcher.

## Verification

```sh
uv run pytest tests/test_meeting_watch.py   # 25 fixture tests
uv run pytest tests/test_tray.py tests/test_tray_sni.py tests/test_meetings.py
```

Covered: supported apps from PipeWire fixtures, browser meetings and
Slack huddles with/without title hints, false positives (playback-only,
ordinary titles), polling dedup, no start without consent, auto-record
default off, dismiss/snooze/end offer, prompt metadata without titles,
and settings read as server truth *over time* — a mid-run toggle
honoured (auto-record revoked ⇒ no later auto-start; offers off ⇒ no
prompt, pending offer cleared) and a failed fetch keeping the last known
values instead of the defaults.

**Not covered / needs a real session:** a real meeting app on real
hardware, a real plasmashell panel rendering the prompt, and XWayland
title reads on CachyOS. Treat this as implemented-but-unaccepted until
those are run.
