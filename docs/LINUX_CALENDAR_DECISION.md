# L3b decision — Linux calendar provider for meeting titles

Status: **proposal for approval; no implementation, no calendar data accessed.**
Task: `t_f121234a` (decision gate under `t_f35d2739` `linux-macos-parity.md`).
Checked 2026-11-04 against `origin/main` `2e2e65f`. macOS EventKit behavior is
unchanged and out of scope; nothing under `macos/` is touched by this decision.

Evidence labels: **[verified]** = checked by me this run (source cited),
**[reported]** = from repo docs or parent handoff, **[unknown]** = must be
probed on the target machine before/while implementing.

## 1. What macOS does today (the contract to mirror)

Source: `macos/Sources/ListnMenu/CalendarTitle.swift`, `listn/meetings.py`,
`tests/test_meetings.py` **[verified]**.

- One-time OS prompt (`requestFullAccessToEvents`, macOS ≥ 14; `requestAccess`
  before). Denied → no title, recording unaffected.
- Query window: `now − 6 h … now + 12 h`, all calendars.
- Selection rule: shortest event with `start <= moment < end` (all-day hold
  loses to the real meeting). Mirrored in Python `listn.meetings.pick_event`.
- Title + attendee names stored on the session (attendees capped `[:40]`, each
  `[:80]`; title capped `[:200]` in `title_and_attendees`) **[verified]**.
- Empty title = no match. Fall back to no title; the engine then names the
  session `Recording %b %d, %H:%M` (`listn/engine.py:99`) **[verified]** —
  this is the required "generic timestamped session title".
- Occurrences are expanded by EventKit within the window, so recurring
  meetings do match on macOS **[verified behavior claim from EventKit
  predicate semantics; consistent with README]**.

Integration seams already present on Linux **[verified]**: socket `START`
carries `title`/`attendees` (test at `tests/test_meetings.py:40`), library
`PATCH` validates the same caps (`listn/server.py:480-487`), web start body
passes `title` only (`listn/server.py:283,294` — attendees there would need a
one-line body addition when implementation happens).

## 2. Verified environment facts

| Fact | Evidence |
|---|---|
| **No calendar portal exists.** The official xdg-desktop-portal docs list 33 portal interfaces (Account … Wallpaper); none provides calendar events. | Fetched `flatpak.github.io/xdg-desktop-portal/docs/for-app-developers.html` and enumerated `doc-org.freedesktop.portal.*` this run **[verified]** |
| **EDS exposes read-only calendar D-Bus methods with no consent gate.** `org.gnome.evolution.dataserver.Calendar` has `GetObjectList`/`GetView`/`GetObject`; the entire evolution-data-server tree (1 699 files) contains **zero** polkit/policykit files, so reads are gated only by session-bus membership — any same-user process can read, no prompt is ever shown. `UserPrompter` is a generic GTK dialog for auth prompts, not read consent. | Source tree + XML from `raw.githubusercontent.com/GNOME/evolution-data-server/master/src/private/*.xml` **[verified]** |
| **EDS and Akonadi are installable on Arch/CachyOS but are not listn deps.** `extra/evolution-data-server 3.62.1`, `extra/akonadi 26.08.2`. listn's PKGBUILD depends are `python, uv, portaudio, pipewire, wireplumber` (+`dbus-next` only via the `tray` extra). Whether the target laptop has either installed is **[unknown]** — not in any recorded evidence. | archlinux.org package API this run; `packaging/arch/PKGBUILD`, `pyproject.toml` **[verified]** |
| **KDE's native event feed is PIM/Akonadi.** Plasma's digital clock shows events through the "PIM Events Plugin", enabled in clock settings, which needs KDE PIM (Merkuro/Kontact) running Akonadi. Stock Plasma without KDE PIM shows no events. | `raw.githubusercontent.com/KDE/merkuro/master/README.md` FAQ **[verified]** |
| listn already speaks D-Bus and tests against a **real private `dbus-daemon --session`**. | `listn/tray_sni.py`, fixture at `tests/test_tray_sni.py:67` **[verified]** |
| No ICS/EDS/Akonadi/CalDAV code exists anywhere in the repo; the shared rule file `listn/meetings.py` is pure Python and platform-neutral. | repo grep this run **[verified]** |
| `python-dateutil` is in `uv.lock` only transitively (via pandas/matplotlib in the `diarize` extra); `icalendar` is absent. Neither is a base dependency. | `uv.lock` scan **[verified]** |
| Two shared-rule defects that every real calendar source will hit: (a) aware-vs-naive datetimes raise `TypeError` inside `pick_event` (the comparison at `listn/meetings.py:68` is outside its `try`); (b) equal-length ties resolve by input order — reversed input picks a different title. | Reproduced by running `pick_event` this run: `TypeError: can't compare offset-naive and offset-aware datetimes`; tie → `A-tie1` vs `A-tie2` on shuffle **[verified]** |

## 3. Options considered

| # | Option | Available on target? | New dependencies | OS-level consent? | Local-only? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Opt-in local ICS file/directory** — user points Settings at one `.ics` file or a folder of them; listn parses at recording start | **Always** (any DE; covers exported Google/Nextcloud/Merkuro calendars and `davx5`-style sync folders) | `icalendar` + `python-dateutil` (pure Python, locked, no system service) | No OS prompt possible — in-app consent is the gate | **Yes** — file read in place, zero network, zero D-Bus | **Recommended primary** |
| 2 | **Evolution-Data-Server D-Bus** (`CalendarFactory.OpenCalendar` → `Calendar.GetObjectList`, sources via source-registry) | Only where `evolution-data-server` runs — GNOME default, **not stock KDE**; presence on the laptop `[unknown]` | `dbus-next` promoted to a Linux base dep + the system package `evolution-data-server` (user/`pacman`, never installed by listn) | **None exists** (§2) — in-app consent mandatory, weaker than macOS | Reads listn-side is local; EDS itself may hold synced cloud data, but listn makes no network calls | **Recommended as approval-gated adapter #2** |
| 3 | **KDE Akonadi D-Bus** (what the Plasma PIM Events plugin uses) | Only with KDE PIM (Merkuro/Kontact) + running `akonadiserver`; `[unknown]` on target | akonadi stack, item-payload fetching, heaviest integration | None | Local, but pulls the whole PIM stack | **Reject for now** — revisit only if the probe finds Akonadi and no other source |
| 4 | **CalDAV client** (Nextcloud/Google direct) | Anywhere with network | HTTP + credential storage | No standard prompt; credentials are the problem | **No** — outbound traffic + secrets, violates local-first | **Reject** — would need explicit privacy approval this card does not grant |
| 5 | **xdg-desktop-portal calendar interface** | Does not exist (§2) | — | — | — | **Reject — nothing to implement** |
| 6 | **PyKDE6 / KCalendarCore** (direct library binding) | KDE-only | PyKDE6 — very large; no D-Bus story anyway | None | Local | **Reject** |
| 7 | Reading Merkuro/Akonadi internal storage files directly | Fragile | none | none | local | **Reject** — undocumented internal paths, version-fragile |

### Recommendation

Implement behind one provider interface with **Option 1 (ICS) as the
shipped, default-recommended provider**, and **Option 2 (EDS) as a second
adapter that ships only after its two approvals land** (system package
`evolution-data-server` + `dbus-next` as a Linux base dependency). This gives
a working, deterministic, dependency-light feature now and a zero-friction
path later without ever reading anyone's calendar before explicit consent.
Both adapters return the same event-dict shape, so the selection rule,
status handling and tests are shared.

**Names/settings decided here** (so later cards cannot diverge):

- Module: `listn/calendar.py` — `IcsProvider`, `EdsProvider` (gated),
  `select_provider(settings)`, `CalendarStatus`.
- Settings keys in `listn/config.py::Settings`: `calendar_provider: str =
  "off"` (`off | ics | eds`), `calendar_ics_path: str | None = None`,
  `calendar_consent_at: str | None = None`.
- Status surface (settings API/page only, never logs): `calendar_status:
  {state, reason}` where `state ∈ {off, ok}` and `reason ∈ {consent_missing,
  path_missing, parse_error, service_absent, time_mismatch}` — **reason codes
  only, never event titles**.

## 4. Permission / consent model (required for both options)

There is **no OS prompt on Linux for either option** (§2), so listn must run
an explicit in-app consent gate — this is the Linux equivalent of the macOS
`requestFullAccessToEvents` dialog:

1. **Default off.** `calendar_provider="off"`; no provider object is ever
   constructed, no file or D-Bus calendar method is touched (test-spied).
2. **First enable is an explicit action** in Settings → Calendar with
   plain-language scope, e.g.: *“Allow listn to read calendar titles? When a
   recording starts, listn reads the event covering that moment — title and
   attendee names only — from `<source>` on this machine. Titles become the
   recording's name. Nothing is sent anywhere, no titles are logged, and you
   can remove access here at any time.”* ICS additionally requires the user
   to supply the path (a deliberate typed path, not an upload).
3. **Consent is recorded** as `calendar_consent_at` timestamp next to the
   provider setting; **revoking** resets provider+path+consent in one step
   and subsequent recordings read nothing.
4. **Reads happen only inside the recording-start flow** (same moment macOS
   resolves `pendingCalendar`), window `now−6 h … now+12 h`, fields limited
   to `SUMMARY` → title and `ATTENDEE`/`ORGANIZER` display names → attendees
   (existing 200/40/80 caps apply). No polling, no background service, no
   startup read, no network, no calendar content in logs/status/errors/
   dashboards.
5. **macOS keeps its system prompt** — the in-app gate is Linux-only; nothing
   in `macos/` changes.
6. If a future environment genuinely offers an OS-level calendar permission
   (not present today), listn should prefer it; the in-app consent still
   applies on top.

## 5. Deterministic selection contract (provider → `pick_event`)

1. **Normalization:** providers emit `{"title", "start", "end", "attendees"}`
   with **naive local wall-clock datetimes** (aware input → `astimezone()` →
   naive; all-day/`DATE` events → naive local midnight span, matching
   EventKit's local-midnight behavior). `moment` passed in is the naive local
   start time. Uniformity is asserted by tests — the mixed-awareness
   `TypeError` (§2) can then never reach `pick_event`.
2. **Canonical ordering:** before `pick_event`, providers sort by
   `(start, end, title, uid)` ascending. `pick_event`'s strict `<` then
   yields one stable winner regardless of provider iteration order —
   fixing the reproduced tie order-dependence. Exact-length ties pick the
   earliest-starting/lexically-first event; the theoretical divergence from
   macOS (EventKit's own order on equal lengths) is documented as accepted —
   it requires two overlapping same-length events both containing “now”.
3. **Window parity:** query exactly `now − 6 h … now + 12 h`, like
   `CalendarTitle.overlapping`.
4. **Recurrence:** expand `RRULE` occurrences into concrete instances inside
   the window (EventKit does the same). Engine: `python-dateutil.rrule`
   over `icalendar`-parsed components. Supported subset: `FREQ` =
   DAILY/WEEKLY/MONTHLY, `INTERVAL`, `COUNT`/`UNTIL`, `BYDAY`, plus `EXDATE`
   and `RDATE`. An unsupported rule → that event is skipped and the settings
   status counts it (`recurrence_unsupported` count, no titles); never a
   crash, never a silently wrong title.
5. **Shared-rule hardening (part of this approval):** widen the `try` in
   `listn/meetings.py::pick_event` to cover the `start <= moment < end`
   comparison, so an uncomparable entry is skipped exactly like a malformed
   timestamp already is (a provider bug then degrades to “no match” +
   status instead of an exception in the recording path). This cannot change
   results for well-formed uniform inputs; `tests/test_meetings.py` keeps the
   existing assertions and gains a regression for it. The Swift mirror stays
   byte-identical.
6. **No-match / empty title →** `title_and_attendees` returns `None` →
   engine default `Recording %b %d, %H:%M`, attendees `[]` — the honest
   generic fallback required by the parent story.

## 6. Failure handling and fallback

Recording start must never be blocked or delayed by calendar work; provider
calls are wrapped at one boundary that converts every failure into
`(no title, status reason)`:

| Failure | Behavior |
|---|---|
| Consent off / revoked | Provider never invoked; generic title; status `off` |
| ICS path missing / unreadable | Generic title; status `path_missing` |
| ICS unparseable / malformed entries | Bad events skipped; if nothing usable → generic title + `parse_error` (count of skipped events only) |
| ICS stale (old mtime) | Still used (user's explicit choice) — settings show file mtime so staleness is visible; content never shown |
| EDS service absent (option 2) | Generic title; status `service_absent` (adapter never claims a title it didn't read) |
| Mixed-awareness / internal error | Generic title; status `time_mismatch` / `parse_error`; exception never propagates into `start_live` |
| No event overlaps now | Generic title; status stays `ok` (access works, nothing matched) |

`docs/DECISIONS.md`/README already require “show what is unavailable and what
needs action” — the settings status line is that surface.

## 7. Approvals this gate requires before any code

- [ ] **New runtime deps:** `icalendar>=6,<7` and `python-dateutil>=2.9,<3`
      (pure Python, no system libs, no runtime network; downloaded once by
      the existing locked `uv`/PKGBUILD first-run mechanism). Alternative if
      rejected: a bounded in-repo ICS subset parser (more code, more risk) —
      dateutil is still required for recurrence either way.
- [ ] **Option 1 (ICS provider)** as described — recommended approve.
- [ ] **Option 2 (EDS adapter):** `dbus-next>=0.2,<1` promoted to a Linux
      base dependency (marker `sys_platform == "linux"`; already proven by
      the tray tests) **and** `evolution-data-server` as a documented
      user-installed system package (PKGBUILD `optdepends`, never installed
      by listn). Recommended approve **only after** a probe of the target
      machine shows EDS present (or the user opts in to installing it).
- [ ] **Environment probe** (presence only, no event content): a
      `listn calendar --probe`-style check reporting D-Bus names
      (`…evolution.dataserver…`, `org.freedesktop.Akonadi1`), packages, and
      whether `.ics` paths exist — folded into
      `scripts/linux_hardware_check.sh` or run manually on the laptop.
- [ ] **Explicit rejections to record:** CalDAV/network sources, Akonadi
      adapter, PyKDE6, portal (nonexistent), internal-storage scraping.

No calendar content was read while making this decision: only repository
source/tests, public documentation and package metadata were inspected.

## 8. Test plan (implementation must ship all of these)

Fixture-driven, zero real calendar data, isolated `LISTN_DATA_DIR`:

1. `tests/test_calendar_ics.py` — ICS fixtures covering: overlap picks
   shortest; **boundary** (`start` inclusive, `end` exclusive, `moment ==
   end` → no match); all-day loses to timed; timezone trio (UTC `Z`, `TZID`
   VTIMEZONE, floating); recurring weekly `RRULE` matches inside the window
   and not outside; `EXDATE` removes an occurrence; unsupported `RRULE`
   skipped with count; folded lines, escaped `\,`/`\;`/`\\`, `STATUS:CANCELLED`
   excluded; malformed VEVENT skipped without crashing; empty file; no event.
2. `tests/test_calendar_determinism.py` — same events in shuffled provider
   order → identical title (canonical sort); equal-length tie → documented
   winner; aware/naive mix never raises (regression for the reproduced
   `TypeError`).
3. `tests/test_calendar_consent.py` — provider spy **not** constructed while
   `calendar_provider="off"`; enable → consent timestamp recorded; revoke →
   provider+path+closed status; read occurs only inside the start call (one
   read per start, window −6 h/+12 h asserted); **captured logs from a
   recording start contain no title/attendee strings**.
4. Fallback path — denied/unavailable/no-match → session title matches
   `Recording %b %d, %H:%M` pattern and recording completes (existing
   engine-test harness).
5. `tests/test_calendar_eds.py` (only if option 2 approved) — fake
   `CalendarFactory`/`Calendar` services on the private
   `dbus-daemon --session` bus (pattern: `tests/test_tray_sni.py:67`);
   service absent → `service_absent`; consent-off → no D-Bus call.
6. API/settings contract — GET/PUT round-trip of the three settings keys,
   422 on unknown `calendar_provider`, `calendar_status.reason` from the
   closed set only.
7. **Preservation:** full suite green with zero diffs under `macos/`;
   `tests/test_swift_sources.py` and `tests/test_meetings.py` unchanged
   except the additive `pick_event` regression.
8. Manual acceptance (Mark's laptop, later, with consent): probe output,
   consent prompt copy, one recording titled from a fixture ICS, one with
   access revoked → generic title. Not claimed by this card.

## 9. Out of scope / non-claims

Meeting detection and prompts are L3a (`t_f878fc98`); notification wording
coordination with L2 stays there — calendar consent lives only on the
Settings page. This card delivers a decision, not parity: no Linux calendar
feature exists yet, nothing was implemented, and none of the above should be
described as shipped until the approvals in §7 land and the §8 tests pass.
