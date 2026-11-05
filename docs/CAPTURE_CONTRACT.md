# Capture source and per-track failure contract (L0)

Status: **normative for Linux (PipeWire) capture.** This is the L0
foundation the per-app capture (L1) and device-lifecycle (L7) stories build
on: it fixes what a capture source *is*, how it identifies itself, what a
per-track state or error means, how pause and stop behave, and what partial
failure may and may not claim. The executable form is
[`listn/capture_contract.py`](../listn/capture_contract.py); both are pinned
to the producer by [`tests/test_capture_contract.py`](../tests/test_capture_contract.py)
and the fixtures under `tests/fixtures/capture_contract/`.

Scope: the Python engine's two-track path (`listn/pipewire.py`,
`listn/pipewire_live.py`, `listn record`, the Live tab, `listn tray`).
**No `macos/` behavior is defined or changed here** — the macOS capture
client keeps its own native contract; only the shared track ids and the
engine's ingest seam are common. Prototype status, hardware limits and the
observed PipeWire behaviours stay in [LINUX_CAPTURE](LINUX_CAPTURE.md).

## 1. Track roles

| Track | Frame id | Speaker label | Contents |
|---|---|---|---|
| `system` | 0 (`listn.ipc.TRACK_SYSTEM`) | **Others** | What everyone else says: whole system output today, one application once L1 lands |
| `mic` | 1 (`listn.ipc.TRACK_MIC`) | **Me** | The user's own microphone |

The ids and labels are the shared ingest-frame contract the macOS capture
client already uses, so pause, silence detection, diarization (Others only)
and transcription apply unchanged.

## 2. Capture source abstraction

### 2.1 Source kinds

Every capture target has exactly one **source kind**, derived from its
PipeWire media class — never guessed, `None` if the class is unknown:

| Kind | Media class | Meaning |
|---|---|---|
| `sink_monitor` | `Audio/Sink` | Whole-sink mode: the sink captured through its monitor ports (L0's system track) |
| `mic` | `Audio/Source` | A real input device (never a monitor, never a sink) |
| `app` | `Stream/Output/Audio` | One application's output stream — **reserved for L1**; L0 never selects it |

What each track may point at:

* **system track** → `sink_monitor` today, `app` once L1 lands (`SYSTEM_KINDS`);
* **mic track** → `mic` only (`MIC_KINDS`). A session default that names a
  sink is re-checked against the media class and refused/fallen back — a
  monitor is never offered as a microphone.

Whole-sink capture additionally passes `--properties stream.capture.sink=true`
unconditionally (see LINUX_CAPTURE behaviour 12): `--target` only *names* a
node and the binding is environment-dependent. That is an implementation
rule of `sink_monitor`, not a client option.

### 2.2 Stable identity

A target's **stable identity is its PipeWire node name** (`node.name`):

* `id` is session-scoped and **recycled** by PipeWire (observed: the same
  id reused with a new serial), so an id alone never identifies a target
  across time;
* `serial` (`object.serial`) accompanies the name: liveness is
  `pw-dump <id>` → select the element carrying that id → confirm serial
  name equality (LINUX_CAPTURE behaviours 4 and 7);
* persisted choices (tray target file, UI selections) store and re-resolve
  **names**. A remembered name that no longer exists re-adopts only the
  graph's own session default, never the first node seen.

`capture_contract.stable_identity(target)` returns that name.

### 2.3 Selection labels and the refusal rule

| Label | Meaning |
|---|---|
| `explicit` | The caller named this node (CLI flags, web/tray pickers) |
| `default` | The session default, re-checked against the media class |
| `fallback` | The highest-priority remaining node of the right class |

listn **never guesses** a capture target silently:

* the web UI and tray require *both* targets picked explicitly from
  `GET /api/pipewire/devices` and refuse Start with the server's reason
  otherwise (HTTP 400);
* the CLI may fall back, but the report and the printed line always carry
  the selection label, so a guess is never presented as a default;
* discovery finding nothing usable is a refusal, not an empty start.

### 2.4 Session-level declaration

A two-track run declares itself to the engine as
`{"source": "system", "mic": True}` → `LiveSession.start_ingest(...)`,
then the adapter stamps `device_info.kind = "pipewire"`,
`device_info.source = "system"` (the ingest source enum already reserves
`"app"` and `"device"` — L1 uses `"app"`), `device_info.mic = True`, and
`source_name = "PipeWire · <system node> + <mic node>"`.

## 3. Final report schema

`TwoTrackRecorder.stop()` returns one report; `listn record --json` prints
it, `POST /api/live/stop` serves it under `pipewire`, and the live status
embeds it as `report` after the capture ends.

Top level (`REPORT_KEYS`):

| Key | Type | Meaning |
|---|---|---|
| `ok` | bool | The run succeeded — policy in §5, never trust it without §5 |
| `out_dir` | str | Private local directory (mode 0700) holding the WAVs and `pw-record` logs |
| `rate` | int | Capture sample rate (48000) |
| `targets` | object | `system` / `mic` target descriptors (§2) |
| `tracks` | object | `system` / `mic` per-track reports |

Target (`TARGET_KEYS`): `id`, `serial`, `name`, `media_class`,
`description`, `track`, `selection`. `tracks.<name>.target` is always
equal to `targets.<name>`.

Per track (`TRACK_KEYS`):

| Key | Type | Meaning |
|---|---|---|
| `track` / `track_id` | str / int | Role name and frame id (must match §1) |
| `target` | object | The exact target this track recorded |
| `state` | str | `stopped` or `failed` — a final report only ever carries terminal states |
| `file` | str\|null | The track's WAV; non-null with frames when `stopped` |
| `duration` / `frames` / `samplerate` / `channels` | numbers | From the WAV itself, not the header's word |
| `rms_db` / `peak_db` / `fs_fraction` | number\|null | Measured level and share of samples at the format maximum |
| `verdict` | str\|null | `ok` / `quiet` / `silent` / `clipped` (§4) |
| `error` | str\|null | The cause; **always non-null when `state` is `failed`** |
| `exit_code` | int\|null | `pw-record`'s exit status |
| `log` | str\|null | This track's `pw-record` stderr log, if it exists |

Error *text* is human-readable and **not** a machine contract — clients
must branch on `state`, `verdict` and `ok`, never parse `error`.

## 4. Signal verdicts

From `analyze_wav()` (measured on the WAV's samples):

| Verdict | Rule |
|---|---|
| `silent` | peak below −60.0 dBFS |
| `quiet` | peak below −35.0 dBFS |
| `clipped` | ≥ 2% of samples at/above 32766/32768 (full scale) |
| `ok` | everything else |

An unfinalized/truncated/missing file is an analysis `error`, not a
verdict: the track fails. A clean stop of `pw-record` leaves exit code
0, 1 or 130 (observed 1 after SIGINT); anything else fails the track.

## 5. The `ok` policy and partial-failure rules

**Normative formula** (`capture_contract.expected_ok`):

```
ok = every track reached `stopped` with error == null
     AND mic verdict != "silent"
```

Consequences, all covered by tests and fixtures:

* a **silent system track alone stays ok** — nobody may have been playing
  anything; the engine ignores system silence too;
* a **silent microphone always fails** the run;
* a **clipped** track fails (its verdict stays visible, its state flips to
  `failed` with the measured full-scale fraction);
* **two identical tracks fail both** — one node's audio in both files means
  the independence guarantee broke, even when each per-track verdict reads
  `ok` (the measured real-hardware fallback, corr=1.000000);
* any `failed` track ⇒ `ok` false, whatever the other track did.

### Partial-failure matrix

| When | Behavior |
|---|---|
| **Start** | **All-or-nothing.** Both targets are liveness-probed before anything spawns; a spawn/validation failure rolls back what started, raises `CaptureError`, and leaves no empty run directory. CLI exits 3; the web refuses with the reason and no session exists. |
| **One track dies mid-run** | That track goes `failed` with its own cause (node disappeared, process exited, force-killed, no audio). **The other track keeps recording**; one `live.error` event is published for the dead track. The capture is *not* torn down behind the user's back. |
| **Both tracks dead** | The capture is aborted: both processes stopped, the session is stored as `error` with the combined cause, and the status stops saying "recording". |
| **Stop** | Both tracks are finalized to WAVs; whatever was captured is **retained** (per-track WAVs, engine session audio, transcription of what exists). Partial audio is never deleted because its sibling failed, and never laundered into `ok`. |

`Capture OK` (CLI), `ok: true` (JSON) and `state: stopped` (live status)
are therefore equivalent claims, and every one of them is refused by the
validator if any per-track fact disagrees.

## 6. Pause and stop semantics

**Pause is an engine-session state, not a capture-source state.** The
capture source has no paused state, and the live status must never claim
one (`"paused"` is absent from `LIVE_STATES`; a fixture pins the refusal):

* `POST /api/live/pause` pauses the session: transcription stops and the
  mix is written as silence, with a `[paused]` marker line on resume;
* **both `pw-record` streams keep running untouched** — pause never stops,
  reopens, restarts or retargets a stream, and the per-track WAVs keep
  recording what the sources produce;
* mid-run status during pause is still `state: "recording"` with both
  tracks `recording` and frame counters still climbing (asserted in tests);
* device re-selection on pause/resume must not happen silently — L7's rule
  "pause/resume does not reopen or switch unnoticed" starts here: today
  pause performs no device operation at all, which is exactly why nothing
  can switch unnoticed.

**Stop** is idempotent and ordered:

1. both tracks get `stop_requested`, `SIGINT` to both processes;
2. grace period (`STOP_GRACE_SECONDS`, 5 s) for readers to drain and WAVs
   to finalize; a process that ignores SIGINT is SIGKILLed and its track
   fails with *"…force-killed; the track is not trustworthy"*;
3. the report is computed once (a second `stop()` returns the same report);
4. the adapter stops the recorder **first**, then the engine session, so
   the report exists before the session is finalized.

Mid-run (`GET /api/live` → `pipewire`) uses `LIVE_STATES`:
`recording` (active, no report yet) → `stopping` (stop requested, report
not ready) → `stopped` (final report, `ok: true`) or `failed` (final
report, `ok: false`). The top-level state is derived only from
`(active, report)`; per-track states during those phases are
`pending`/`recording`/`stopped`/`failed`, plus `unknown` while no recorder
is attached yet.

## 7. API and CLI surface

| Surface | Contract |
|---|---|
| `GET /api/pipewire/devices` | Discovery only, never selects: `available`, `sinks`, `sources`, `apps` (L1: the `Stream/Output/Audio` candidates for the system track), `defaults`, `usable`, `error` |
| `POST /api/live/start` | Both targets required (400 with the refusal reason); 409 when a recording already runs; nothing starts on refusal |
| `GET /api/live` | `pipewire` object = live status (§6), `report` null until stopped |
| `POST /api/live/stop` | Response `pipewire` = the final report (§3) |
| `listn record --json` | Exit 0 = `ok`, 1 = report not ok, 2 = bad argument, 3 = unavailable/refused (nothing started) |

## 8. Extension points

Additive only; the `ok` policy (§5), the refusal rule (§2.3) and the
track ids (§1) are **not** extension points — changing them is a contract
change requiring doc + module + fixtures in one commit.

* **L1 (per-app capture):** add application-stream targets of kind `app`
  (`Stream/Output/Audio`) to `SYSTEM_KINDS` selection at runtime; identity
  stays the node name; declare `device_info.source = "app"`. A vanished or
  ambiguous app stream fails the system track with its cause — falling back
  to whole-sink capture silently is forbidden, just as a vanished sink node
  is today. App/window **titles are never logged or stored**; the app name
  may be session metadata only as the user-visible selection. Whole-sink
  mode (`sink_monitor`) remains fully supported.
* **L7 (device lifecycle):** node loss already fails the affected track
  (§5). L7 may add *additive* fields (e.g. a machine-readable `cause` on a
  track, device state in the live status) by extending the key tuples in
  `capture_contract.py` and this doc together; it may not introduce silent
  retargeting, a paused capture state, or a looser `ok`.
* **Changing any key set:** update `REPORT_KEYS` / `TRACK_KEYS` /
  `TARGET_KEYS` / `LIVE_*`, this document, and the fixtures in the same
  commit — `tests/test_capture_contract.py` compares them byte-for-byte
  (set-equality) and fails otherwise.

## 9. Privacy and local-only invariants

* The report and live status are **metadata only**: node names and
  descriptions, numeric analysis, states, and local file paths. No audio
  samples, PCM, transcripts, window/app titles, tokens or URLs — a test
  walks every real report and every fixture and fails on such a key or
  value.
* `out_dir` lives under the private data directory, mode 0700; the only
  files named are listn's own WAVs and `pw-record` stderr logs.
* No network, no telemetry: this contract and its module make no outbound
  calls; discovery and capture shell out to `pw-dump`/`pw-record` on this
  machine only.
* Tests use synthetic tones from stand-in executables; fixtures are
  hand-sized metadata (each fixture file < 20 KB), never recordings.

## 10. Evidence

| Claim | Test |
|---|---|
| Enums/thresholds match the producer | `test_contract_constants_match_the_producer` |
| Source kinds incl. reserved `app`, stable identity | `test_source_kind_derivation_*`, `test_stable_identity_*` |
| Canonical decisions (ok, silent mic, silent system, identical, clipped, partial) | `test_decision_fixture_matches_the_contract` |
| Validator refuses violations incl. a lying `ok` | `test_validator_refuses_contract_violations` |
| Real reports satisfy the contract and the `ok` policy | `test_real_*_report_*`, `test_partial_track_death_*` |
| Mid-run status shape, no paused capture state | `test_live_status_*`, `test_pause_is_not_a_capture_source_state` |
| Pause leaves both streams running; stop still honest | `test_pause_never_stops_or_retargets_capture_streams` |
| One track dying keeps the capture and the partial audio | `test_single_track_node_loss_keeps_capture_alive_and_reports_honestly` |
| Report/fixtures carry local metadata only | `test_real_report_carries_only_local_metadata`, `test_fixtures_carry_only_local_metadata` |
| This doc states the same enumerations | `test_contract_doc_states_the_enumerations` |

Hardware acceptance (real microphone, real Plasma session) is a separate
gate in [LINUX_CAPTURE](LINUX_CAPTURE.md) — none of the above claims it.
