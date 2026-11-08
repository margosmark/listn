# Linux device lifecycle (L7): hotplug, node loss and recovery

Status: **implemented and unit-tested with injected events; real-hardware
Plasma acceptance is NOT executed yet** (requirements and procedure below).
Builds on the L0 contract ([CAPTURE_CONTRACT](CAPTURE_CONTRACT.md)) and
never changes `macos/`. No cloud, no telemetry, no production audio in any
test or artifact.

## What the code does now

| Situation | Behaviour | Where |
|---|---|---|
| Selected node vanishes mid-run | That track **fails** with cause `node_lost`, its `pw-record` is stopped, the other track keeps recording, the partial WAV stays open and usable | `listn/pipewire.py` guard + `TwoTrackRecorder` |
| Device replugged (same node name) | Reported as `reconnectable: true` + `node_reappeared` event; **nothing is adopted** — the track stays `lost` until the user chooses resume | `listn/device_lifecycle.py` |
| No device of the required class left | Track reports `no_device`, `replacement_candidates` is empty | `listn/device_lifecycle.py` |
| Session default sink/source changes | `default_changed` event + `defaults.observed` updated; the running capture keeps its explicit target, and `defaults.remembered` only follows the change **while idle** | `listn/device_lifecycle.py` |
| PipeWire daemon/tools/permission unavailable | `unavailable` with cause `daemon_unavailable` (never misreported as a dead device); `daemon_restored` when it comes back | `listn/device_lifecycle.py` |
| Mic signal is silent | `signal: "silent"` and state `silent` (engine's `live.silent` verdict folded in) | `listn/pipewire_live.py` |
| User asks to resume on a chosen device | `POST /api/live/replace {"track","node"}` → validated → one new stream on the same WAV, with the outage padded as silence so both tracks stay on one timeline | `listn/server.py`, `PipeWireLive.replace_track`, `TwoTrackRecorder.replace_track` |
| Repeated failures | **Bounded**: `max_recovery_attempts` (default 2) per track per capture; then `exhausted` and every further request is refused with a reason | `listn/device_lifecycle.py` |

**No silent retarget, ever.** `devices.retargets` counts committed
replacements; it is `0` after every automatic event (node loss, hotplug,
default change, daemon loss, pause/resume). The only path that moves a
target is an explicit `begin_replacement` → spawn → `commit_replacement`,
and a failed spawn is rolled back as `recovery_refused` without consuming
the budget.

**Pause/resume cannot switch anything.** Pause is engine-session only
(L0 §6): neither `listn.device_lifecycle` nor the recorder has a pause
operation, no stream is reopened, and a replacement request while paused
is refused with `409 Resume the recording before replacing a device.`

**Never two streams.** A replacement is refused while the track already
has a live `pw-record`, while its WAV was already finalized, while the
sibling track is not recording (L0's both-dead abort), for a node of the
wrong source kind, for the sibling's own target, or for a node that is not
in the graph.

## Where the state shows up

* `GET /api/live` → `pipewire.devices` (and `GET /api/dashboard` → `pipewire.devices`):

  ```json
  "devices": {
    "active": true,
    "tracks": {"system": {"state": "lost", "cause": "node_lost",
                          "node": "alsa_output.pci-0000.analog-stereo",
                          "media_class": "Audio/Sink", "signal": null,
                          "reconnectable": true, "recovery_attempts": 0,
                          "max_recovery_attempts": 2,
                          "replacement_candidates": [{"name": "…", "media_class": "Audio/Sink", "description": "…"}],
                          "detail": "target node '…' disappeared …"}},
               "mic": {…}},
    "defaults": {"observed": {"sink": "…", "source": "…"},
                 "remembered": {"sink": "…", "source": "…"}},
    "events": [{"t": 1.2, "kind": "node_lost", "track": "system", "node": "…", "detail": "…"}],
    "retargets": 0
  }
  ```

  States: `recording`, `stopped`, `lost`, `no_device`, `unavailable`,
  `recovering`, `exhausted`, `replaced`, `silent`. Causes: `node_lost`,
  `process_exit`, `daemon_unavailable`, `no_device`, `silent_signal`,
  `recovery_exhausted`, `recovery_in_progress`. The shape is validated by
  `listn.capture_contract.validate_live_status` (keys are contract
  surface: `LIVE_DEVICES_KEYS`, `LIVE_DEVICE_TRACK_KEYS`,
  `LIVE_DEVICE_DEFAULT_KEYS`, `DEVICE_EVENT_KEYS`).
* Tray: one line only when something needs attention —
  `Recording — 02:15 · system 96000f — system: device lost` (plus
  `N device replacements` after a resume).
* Report: unchanged L0 schema; a partial track keeps `state: failed`,
  its `error` text, and `devices` keeps the machine cause after the stop.

## Tests (all synthetic, no device, no production audio)

`tests/test_device_lifecycle.py` (23 tests):

* injected process/graph events: node loss, process exit, duplicate-death
  de-duplication, daemon loss/restore, default change (active vs idle),
  replug → offered not adopted, `no_device`, silent signal, stop keeps the
  cause;
* recovery rules: every refusal code, sibling-dead, bounded budget →
  `exhausted`, committed replacement counted, `retargets` stays `0` for
  all automatic events;
* contract: status shapes are set-equal to the contract key tuples, a real
  status validates and a lying `devices.active` is refused, metadata-only
  (no transcript/title/window/URL/PCM keys);
* recorder end-to-end with the stand-in `pw-record`: replacement resumes
  the **same** `system.wav` and pads the gap (frames + ≥4800, `ok: true`),
  refuses a live track (no second stream), wrong kind/missing node, and a
  dead sibling.

## Actual Plasma hardware requirements (acceptance still open)

The following needs a real KDE Plasma session on real hardware. It is
**not** claimed by the tests above, and running it must not capture or
retain anyone's real conversations — use generated tones or a silent
session, and delete any recording afterwards.

| # | Step | Required hardware/session | Expected observable |
|---|---|---|---|
| 1 | Unplug the USB/Bluetooth microphone mid-recording | Plasma session, PipeWire + WirePlumber, one real USB or BT mic, audible output | mic track fails with `node_lost`, system track continues, tray says `mic: device lost`, `POST /api/live/replace` offers the remaining source |
| 2 | Replug the same microphone | as #1 | `node_reappeared`, `reconnectable: true`, **no** audio from it until the user posts `/api/live/replace` |
| 3 | Switch the output device (sink change) in Plasma | two real outputs (e.g. speakers + HDMI/USB DAC) | `default_changed` event; recorded target unchanged; tray/API still `recording` |
| 4 | Change the session default source while recording | two real inputs | same as #3; `defaults.remembered` untouched while recording, updated when idle |
| 5 | Bluetooth headset profile change (A2DP ↔ HFP) and headset disconnect | a real BT headset paired to the session | explicit state (`lost`/`unavailable`) and a truthful report; never a green run on the wrong device |
| 6 | Suspend/resume the machine mid-recording | laptop with working suspend | after resume: explicit device state or an honest failed capture; no duplicate `pw-record` processes (`pgrep -a pw-record` shows at most one per track) |
| 7 | Kill `pipewire`/`wireplumber` mid-recording | any of the above | `unavailable` / `daemon_unavailable`, not "node lost"; `daemon_restored` on restart |

Recording versions and outcomes only (distro/kernel, Plasma, PipeWire,
WirePlumber, pass/fail per step) — never the audio. A headless null-sink
rig does **not** substitute for this gate.
