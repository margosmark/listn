# Linux capture prototype (PipeWire)

Status: **prototype with automated tests and one real-daemon verification on a
headless machine — not validated on real audio hardware or a KDE session.**
The real-daemon verification is reproducible in one command
(`scripts/linux_capture_rig.sh`, see
[Reproduce the real-daemon rig](#reproduce-the-real-daemon-rig)).
A real-hardware routing defect found through this verification path —
both tracks recording the session default source instead of the sink
monitor (reproduced and A/B-tested on a CachyOS/KDE laptop, see
[observed behaviour 12](#observed-pipewire-142-behaviour-why-the-code-is-shaped-this-way))
— is fixed in the current code: sink targets request monitor capture
explicitly, and the report refuses to call a capture successful when the
two tracks are the same waveform or a track is pinned at full scale.
**The acceptance rerun of the fixed app on real hardware is pending.**
This implements roadmap step 2 in [DECISIONS](DECISIONS.md) as a bounded
increment; it does not ship KDE desktop support. The recorder is now wired
into the app: `listn record` on the CLI, the Live tab of the local web UI,
and the KDE Plasma tray (`listn tray`) all start two-track captures through
the same code paths.

## What it is

The normative contract for this capture path — source kinds and stable
identity, per-track states/errors, the `ok` policy, pause/stop and
partial-failure rules, and the L1/L7 extension points — is
[CAPTURE_CONTRACT.md](CAPTURE_CONTRACT.md); its executable form is
`listn/capture_contract.py`, pinned by `tests/test_capture_contract.py`.

`listn/pipewire.py` records the system output and the microphone as two
independent tracks through PipeWire, the audio stack KDE Plasma uses:

- `discover()` reads the graph with `pw-dump`; `select_targets()` picks an
  `Audio/Sink` for the system track (recorded through the sink's monitor
  ports) and an `Audio/Source` for the microphone. Session defaults are
  re-checked against the node's media class before use, explicit `--system`
  / `--mic` selectors are validated, and selection is labelled `default`,
  `fallback` or `explicit` in every report so a guess is never presented as
  a default.
- The system track can instead point at **one application's output stream**
  (`Stream/Output/Audio`) — see
  [Per-application capture (L1)](#per-application-capture-l1). The
  microphone is a separate track either way, and the whole-sink default is
  unchanged.
- `TwoTrackRecorder` runs one `pw-record` per track (raw int16 PCM on
  stdout) into its own WAV file — the system track's stream additionally
  passes `--properties stream.capture.sink=true`, because `--target`
  only names a node and a sink target without that property can bind to
  the session's default source instead of the monitor (behaviour 12) —
  and can hand each block to an `on_block` callback. That callback
  matches the engine's
  `LiveSession.push_audio(track, sample_rate, samples)` — the same seam the
  macOS capture client uses — so the existing ingest pipeline, separate
  `Me`/`Others` tracks, pause semantics and silence detection apply
  unchanged when this is wired up.
- Each track's target node is watched while recording (`pw-dump <id>` plus
  serial/name identity). If the node disappears the track fails and its
  process is stopped — see the observed behaviours below for why that is
  necessary.

Run it:

```sh
uv sync --group dev
uv run python -m listn.pipewire --dry-run          # discovery only (module probe)
uv run python -m listn.pipewire --duration 5       # record both tracks
uv run python -m listn.pipewire --duration 5 --json > report.json

uv run listn record --list                         # discovered nodes, app CLI
uv run listn record --system-target <node> --mic-target <node> --duration 5
uv run listn record --system-target <app-stream> --mic-target <node> --duration 5
                                                   # ^ one application only (L1)
uv run listn ui --no-browser                       # web UI: pick both targets in
                                                   # the Live tab, then Start

scripts/linux_capture_rig.sh                       # the whole headless rig, one
                                                   # command (see below)

uv run pytest tests/test_pipewire.py               # module tests
uv run pytest tests/test_pipewire_wiring.py        # CLI/server wiring tests
uv run pytest tests/test_pipewire_app.py           # L1 per-application tests
LISTN_RIG_IT=1 uv run pytest tests/test_capture_rig.py  # full daemon rig
```

The app CLI and the web UI go through `listn/pipewire_live.py`, which only
adapts the module: `discover()` and `select_targets()` resolve targets (the
web UI requires both to be picked explicitly from `GET /api/pipewire/devices`
and never starts when discovery returns nothing usable), and
`TwoTrackRecorder` feeds every block into the engine's ingest seam
(`LiveSession.push_audio`), so the session gets the independent system/mic
tracks, pause semantics and transcription unchanged. A completed web run
reports the recorder's own per-track WAV analysis (state, duration, verdict,
peak/RMS, error) — never a bare "saved".

- Wiring (this increment): `uv run pytest` — **267 passed** (254 pre-existing
  + 13 new tests in `tests/test_pipewire_wiring.py` covering the CLI and
  server paths with stand-in `pw-dump`/`pw-record` executables, including
  the id-probe contract and every refusal path).
- Real-daemon wiring run on this machine (same headless daemon as above, a
  440 Hz tone piped into the sink and into the pipe-tunnel source — the rig is
  now scripted, see [Reproduce the real-daemon rig](#reproduce-the-real-daemon-rig)
  below):

  ```text
  $ uv run listn record --list
  System outputs (Audio/Sink):
    listn.test.system  (id 34) — Test System Output  [session default]
  Microphones (Audio/Source):
    listn.test.mic  (id 38) — Test Mic Tone

  $ uv run listn record --system-target listn.test.system --mic-target listn.test.mic --duration 4 --json
  ok: true, CLI exit 0
  system: state=stopped 4.011 s 192512 frames peak=-10.5 dBFS rms=-13.5 verdict=ok
  mic:    state=stopped 3.989 s 191488 frames peak=-7.4  dBFS rms=-11.0 verdict=ok

  $ curl -X POST /api/live/start -d '{"capture":"pipewire",
      "system_target":"listn.test.system","mic_target":"listn.test.mic"}'
  state=recording; mid-run GET /api/live: system 293888 frames, mic 294912 frames
  POST /api/live/stop -> report ok=true; session audio-system.wav 197600 B,
  audio-mic.wav 198738 B; transcripts empty (a pure tone carries no speech —
  expected). Starts without both explicit targets are refused with 400.
  ```

## Per-application capture (L1)

The system track has two modes, and which one a run used is never a guess
— it is `targets.system.media_class` in every report:

| Mode | Target node | Records | `media_class` → source kind |
|---|---|---|---|
| **All applications** (default) | an `Audio/Sink` | everything mixed at that sink: every application, system sounds, notifications | `Audio/Sink` → `sink_monitor` |
| **One application** | one `Stream/Output/Audio` | that application's output stream and nothing else | `Stream/Output/Audio` → `app` |

With no `--system-target` / `system_target` the whole-sink default
applies, unchanged — that *is* the "all applications" mode. The
microphone is an independent track in both modes (`Audio/Source`, kind
`mic`) and is not affected by which mode is chosen. Selecting one
application therefore **excludes** every other application: there is no
"the selected app plus whatever else is playing" state.

### Enumerating and selecting

* `discover()` parses the graph; `Graph.app_streams` returns **only**
  `Stream/Output/Audio` nodes. Sinks, sources, monitor sources and
  `Stream/Input/Audio` capture streams (listn's own `pw-record` among
  them) are excluded by construction, so a capture stream can never be
  mistaken for application audio.
* The stable identity is the **node name** — L0
  [capture contract](CAPTURE_CONTRACT.md) §2.2. An explicit selector
  resolves in order: node id → exact node name → the application's own
  `application.name` (case-insensitive). That last step is taken only
  when **exactly one** stream carries the name; ids and PIDs churn, so
  neither is ever the identity.
* *Binding* is separate from identity: an application stream is handed
  to `pw-record` as its unique **`object.serial`** (`--target` accepts
  "serial or name", and digits resolve as serials — measured below),
  because a `node.name` may be carried by several nodes and PipeWire
  binds the first-created match. Sinks and microphones keep the measured
  name-based binding, so a shared name among *those* is refused instead
  (matrix below).
* The user-visible label is the application's own name
  (`application.name`, falling back to `application.process.binary`).
  **Window, media and stream titles are never read**: a stream node's
  `node.description`, `media.name` and `window.title` routinely carry
  what is playing (VLC publishes the track title there), and
  `parse_graph()` deliberately does not copy them into the node or the
  report.
* The same list everywhere: `listn record --list`,
  `GET /api/pipewire/devices` (`apps`), and the Live tab's target picker.

### Refusals — a target is never silently replaced

`pw-record --target` accepts a **serial or a name**, and digits resolve
as *serials*: an unknown serial is not an error either — PipeWire 1.4.2
records something else instead (observed behaviour 1, re-measured for
streams below — an unknown target here captured the **whole sink**; a
registry *id* passed as digits does not resolve and falls back the same
way). A *name* it cannot match uniquely behaves the same: it binds the
first-created node carrying that name. listn therefore binds application
streams by their unique `object.serial` and refuses any name-bound
target whose name is shared. So:

| Situation | Behaviour |
|---|---|
| Selector matches nothing | `CaptureError`: CLI exit 3, HTTP 400, no recording directory. The refusal lists the nodes seen *and* the application streams available. The default sink is **not** adopted. |
| Selector matches several nodes (duplicate `node.name`) | Refused as ambiguous, naming every match. |
| Application name matches several streams | Refused as ambiguous, naming each node — one application holding several streams is a common case, and taking the first would be a guess. |
| Resolved node's `node.name` is shared with another node, and the target is **name-bound** (sink, microphone, or an application stream without a serial) | Refused *after* resolution of an explicit selector: the selector was unique but the name `pw-record` would bind is not, and it binds the first-created match. CLI exit 3, HTTP 400, no recording directory. (A default/fallback target with no selector is the session's own metadata choice and is not re-checked here.) |
| Application stream target with a duplicated `node.name` | **Not** a refusal: bound by the node's unique `object.serial`, so id and application-name selectors record exactly the selected stream — measured below ("duplicated `node.name`" lines). |
| Selected stream gone before the spawn | `TwoTrackRecorder.start()`'s liveness probe fails; start is all-or-nothing (L0 §5), nothing runs. |
| Selected stream closes mid-recording | That track goes `failed` with *"the application stream '…' closed while recording (its node left the PipeWire graph)"*, the **microphone keeps recording**, partial audio is retained and `ok` is false (L0 §5 partial-failure matrix). Falling back to the whole sink is forbidden. |
| Selector names a sink | Whole-sink mode with `selection: explicit` — a visible choice, not a fallback. |

A selected application that is simply **not playing yet** produces a
system track with `verdict: "silent"` and `state: "stopped"` — never
`verdict: "ok"`. The run itself stays `ok` because the microphone
carries signal: L0 §5 keeps a silent *system* track from failing the run
("nobody may have been playing anything"), and the contract validator
refuses any other combination. The honest claim is the per-track
verdict.

### What was measured for this increment

Headless Debian LXC, PipeWire 1.4.2 + WirePlumber 0.5.8, no `/dev/snd`.
Two `pw-cat --playback` processes stand in for two applications, each
publishing a real `Stream/Output/Audio` node, playing 700 Hz and 1300 Hz
into a null sink; each line below is the dominant-bin analysis of the
produced WAV with its three strongest 10 Hz-spaced bins:

```text
pw-dump → listn.probe.sink   Audio/Sink
           listn.probe.appA  Stream/Output/Audio
           listn.probe.appB  Stream/Output/Audio

pw-record --target listn.probe.appA → recA.wav
  4.01 s  dominant 700.0 Hz   top=[700Hz x1.00, 710Hz x0.00, 690Hz x0.00]
pw-record --target listn.probe.appB → recB.wav
  4.01 s  dominant 1300.0 Hz  top=[1300Hz x1.00, 1290Hz x0.00, 1310Hz x0.00]
pw-record --target listn.probe.sink --properties stream.capture.sink=true
  2.99 s  dominant 700.0 Hz   top=[700Hz x1.00, 1300Hz x1.00, 690Hz x0.00]
pw-record --target listn.probe.nothing   # a name that does not exist
  2.97 s  dominant 1300.0 Hz  top=[1300Hz x1.00, 700Hz x1.00, 690Hz x0.00]
```

Duplicated `node.name` — two streams, both `node.name=qa.f1.dup`,
`application.name=DupAppA` 700 Hz created first, `DupAppB` 1300 Hz
second, mic tunnel 1000 Hz; DupAppB was `id=58 serial=59`:

```text
listn record --system-target 58 --mic-target <mic> …      # id of DupAppB
  system dominant 1300.0 Hz   700 Hz and the mic >40 dB below   (rc 0, report ok)
listn record --system-target DupAppB --mic-target <mic> …  # application name
  system dominant 1300.0 Hz   700 Hz >40 dB below              (rc 0, report ok)
listn record --system-target qa.f1.dup …                   # the shared name itself
  rc 3  "'qa.f1.dup' names 2 PipeWire nodes: …"  no recording directory
raw pw-record --target 59   # DupAppB's serial → 1300 Hz only, 700 >40 dB below
raw pw-record --target 58   # DupAppB's *id*    → both tones: ids do not resolve,
                            # an unknown serial falls back to the whole sink
raw pw-record --target qa.f1.dup → 700 Hz only: the first-created match,
                            which is exactly the capture the pre-fix listn
                            reported as DupAppB's (QA F1, 5/5)
```

Reading:

* **duplicated `node.names` no longer redirect the capture** — before
  the fix the id and application-name selectors recorded DupAppA's
  700 Hz while `report.ok` claimed DupAppB; binding by the unique
  `object.serial` puts the selected stream's 1300 Hz in the WAV (both
  selectors, both >40 dB rejection of the other tone), and a selector
  that *is* the shared name still refuses with exit 3 and no recording
  directory;

* **per-app isolation is real** — the selected stream's capture holds
  only its own tone (the other tone's bin is `x0.00`, i.e. below the
  noise floor), while the whole-sink capture holds **both** tones. That
  is the "all applications" vs "one application" distinction measured,
  not asserted;
* **an unknown target silently becomes whole-sink capture** — the last
  line recorded the sink's mix without a word. This is exactly the
  failure listn must not ship: it is why an explicit selector is
  validated before anything spawns and re-checked every second while
  recording, and why a vanished stream fails the track instead of
  carrying on.

This is *mechanism* evidence on a headless daemon with synthetic tones.
It is **not** the hardware acceptance below, and it used no private or
production audio.

### Acceptance on a real Plasma session (not yet run)

Required for the L1 release gate; the headless rig above does not
substitute for it. Run on the CachyOS/KDE laptop with a real microphone
and audible output, using two **desktop applications** (not `pw-cat`),
each playing a generated tone:

```sh
# tones only — no recordings, no copyrighted/private audio
python - <<'EOF'
import math, wave, numpy as np
for name, hz in (("tone440", 440.0), ("tone1300", 1300.0)):
    n = int(30 * 48000); t = np.arange(n) / 48000
    s = (0.4 * 32767 * np.sin(2 * math.pi * hz * t)).astype("<i2")
    w = wave.open(f"/tmp/{name}.wav", "wb")
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes(np.column_stack([s, s]).tobytes()); w.close()
EOF
ffplay -nodisp -autoexit /tmp/tone440.wav   &   # app A
mpv --no-video /tmp/tone1300.wav            &   # app B
pw-dump | jq -r '.[] | select(.info.props["media.class"]=="Stream/Output/Audio")
                 | .info.props["node.name"]'         # expect both streams
```

Criteria — record distro/kernel, Plasma, PipeWire and WirePlumber
versions and the pass/fail of each; record outcomes, never audio:

1. **Enumeration.** `listn record --list` shows both streams under
   *Application output streams*, labelled with the application name, and
   shows no window or media title anywhere.
2. **Isolation A.** `listn record --system-target <appA-node>
   --mic-target <default mic> --duration 8 --json` exits 0;
   `targets.system.media_class == "Stream/Output/Audio"`,
   `selection == "explicit"`; the produced `system.wav` has its dominant
   frequency within ±2 Hz of 440 Hz **and** no 1300 Hz peak above the
   noise floor (same check `scripts/linux_capture_rig.sh` performs).
   Both applications stay audible on the speakers during the capture —
   the tap is passive, not a reroute.
3. **Isolation B.** Repeat targeting app B: dominant within ±2 Hz of
   1300 Hz, no 440 Hz peak.
4. **Mic stays separate.** In both runs the `mic` track is an independent
   `Audio/Source` capture: identical-sample share ≤ 25 % and |corr| ≤ 0.99
   against `system.wav`, and speaking into the microphone appears only in
   `mic.wav`.
5. **App closed mid-run.** Quit app A while recording: within one watch
   interval the `system` track reads `state: failed` with the
   "application stream … closed" cause, the `mic` track keeps
   recording, `ok` is false, and both WAVs are retained.
6. **Ambiguous selection.** With two streams sharing an application name
   (e.g. two browser tabs), selecting by that name refuses with
   "names N PipeWire nodes" and does not pick the first.
7. **Stale selection.** After app A quits, re-running with its node name
   exits 3, creates no recording directory, and does not adopt the sink.
8. **Whole-sink regression.** With no `--system-target`, the report says
   `Audio/Sink` / `sink_monitor` and `system.wav` contains **both** tones.
9. **Privacy.** No window/media title appears in the report, the CLI
   output, the logs or the session metadata; nothing is written outside
   the local data directory; no network calls.

None of 1–9 has been run on real hardware yet — the synthetic graph
tests in `tests/test_pipewire_app.py` and the headless measurements above
cover the mechanism only.

## Reproduce the real-daemon rig

`scripts/linux_capture_rig.sh` regenerates the real-daemon results above from
a clean checkout on any headless Linux machine with PipeWire installed — no
`/dev/snd`, no desktop session, no audio hardware:

```sh
scripts/linux_capture_rig.sh                 # 4 s capture; exit 0 only if both verdicts are ok
                                             # AND the tracks differ and each holds its own tone
scripts/linux_capture_rig.sh --duration 8 --out ./rig-artifacts   # also keep WAVs + report + log
LISTN_CMD="uv run listn" scripts/linux_capture_rig.sh             # app CLI via uv instead of `python3 -m listn`
scripts/linux_capture_rig.sh --list          # stand the rig up, run `listn record --list`, tear down
```

Everything is built inside a fresh `mktemp -d` (private `XDG_RUNTIME_DIR` +
`XDG_CONFIG_HOME`) and removed on exit:

1. a `pipewire.conf.d` drop-in adds a `support.null-audio-sink` adapter as
   the system sink (`listn.rig.system`) and loads
   `libpipewire-module-pipe-tunnel` with `tunnel.mode=source` +
   `pipe.filename` (a FIFO in the rig dir) as the microphone source
   (`listn.rig.mic`);
2. headless `pipewire` + `wireplumber` start against that private config;
3. a paced writer feeds a 1000 Hz s16le/48 kHz/mono tone into the FIFO,
   holding the write end open continuously (an empty FIFO reads as EOF and
   the source goes silent — observed behaviour 10 below);
4. `pw-cat` plays a 440 Hz WAV into the null sink while the app's own
   `listn record … --json` captures both tracks (different frequencies on
   purpose: distinctness and target correctness become deterministic to
   assert);
5. per-track `state/duration/frames/peak/rms/verdict` and each WAV's
   dominant frequency are printed, then three cross-checks run on the
   produced files: each track must hold its own source's tone (440 Hz in
   `system.wav`, 1000 Hz in `mic.wav`, ±2 Hz), and the two tracks must
   differ (identical share ≤25% **and** `|corr|` ≤ 0.99 over the
   overlapping span). Only then does the script print `rig result: OK`;
   then every process is killed and the private runtime dir removed.

Exit codes: `0` both tracks `verdict=ok` with signal **and** distinct and
correctly targeted, `1` a track was not ok (silent/quiet/failed), a tone
landed in the wrong track, or the two WAVs were (nearly) identical — the
script refuses instead of claiming success, which is exactly how the
identical-tracks regression escaped the first time — `3` the rig itself
could not start (missing tools; daemon or nodes did not appear).

Measured on this machine (PipeWire 1.4.2, WirePlumber 0.5.8, headless
Debian LXC, `LISTN_CMD=".venv/bin/python -m listn"`), 2026-10-31:

```text
$ scripts/linux_capture_rig.sh --duration 4
rig: daemon up, nodes ready:
rig:   listn.rig.system Audio/Sink
rig:   listn.rig.mic Audio/Source
system: state=stopped 3.968 s 190464 frames peak=-8.0 dBFS rms=-11.0 dBFS verdict=ok dominant=440.0 Hz
mic: state=stopped 3.989 s 191488 frames peak=-14.0 dBFS rms=-17.0 dBFS verdict=ok dominant=1000.0 Hz
cross-track: identical=0.56% corr=-0.000044 over 190464 samples -> distinct tracks
rig result: OK — both tracks recorded with real signal and differ   # exit 0
```

Negative checks of the rig itself, run while writing it and reproducible by
hand: feeding digital silence into the FIFO yields
`mic: … peak=-120.0 dBFS verdict=silent` and exit 1; removing WirePlumber
makes both `pw-record` streams produce no audio and the script refuses with
exit 1; `LISTN_CMD="uv run --frozen listn"` behaves identically to the
default. The cross-track refusal is covered under the same gate: a
wrapper plants identical WAVs after a clean recording and the rig must
exit 1 printing `SAME SIGNAL in both tracks` (per-track `verdict=ok`
lines alone would still read green — that is the failure mode this check
exists for). The full end-to-end run is also gated into pytest:
`LISTN_RIG_IT=1 uv run pytest tests/test_capture_rig.py`.

Exit codes: `0` capture succeeded with microphone signal, `1` capture failed
or the microphone recorded only silence, `3` PipeWire/tools unavailable or
no usable node. Reports always carry per-track `state`, `verdict`
(`ok`/`quiet`/`silent`/`clipped`), `duration`, `error`, `fs_fraction` and
`exit_code`; silence is never reported as success, and a silent system
track alone does not fail the run (nobody may have been playing anything)
— the same rule the engine uses. Two further refusals run in the normal
report path: a track pinned at the format maximum across ≥2% of its
samples is marked `clipped` and failed with the measured full-scale
fraction, and two tracks containing the same waveform fail both tracks
with `system and mic tracks are identical: <corr and identical-share>` —
`Capture OK` is printed only when every check passes.

## Verified on this machine

- `uv run pytest` — **251 passed** (219 pre-existing + 32 new tests in
  `tests/test_pipewire.py`), Python 3.12, Linux. The new tests use a fake
  `pw-record` executable and synthetic audio; no daemon and no hardware.
- Real PipeWire 1.4.2 run in this repository's CI-style container (headless
  Debian LXC, `pipewire` + `wireplumber` started manually with two null
  sinks and an `audiotestsrc` tone source, no `/dev/snd`):

  ```text
  $ python -m listn.pipewire --dry-run
  system: listn.test.system (Audio/Sink, id 32, selection: default)
  mic: listn.test.mic (Audio/Source, id 34, selection: fallback)

  $ python -m listn.pipewire --duration 4 --json   # 440 Hz tone played into the sink
  ok: True
  system: state=stopped 4.011s verdict=ok peak=-8.0 dBFS  (dominant 440.0 Hz)
  mic:    state=stopped 3.989s verdict=ok peak=-0.0 dBFS  (dominant 440.0 Hz)

  $ python -m listn.pipewire --dry-run --system no.such.node
  PipeWire capture unavailable: No PipeWire node named 'no.such.node'. ...
  rc=3
  ```

  Two independent tracks were captured simultaneously from a real PipeWire
  daemon, both WAVs finalized and spectrally checked (440 Hz present in
  each). `mic: selection: fallback` is the module refusing the session
  default, which named a sink on this rig (see below). This hand-built
  container rig (two null sinks + an `audiotestsrc` adapter) was never
  scripted, and the `audiotestsrc` SPA factory is commented out of the
  default `context.spa-libs` on this stack, so the scripted
  [capture rig](#reproduce-the-real-daemon-rig) builds the microphone side
  as a pipe-tunnel source instead — that is the reproducible path.

## Observed PipeWire 1.4.2 behaviour (why the code is shaped this way)

Measured against a headless PipeWire 1.4.2 + WirePlumber 0.5.8 daemon:

1. **`pw-record --target <unknown>` does not fail.** It silently records a
   fallback source instead — a mistyped or vanished target still produces a
   valid-looking WAV with someone else's audio. Discovery therefore resolves
   targets itself, and a spawn is preceded by a liveness probe.
2. **A destroyed target does not stop `pw-record`.** It keeps running and
   keeps writing (silently re-targeted after the node vanished). That is why
   each track's node is polled during recording and the track is failed and
   stopped when its node is gone.
3. **`default.audio.source` may name a sink.** WirePlumber on a machine
   without microphones published an `Audio/Sink` as the default source;
   `select_targets()` re-checks media classes and falls back to a real
   `Audio/Source`.
4. **`pw-dump <id>` prints nothing and exits 0 for a vanished id**, and ids
   are recycled (a second `pw-record` reused id 56 with serial 108 → 117), so
   liveness is `pw-dump <id>` plus serial/name identity.
5. **Exit codes:** `pw-record` exits 1 after SIGINT while still finalizing
   its file; SIGKILL leaves `RIFF` size 8 / `data` size 0, which the stdlib
   `wave` module refuses to open. The module treats 0/1/130 after a stop
   signal as clean, escalates to SIGKILL after a grace period, and parses
   WAV headers manually so an unfinalized file is reported, not crashed on.
6. **Sink-monitor capture works by targeting the sink node itself** (no
   separate `.monitor` node exists in native `pw-dump` output) — *on this
   headless rig*. On real desktop sessions the same invocation may not
   bind to the monitor at all; see behaviour 12, which is why the
   recorder also passes `stream.capture.sink=true`.
7. **`pw-dump <id>` can return more than the queried object.** Measured on a
   bare headless 1.4.2 daemon (no WirePlumber): with node ids and object
   serials drifted apart, `pw-dump 40` returned two objects — a node with
   `id=37, object.serial=40` first, then the requested `id=40` node — so an
   object whose `object.serial` merely equals the queried id can lead the
   dump. Reading `payload[0]` compares that stranger's serial and reports a
   healthy node as gone (issue #4), which blocks `start()` outright. The
   probe therefore selects the element whose `id` equals the node's id and
   only then checks serial/name; no element with that id means gone.
8. **`pw-cli create-node` fails silently.** On this build (PipeWire 1.4.2,
   Debian package) unquoted property values fail client-side — `media.class=
   Audio/Sink` is rejected with `syntax error in properties … Expected
   object value` at the `/`, so values must be JSON — and with JSON-quoted
   values the command returns rc=0, prints nothing, and creates no node at
   all, with no error in the daemon log. The rig therefore configures its
   nodes in a `pipewire.conf.d` drop-in (loaded at daemon start, where
   failures are fatal and visible) instead of driving `pw-cli`.
9. **`libpipewire-module-pipe-tunnel` silently ignores wrong property
   names.** The path property is `pipe.filename` (not `path`), the direction
   is `tunnel.mode=source` (not `media.class=Audio/Source`), and the stream
   format is `audio.format`/`audio.rate`/`audio.channels` (not
   `format`/`rate`/`channels`). Unknown keys are dropped without a word and
   the module falls back to its defaults — `/tmp/fifo_output`, S16LE stereo
   48 kHz — so `pw-cli load-module … path=…` returns rc=0 and produces a
   node that carries nothing useful. This is why earlier hand-typed
   invocations "loaded fine" yet yielded no recordable microphone.
10. **A pipe-tunnel source goes silent if its FIFO ever reads empty.** The
    capture side reads the FIFO non-blocking; a momentarily empty pipe is
    treated as end-of-stream — the node exists, `pw-record` records exactly
    1024 zero frames and then nothing (`underrun 0 < 4096` in the daemon
    log), and this happens with *and* without WirePlumber. The rig's writer
    therefore holds the write end open continuously and paces at ~95 % of
    realtime. Flooding the FIFO instead triggers `resync` log spam and a
    measurable pitch error (dominant 430 Hz instead of 440 Hz).
11. **WirePlumber is load-bearing on a rig with no real devices.** With the
    same drop-in config and daemon but no session manager, both `pw-record`
    streams end with `recording was not finalized and contains no audio` —
    even the null sink's monitor port carries no buffers. The rig starts
    WirePlumber next to the daemon; a real desktop run inherits a session
    manager anyway.
12. **`pw-record --target <sink>` is only a *name*: without
    `stream.capture.sink=true` it can bind to the session default source
    with no warning.** Measured on Mark's CachyOS/KDE laptop (operator's
    raw A/B with two simultaneous `pw-record` processes and no listn code
    involved, 2026-10-30): with the app's exact args, the sink-targeted
    and mic-targeted captures came back corr=1.000000, 96.6% identical
    samples, rms(s-m)=0.000003, both `*.pw-record.log` files empty — the
    "system" track had recorded the microphone. Adding
    `--properties stream.capture.sink=true` to the sink-targeted
    invocation only: corr=-0.004773, 0.0% identical samples, system peak
    -8.77 dBFS vs mic -9.87 dBFS — two independent tracks. The headless
    rig links the same argv *without* the property correctly (behaviour
    6), so the routing is environment-dependent and cannot be inferred:
    the property is passed unconditionally, the rig asserts track
    distinctness and per-track tone placement, and `report()` refuses any
    capture whose two tracks are the same waveform.

## Hardware limitations of this verification

- The machine is a headless Proxmox LXC container: **no `/dev/snd`, no real
  microphone, no speakers, no KDE/Plasma session, no display manager.**
  PipeWire was installed into this container only to run an ephemeral daemon
  with null sinks.
- Therefore: **no real-hardware claim.** Real microphone capture, real
  device unplug/replug, default-device switching during a recording, system
  output from actual applications, Bluetooth headsets, and KDE-specific
  integration (portal permissions, session defaults on a desktop, and the
  tray as rendered by a running plasmashell panel) are **untested**.
- To validate on the target machine, run the commands above on CachyOS/KDE
  with real devices: check both verdicts are `ok` while speaking and playing
  audio, unplug the microphone mid-recording (expect `mic: failed … target
  node disappeared`), and confirm `pw-dump`/`pw-record` come from the
  distribution's PipeWire packages.

## KDE tray (`listn tray`)

The tray is a *client* of the local server — the same loopback API the web
UI uses — rendered as a StatusNotifierItem plus `com.canonical.dbusmenu`
over the session bus (`dbus-next`, installed with `uv sync --extra tray`; no
Qt dependency). `listn tray` starts the engine when none is answering on the
port; `--no-server` refuses to start one and fails with a clear message.

- Controls: recording status line, Start/Stop, Pause/Resume, system-output
  and microphone pickers (radio items; session defaults are adopted as
  labelled suggestions, a vanished remembered node re-adopts only the
  graph's own default, never the first node it saw), Open privacy settings
  (opens the web UI's settings page — listn's own retention/audio settings
  are the privacy surface on Linux; PipeWire has no macOS-style per-app
  permission list), and Quit, which stops an active recording first.
- Honest states: idle / starting / recording (with per-track frame counts)
  / paused / stopping / failed (with the recorder's own per-track cause) /
  unreachable ("listn is not answering"). A refused Start keeps the server's
  reason visible instead of pretending to record. The macOS-only or
  unimplemented controls (meetings, auto-record, launch-at-login) are
  absent, not greyed.
- Verification: the pure state machine and controller are covered in
  `tests/test_tray.py` against the real server app with the stand-in
  `pw-dump`/`pw-record` tools; the D-Bus contract (properties, menu layout,
  `Event(clicked)` dispatch, watcher registration, `LayoutUpdated` on state
  change, failure icons) is covered in `tests/test_tray_sni.py` against a
  real private `dbus-daemon` session and a real uvicorn engine. **Not
  exercised: interaction with a running plasmashell panel** — that needs a
  real KDE session.

## Not in this increment

Pause inside the capture layer (pause stays with the engine session
consuming `on_block`), an application-stream picker in the **KDE tray**
(`listn tray` still offers whole-sink targets only — the CLI, the local
API and the web UI's Live tab do offer application streams), packaging,
GPU/model changes, and any change to macOS behaviour or to the ingest
socket protocol. Web-UI/server wiring now exists (this increment):
`listn record`, `GET /api/pipewire/devices` (with `apps`),
`POST /api/live/start` with `capture: "pipewire"` plus explicit
`system_target`/`mic_target`, the report on `GET /api/live` and in the
`POST /api/live/stop` response. The KDE tray controls now exist (this
increment): `listn tray` with the menu above.
