# Linux capture prototype (PipeWire)

Status: **prototype with automated tests and one real-daemon verification on a
headless machine — not validated on real audio hardware or a KDE session.**
This implements roadmap step 2 in [DECISIONS](DECISIONS.md) as a bounded
increment; it does not ship KDE desktop support. The recorder is now wired
into the app: `listn record` on the CLI and the Live tab of the local web UI
both start two-track captures through the same code paths.

## What it is

`listn/pipewire.py` records the system output and the microphone as two
independent tracks through PipeWire, the audio stack KDE Plasma uses:

- `discover()` reads the graph with `pw-dump`; `select_targets()` picks an
  `Audio/Sink` for the system track (recorded through the sink's monitor
  ports) and an `Audio/Source` for the microphone. Session defaults are
  re-checked against the node's media class before use, explicit `--system`
  / `--mic` selectors are validated, and selection is labelled `default`,
  `fallback` or `explicit` in every report so a guess is never presented as
  a default.
- `TwoTrackRecorder` runs one `pw-record` per track (raw int16 PCM on
  stdout) into its own WAV file, and can hand each block to an `on_block`
  callback. That callback matches the engine's
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
uv run listn ui --no-browser                       # web UI: pick both targets in
                                                   # the Live tab, then Start

uv run pytest tests/test_pipewire.py               # module tests
uv run pytest tests/test_pipewire_wiring.py        # CLI/server wiring tests
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
  440 Hz tone piped into the sink and into the pipe-tunnel source):

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

Exit codes: `0` capture succeeded with microphone signal, `1` capture failed
or the microphone recorded only silence, `3` PipeWire/tools unavailable or
no usable node. Reports always carry per-track `state`, `verdict`
(`ok`/`quiet`/`silent`), `duration`, `error` and `exit_code`; silence is
never reported as success, and a silent system track alone does not fail the
run (nobody may have been playing anything) — the same rule the engine uses.

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
  default, which named a sink on this rig (see below).

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
   separate `.monitor` node exists in native `pw-dump` output).
7. **`pw-dump <id>` can return more than the queried object.** Measured on a
   bare headless 1.4.2 daemon (no WirePlumber): with node ids and object
   serials drifted apart, `pw-dump 40` returned two objects — a node with
   `id=37, object.serial=40` first, then the requested `id=40` node — so an
   object whose `object.serial` merely equals the queried id can lead the
   dump. Reading `payload[0]` compares that stranger's serial and reports a
   healthy node as gone (issue #4), which blocks `start()` outright. The
   probe therefore selects the element whose `id` equals the node's id and
   only then checks serial/name; no element with that id means gone.

## Hardware limitations of this verification

- The machine is a headless Proxmox LXC container: **no `/dev/snd`, no real
  microphone, no speakers, no KDE/Plasma session, no display manager.**
  PipeWire was installed into this container only to run an ephemeral daemon
  with null sinks.
- Therefore: **no real-hardware claim.** Real microphone capture, real
  device unplug/replug, default-device switching during a recording, system
  output from actual applications, Bluetooth headsets, and KDE-specific
  integration (tray, portal permissions, session defaults on a desktop) are
  **untested**.
- To validate on the target machine, run the commands above on CachyOS/KDE
  with real devices: check both verdicts are `ok` while speaking and playing
  audio, unplug the microphone mid-recording (expect `mic: failed … target
  node disappeared`), and confirm `pw-dump`/`pw-record` come from the
  distribution's PipeWire packages.

## Not in this increment

KDE tray controls, per-app capture, pause inside the capture layer (pause
stays with the engine session consuming `on_block`), packaging, GPU/model
changes, and any change to macOS behaviour or to the ingest socket protocol.
Web-UI/server wiring now exists (this increment): `listn record`,
`GET /api/pipewire/devices`, `POST /api/live/start` with
`capture: "pipewire"` plus explicit `system_target`/`mic_target`, the report
on `GET /api/live` and in the `POST /api/live/stop` response.
