# listn

Private, local-only transcripts of what you hear on your Mac, and of audio files, using Whisper on the machine itself.

The everyday controls are a **menu bar app**. It captures system audio with a Core Audio process tap (macOS 14.2 or later) and, at the same time, your microphone. Those stay on separate tracks, so your own speech is labelled **Me** and everyone else **Others**. You can record every app or just one (Zoom, Teams, a browser). BlackHole is not required.

The menu bar app does not transcribe. It feeds audio to the Python engine, which still does transcription, optional diarization, the library, exports, the CLI, and the web UI.

**Local only.** The web UI binds to `127.0.0.1` and refuses other host names. The menu bar talks to the engine over a Unix socket in your data folder (`capture.sock`, mode 0600), not over the network. There are no cloud APIs and no telemetry. The only network access is the one-time model download.

---

## Install (macOS)

You need Python 3.10–3.13, Swift 6 (the Command Line Tools are enough; a full Xcode install is not), and [Homebrew](https://brew.sh). This was written for Apple Silicon. An M5 Pro with 48 GB is plenty for the default live model.

```bash
brew install uv
cd listn
uv sync --extra mlx          # Apple Silicon GPU backend; this is the one to install
```

`uv sync` without `--extra mlx` still runs, on the CPU, but live transcription of two tracks will lag. See [Models](#models-and-backends).

BlackHole is optional. Install it only if you want the old "pick an input device" path:

```bash
brew install blackhole-2ch   # optional legacy input, then reboot or: sudo killall coreaudiod
```

## Menu bar app

From the repo:

```bash
macos/scripts/build-app.sh
open macos/Listn.app
```

The script runs `swift build -c release` for macOS 14.2, assembles `macos/Listn.app`, copies `Info.plist` (system-audio and microphone usage strings, menu-bar agent, minimum 14.2), and signs it. It builds into a temporary folder and only replaces `macos/Listn.app` once `codesign --verify --deep --strict` passes, so a failed step leaves the previous app untouched.

The first time you record, the menu asks for **Microphone** and **System Audio Recording** before any audio flows. Allow both. The app does not appear in the Dock (`LSUIElement`).

**Rebuilds and permissions.** By default the app is ad-hoc signed (`codesign --sign -`). macOS keys an ad-hoc grant to that exact build, so after a rebuild the old grant no longer matches. The capture keeps running but macOS hands it zeros: a silent recording with no error. listn now catches this, but the fix is to reset the stale grant and allow again:

```bash
tccutil reset Microphone local.listn.menu
tccutil reset AudioCapture local.listn.menu
```

To make grants survive rebuilds, create a self-signed code-signing certificate once (Keychain Access → Certificate Assistant → Create a Certificate, type *Code Signing*, name it `listn local`) and build with `LISTN_SIGN_IDENTITY="listn local" macos/scripts/build-app.sh`.

**Silent input.** If the microphone delivers only exact zeros for about two seconds, or system audio does while some app is playing, the menu bar icon changes to a warning and its title becomes **No audio**. Zeros from system audio while nothing plays are expected and do not warn; the level line says *no app is playing* instead. The menu's first item says capture returned silence and points at `~/Library/Application Support/listn/capture.log`, with **Open Privacy Settings** under it. The log records the default devices, which processes are running audio output (pid, bundle id, `IsRunningOutput`) at the start and every 5 seconds, whether the default output is running somewhere, each tap's configuration and status codes, the tap UID check, formats, the measured stream rate, and the first buffer peaks. A finished recording that never heard any signal is saved as an error, not as a done session with an empty transcript.

The system-audio tap's aggregate device is clocked to the current default output and follows when that output changes. Recording all system audio tries, in order: a stereo global tap that excludes listn, a tap on the default output device, a mixdown of the apps playing, then the global and device taps without the output clock. If one starts but records only zeros for 3 seconds while an app is playing, listn switches to the next and logs it. The first one that hears audio is remembered for later recordings (also set by the self-test). The rate a stream really runs at is measured during its first second, so a Bluetooth headset in call mode (16 kHz) is not recorded as 48 kHz. The microphone is a separate Core Audio callback, not the tap and not AVAudioEngine.

**Bluetooth headsets.** Opening a headset's microphone switches the headset to call mode: mono, 16 kHz, for playback too. When the default input is a Bluetooth headset, the menu says so and offers **Use built-in microphone**. It records the Mac's microphone and leaves the headset in normal playback, without changing the system input setting.

**Self-test.** Quit listn first (`open` will not pass arguments to an app that is already running). This launch is the one whose permission prompts belong to listn:

```bash
open macos/Listn.app --args --selftest 8
```

Then read `~/Library/Application Support/listn/selftest.log` (the same lines are appended to `capture.log`). A direct run prints to the terminal as well, but macOS attributes its microphone prompt to the shell, so use it only to compare:

```bash
macos/Listn.app/Contents/MacOS/Listn --selftest 8
```

Play something (a YouTube video, Music) while it runs. It records the production capture for the given seconds (default 5, maximum 30). Then it tries each tap configuration for 3 seconds with the microphone still open, and the two main ones again after closing the mic. The configurations are a global tap excluding listn, the same tap-only, with TapAutoStart, with an empty exclude list, mono, the include-only setup of 0.4.0–0.4.1 (always zeros, kept as a control), a mixdown of every app playing, and a tap on the output device clocked and tap-only. The log ends with a `probe summary`: one line per configuration with `AUDIO` or `zeros`, the peak, callbacks, frames, rate, and which apps were playing. The first production configuration that heard audio is saved for recordings. The whole run takes about a minute.

Exit status is 0 when both microphone and system audio were heard, 3 for microphone only, 4 for system audio only, 2 for neither, and 1 when capture could not start.

**Engine version.** The app compares the running engine's version (from `server.json`) with its own. An engine left over from an older build is restarted when it is idle; one in the middle of a recording or a file job is left alone until next time.

What the menu does:

- **Start / Stop recording**
- **Source:** all system audio, one app that is currently doing audio, or the legacy input device selected in the web UI
- **Microphone** on or off. On the legacy device this toggle is disabled, because that device already is whatever you routed into it
- A **level** meter
- The latest **transcript lines** (Me vs Others)
- **Open library**, which is the web UI
- **Record Zoom** (or whichever meeting just started using audio), when that offer is on
- **Bookmark this moment** while a recording is running. Bookmarks show up in the transcript and jump the audio.
- **Launch at login**
- **Quit**, which also stops the engine if the app started it

Control-Option-R starts or stops a recording. Control-Option-B drops a bookmark. Neither hotkey needs Accessibility access.

The menu watches Zoom, Teams, Webex, FaceTime, a Slack huddle, and a browser window whose title looks like Meet, Zoom, Teams, or Webex. It asks before recording unless *Start recording automatically* is on, and it asks again when that call ends. The first time, macOS asks to read your calendars. An event that overlaps "now" becomes the session title and its attendees are stored with the recording. The all-day event loses to the shorter meeting. Calendar access stays on this Mac.

The app starts `listn ui --no-browser` in the background as soon as it launches, using the `listn` written into the bundle at build time (your `.venv`). If you move the `.app` but not the checkout, set `LISTN_BIN` to the `listn` script. Logs: `~/Library/Application Support/listn/engine.log`.

You do not need to start the web server yourself. **Open library** waits until the engine is actually listening, then opens `http://127.0.0.1:8765`. The menu item says **Starting…** while that happens. If the engine cannot start, the error is shown in the menu instead of a browser error page.

## Web UI and CLI

```bash
uv run listn                 # http://127.0.0.1:8765
uv run listn ui --no-browser --port 8765
uv run listn devices
uv run listn devices --test
uv run listn live            # legacy input device in the terminal
uv run listn transcribe call.m4a --format srt
uv run listn list "budget"
uv run listn export <session-id> --format md
uv run listn path
```

The Live page's device picker is the **legacy** path: a microphone, or BlackHole / an aggregate device if you set one up. The page says so. Starting a recording there does not use the process tap. Everyday meetings should be started from the menu bar; the transcript still shows up in the web UI while it is open, because both share the same engine.

`--device` takes a name, part of a name, or an index. With nothing selected, the legacy path prefers BlackHole, then an aggregate, then the default input.

## Models and backends

| Model | Download | Notes |
|---|---|---|
| `tiny` / `base` | 75 / 145 MB | Fastest; fine for clear speech |
| `small` | 480 MB | Default when MLX is not available |
| `medium` | 1.5 GB | Noticeably better on accents and jargon |
| `large-v3-turbo` | 1.6 GB | Default live model on Apple Silicon with MLX |
| `large-v3` | 3 GB | Most accurate, slowest |

**Why `large-v3-turbo` on this Mac.** It is a distilled large-v3: similar accuracy, about 1.6 GB, and much faster. With MLX on an M5 Pro (48 GB), one 6-second chunk finishes well inside that 6 seconds, and doing it twice — system audio, then the microphone — still keeps up. `auto` selects the MLX backend when `mlx-whisper` imports on Apple Silicon. If MLX is not installed, the default model stays `small`, because turbo on CPU int8 can fall behind two live tracks.

The first use of a model downloads it from Hugging Face. That request is unauthenticated unless you set `HF_TOKEN`, so the log may say `Warning: You are sending unauthenticated requests to the HF Hub` once. After the files are in `~/.cache/huggingface`, later loads set `HF_HUB_OFFLINE` and do not use the network.

## Notes

When a recording or file transcription finishes, listn writes a summary, the decisions, and action items into `sessions/<id>/summary.json`. Each action item is attributed to **Me**, **Others**, or a renamed speaker. Turn this off with *Summarize when a recording ends*, or run it again from the library. The notes are editable.

The model is `mlx-community/Qwen2.5-7B-Instruct-4bit` (about 4.3 GB), via `mlx-lm` (`uv sync --extra mlx --extra llm`). It is a 7B instruct model in 4-bit: accurate enough to keep commitments attached to the right person, and small enough to sit next to Whisper large-v3-turbo on a 48 GB Mac. Change the repo id in Settings if you want a different MLX model. The weights download on first use and then load offline. Without MLX, listn still fills the same file from the transcript itself (decisions and commitments matched from the wording) and labels that pass `extractive`.

```bash
uv run listn summarize <session-id>
```

## Ask your meetings

The library search box matches keywords and, separately, a local embedding of each transcript line. The embedding is a hashed mix of words and character trigrams stored in `search-index.json`. It does not download a model. Lines are re-embedded only when that session's transcript changes.

**Ask your meetings** answers from those lines with the same local notes model. Each answer cites the session and the timestamp; choosing a citation opens the recording at that line. Without MLX the answer quotes the closest lines and the citations still seek the audio.

A settings file left over from an older run keeps whatever model you saved. In the web UI, set the model to `large-v3-turbo` and the backend to Auto (or MLX).

Backends:

- **faster-whisper** (always installed): CTranslate2 int8 on CPU, with voice-activity detection.
- **mlx** (`--extra mlx`): Apple Silicon GPU. This is the backend the menu bar setup is meant to use.
- **openai-whisper** (`--extra openai`): the original implementation, for comparison.

Live audio is cut on a pause (about 0.45 seconds of quiet) instead of a fixed clock, and never runs longer than the chunk length in Settings. On a 10-second test with two utterances, one of them crossing the old 6-second mark, fixed chunks cut 1 of the 2 utterances and held the first for 4.6 seconds after it ended. Pause chunking cut 0 of the 2 and closed each one 0.5 seconds after it ended. That comparison is `tests/test_vad.py`.

Names and jargon go in Settings, one per line. They are Whisper's initial prompt on every backend. faster-whisper also receives them as hotwords.

Live chunks are a bit less accurate than a whole file. For an important meeting, open it in the library and **Retranscribe**.

## How capture is split

The menu bar sends two PCM streams over the socket. Track 0 is system audio (or the one app you picked) and is labelled **Others**. Track 1 is the microphone and is labelled **Me**. Each track is resampled, saved, and transcribed on its own, so the labels do not depend on a diarization model. That is the reliable split for a call.

If you turn on speaker labels, pyannote runs only on the system track after you stop, and Me is left alone. Rename either name in the library.

ScreenCaptureKit is not used. It can capture app audio, but the permission is screen recording and the API is built around video. Process taps are the audio API for this, they have been in macOS since 14.2, and 26.7 has them.

The tap is private and **unmuted**, so the meeting keeps playing through your speakers. Nothing is muted and nothing is exclusive.

## Known speakers

Voices on the Others track can be matched to people you have already named. The embedding is 64 log-magnitude bands of a short clip, mean-centred and normalised. Nothing is downloaded. Profiles live in `speakers/<name>.json` (up to 24 clips each). A match has to clear the threshold in Settings (default 0.72). **Me** is never enrolled; that track is already you.

Teach a voice in either of two ways:

- In a saved session, type a name in **This is…** on an Others line. That clip is added to the profile and the line is renamed.
- In Settings, save a short WAV of one person talking.

The next meeting compares each Others chunk with those profiles and labels the line when it is confident. Correcting a wrong name adds the new clip, so the profile gets better. Turn matching off with *Match known voices on the Others track*.

The menu bar can show a floating captions window (**Live captions**). It stays above other windows, including full-screen spaces, and does not take keyboard focus. It is off until you turn it on.

## Speaker labels (optional, for more than Me / Others)

1. `uv sync --extra diarize`
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), and create a read token at <https://huggingface.co/settings/tokens>.
3. Paste the token in Settings, or `export HF_TOKEN=...`. It is only needed for the first download.

## Privacy, pause, and a self-contained app

**Retention.** Settings can delete audio after N days (`0` keeps it). The transcript, notes, and bookmarks stay. A session that is still recording is left alone. The library says when the audio is gone.

**Pause.** Pause stops transcribing and writes silence, then Resume leaves a *Paused* line in the transcript so the gap is visible and the later lines still line up with the audio. The menu bar has the same control. Sleep pauses a recording; wake reconnects and resumes it.

**Battery.** Under 40% on battery, a new recording uses the `small` Whisper model and skips the notes model. Plugging in, or turning the saver off, keeps the model you chose. The saved settings are not rewritten.

**Disk.** Recording will not start when the data volume has less than 2 GB free (change `min_free_gb` in settings). If space runs out during a long meeting, listn stops and keeps what it already wrote.

**Icon and packaging.** `macos/scripts/build-app.sh` builds `Listn.app` with the Command Line Tools and by default copies a private Python environment into `Contents/Resources/python`. The base interpreter is found by resolving `.venv/bin/python` through its symlinks (uv's `cpython-3.13-…` folder is a symlink to the patch release). Its `bin` and `lib` go to `python/prefix`, the venv's packages go into `prefix/lib/python3.X/site-packages`, and `listn` is copied in so nothing points back at the checkout. Bytecode is compiled at build time, and the launcher sets `PYTHONDONTWRITEBYTECODE` so nothing is written into the signed bundle at run time. Every Mach-O file is signed, then the app, then the whole bundle is verified. `LISTN_STANDALONE=0` points at `.venv/bin/listn` instead. The script prints `du -sh` and writes `Contents/Resources/SIZE.txt` before signing.

What that size is: the Swift app plus the interpreter and the installed wheels. Without the MLX extra that is often a few hundred MB. With `uv sync --extra mlx --extra llm` it is commonly 1–3 GB, because the MLX wheels are large. Whisper `large-v3-turbo` (about 1.6 GB) and Qwen2.5-7B-Instruct 4-bit (about 4.3 GB) stay in `~/.cache/huggingface` and are not inside the app. This Linux checkout cannot produce the signed Mac bundle, so confirm the number on the Mac with the script.

A dropped input device is reopened. The menu bar also restarts the process tap when the audio device list changes, and reconnects the capture socket if the engine process disappears.

## Where your data lives

`~/Library/Application Support/listn/` (override with `LISTN_DATA_DIR`):

```
settings.json
server.json                 # port, pid, capture socket path
capture.sock                # menu bar ↔ engine, mode 0600
engine.log                  # only if the menu bar app started the engine
sessions/<id>/meta.json
sessions/<id>/transcript.json
sessions/<id>/audio.wav             # mix for playback
sessions/<id>/audio-system.wav      # Others
sessions/<id>/audio-mic.wav         # Me
speakers/<name>.json            # voice profile, no model download
```

Legacy and file recordings are a single `audio.wav` (or the imported file). Recordings are 16 kHz 16-bit WAV. Turn off *Keep the audio recording* in the web UI if you only want text. Models are cached in `~/.cache/huggingface`.

The socket path has to fit in a Unix socket address (104 bytes on macOS, including the terminating NUL). The default path does. If `LISTN_DATA_DIR` is so deep that `capture.sock` would not fit, the engine listens on `$TMPDIR/listn-<uid>/capture.sock` when that is short enough, otherwise `/tmp/listn-<uid>/capture.sock`. That directory is mode 0700 and the socket is mode 0600. `server.json` stays in the data folder and records the real socket path.

## Development

```bash
uv sync
uv run pytest        # no audio hardware: fake devices, fake Whisper, fake capture client
```

The Swift app does not build on Linux. `macos/scripts/build-app.sh` exits there on purpose.

```
listn/
  cli.py          listn entry point
  server.py       localhost web UI; starts the capture socket with the server
  ingest.py       Unix-socket server (one client; a new connection replaces the old)
  ipc.py          frame codec shared with the menu bar app
  engine.py       device capture and ingest capture, file jobs, events
  capture.py      legacy input streams
  devices.py      device discovery; BlackHole guidance is legacy-only
  audio.py        mixing, resampling, chunking, WAV
  transcriber.py  faster-whisper / MLX / openai-whisper
  diarize.py      optional pyannote, whole recording
  library.py      sessions, search, export
  speakers.py     local voice profiles
  retention.py    delete old audio, keep text
  power.py        battery model choice and free-space check
  bundle.py       copy Python into the .app
  iconart.py      app icon PNGs
macos/
  Package.swift
  Info.plist
  Sources/ListnMenu/
  scripts/build-app.sh
```

### Capture socket

Frames are little-endian: `uint32 length | uint8 type | payload`. Types are HELLO, WELCOME, AUDIO, START, STOP, EVENT, PING, PONG, ERROR, BOOKMARK, PAUSE, RESUME. AUDIO is `track | sample rate | channels | float32 PCM`, track 0 system and track 1 microphone. START's `source` is `system`, `app`, or `device` (the legacy input). Details are in `listn/ipc.py`. The Python tests drive this with a fake client, including a client that reconnects mid-recording.

## On a Mac, before you trust it

This repo's tests do not open a process tap. On the Mac, check:

1. `macos/scripts/build-app.sh` finishes and `codesign --verify --deep --strict macos/Listn.app` is clean. `macos/Listn.app/Contents/Resources/python/bin/listn --version` prints the app version.
2. After a rebuild, run the two `tccutil reset` lines from *Menu bar app*. Quit listn, play some audio, and run `open macos/Listn.app --args --selftest 8`. `~/Library/Application Support/listn/selftest.log` should end with a `probe summary` where `global-stereo/clocked` says `AUDIO`, and exit 0. Then record from the menu: the level meter moves. If capture is silence, the menu bar says **No audio** and **Open Privacy Settings** is at the top of the menu.
3. `open macos/Listn.app`. The menu bar icon appears and no Dock icon does. If an older engine was running, `engine.log` shows a fresh start with the new version.
4. Start recording. macOS asks for system-audio recording and the microphone. After allowing both, the menu's level meter moves when system audio plays and when you speak.
5. Playback of other apps does **not** go silent while the tap is running.
6. A Zoom, Teams, or browser tab can be chosen as the only source, and other apps' audio stays out of Others.
7. A short call: your voice is **Me**, the other side is **Others**, in the menu and in the library.
8. Open Library. The web UI loads on `127.0.0.1` only. The live page shows the same lines. Export still works.
9. Start a Zoom or Meet call. The menu offers to record, and with automatic recording on it starts by itself and offers to stop when the call ends. An overlapping calendar event titles the session. Control-Option-B leaves a bookmark you can seek to. Launch at login survives a reboot.
10. Name an Others line in the library (**This is…**). Record again with that person on the far side: the new lines should use that name when the match is confident. **Live captions** stays above a full-screen meeting and updates as lines arrive.
11. Quit and reopen. If you denied a permission, System Settings shows **listn** (not "python" or a blank name) under System Audio Recording and Microphone.
12. Legacy: install BlackHole only if you want it, pick it in the web UI, choose **Legacy input device** in the menu, and confirm the old path still records.
13. With `uv sync --extra mlx` and model `large-v3-turbo`, two live tracks stay caught up (the UI should not sit on "N chunks waiting" during normal speech).
14. Pause during a recording. The transcript shows a Paused gap, the audio is silent there, and Resume continues the same session. Sleep the Mac: it pauses, and after wake the tap is back and the session is still going.
15. Set audio retention to 1 day, or use a session older than the limit. The wav files go away and the transcript and notes stay. The library says the audio was removed.
16. `macos/scripts/build-app.sh` prints the app size. `codesign --verify macos/Listn.app` is clean. Move `Listn.app` somewhere with no checkout and no uv and open it: the menu bar and the library still come up. Confirm the icon is the indigo waveform.
17. On battery under 40%, start a recording. The session model is `small` and it does not run the notes model. Plug in and the next recording uses the model in Settings.
18. Start a recording, then switch the output device (or unplug headphones). The tap restarts and this session keeps going. Pulling the engine down and letting it come back reconnects the menu without a second START.

## Troubleshooting

- **Menu says it cannot find the engine.** Run `uv sync --extra mlx` in the repo and rebuild, or `export LISTN_BIN=/absolute/path/to/.venv/bin/listn`.
- **Every recording is silent (peak 0, empty transcript).** The menu should say **No audio** and name `capture.log`. Read that log: `default output` should be the device you are playing through, `audio processes` should list the playing app with `out=yes`, and `first second system` should show a non-zero peak. A line `trying the next configuration` means a tap heard only zeros while an app played. Run `open macos/Listn.app --args --selftest 8` after quitting listn, with audio playing, and read the `probe summary`. Permissions still matter after an ad-hoc rebuild: run the two `tccutil reset … local.listn.menu` lines from *Menu bar app* and allow both prompts, or sign with `LISTN_SIGN_IDENTITY`.
- **System audio permission denied.** System Settings → Privacy & Security → Screen & System Audio Recording → listn.
- **An app is missing from Source.** It has to have opened an audio device. Start the meeting audio, then open the menu again.
- **The meter stays silent.** Confirm the meeting is actually playing, and that you did not pick a single app that is quiet. Tap auto-start is off so recording begins immediately; silence until someone speaks is normal.
- **Legacy transcript is empty.** That path still needs BlackHole (or an aggregate) routed in Audio MIDI Setup. The menu-bar tap does not.
- **Live lags.** Install the MLX extra, use `large-v3-turbo` rather than `large-v3`, or raise the chunk length in Settings.
- **Port 8765 is taken.** Quit the other listn (`engine.log` will say so). The menu bar will not attach to a pre-0.3 server that has no capture socket.

MIT License.
