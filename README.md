# listn

**For contributors and agents:** [Agent guidance](AGENTS.md) · [Architecture](docs/ARCHITECTURE.md) · [Operations](docs/OPERATIONS.md) · [Decisions and Linux/KDE roadmap](docs/DECISIONS.md). Linux has a bounded Python CLI and local web workflow, including a PipeWire two-track recorder (`listn record` and the Live tab of the web UI), a read-only session dashboard (`/dashboard`), and a minimal KDE Plasma tray (`listn tray`: start/stop/pause, target pickers, privacy settings, quit); capture and the tray are verified against a headless PipeWire daemon with synthetic audio and a real D-Bus session, not yet on real hardware or a real plasmashell panel.

Private, local-only transcripts of what you hear on your Mac, and of audio files, using Whisper on the machine itself.

The everyday controls are a **menu bar app**. It captures system audio with a Core Audio process tap (macOS 14.2 or later) and, at the same time, your microphone. Those stay on separate tracks, so your own speech is labelled **Me** and everyone else **Others**. You can record every app or just one (Zoom, Teams, a browser). BlackHole is not required.

The menu bar app does not transcribe. It feeds audio to the Python engine, which still does transcription, optional diarization, the library, exports, the CLI, and the web UI.

**Local only.** The web UI binds to `127.0.0.1` and refuses other host names. The menu bar talks to the engine over a Unix socket in your data folder (`capture.sock`, mode 0600), not over the network. There are no cloud APIs and no telemetry. The only network access is the one-time model download.

---

## Contents

- [Quick start](#quick-start)
- [Managing listn](#managing-listn): install, update, start and stop, status, logs, permissions, signing, uninstall
- [Install (macOS)](#install-macos) (the manual steps the script runs)
- [Menu bar app](#menu-bar-app)
- [Web UI and CLI](#web-ui-and-cli)
- [Models and backends](#models-and-backends)
- [Notes](#notes) and [Ask your meetings](#ask-your-meetings)
- [How capture is split](#how-capture-is-split)
- [Speaker labels](#speaker-labels) and [Known voices](#known-voices)
- [Privacy, pause, and a self-contained app](#privacy-pause-and-a-self-contained-app)
- [Where your data lives](#where-your-data-lives) and [Backup](#backup)
- [Development](#development)
- [On a Mac, before you trust it](#on-a-mac-before-you-trust-it)
- [Troubleshooting](#troubleshooting)

## Linux (CachyOS / KDE): bounded Python workflow

The existing Python engine and local web library can run on Linux. This is an
initial file-transcription workflow, not macOS feature parity. It has been
exercised on Linux with synthetic speech, not on CachyOS or physical audio
hardware. Install Python 3.10–3.13, `uv`, and the PortAudio system library
(on Arch/CachyOS, package `portaudio`), then:

```sh
uv sync --group dev
uv run pytest
uv run listn transcribe /path/to/audio.wav --model tiny --backend faster-whisper
uv run listn ui --no-browser
# Open http://127.0.0.1:8765 on this machine
```

The first use of a Whisper model downloads weights from Hugging Face; after it
is cached, inference is local. `tiny` is a small starting point, not necessarily
accurate; choose a larger model as needed. The Files tab supports uploads and
the Library tab browsing/export. The server is loopback-only. Data lives in
`$XDG_DATA_HOME/listn` or `~/.local/share/listn` (override `LISTN_DATA_DIR`).

Separate microphone and system tracks are recorded through PipeWire:

```sh
uv run listn record --list                      # discovered nodes (sinks + microphones)
uv run listn record --system-target <node> --mic-target <node> --duration 10
```

The web UI's Live tab offers the same two-track capture on Linux: it shows the
discovered PipeWire nodes, you pick the system output and the microphone
explicitly, and Start is refused with the reason when discovery finds nothing
usable — listn never guesses a capture target. Both paths use the same
recorder and feed the engine's ingest pipeline, so pause, the independent
tracks and transcription behave as they do on macOS. See
[Linux capture](docs/LINUX_CAPTURE.md) for status and limits: prototype,
verified against a headless PipeWire daemon with synthetic audio and covered
by automated tests, not yet validated on real hardware or a KDE session.

A read-only **dashboard** at `http://127.0.0.1:<port>/dashboard` shows the
state the engine already tracks: the active recording (including per-track
PipeWire state and frames while a two-track capture runs), the transcription
queue, and recent recordings with their status (recording / processing /
failed / done), track separation and transcription result. It renders
`GET /api/dashboard`, a projection of engine and library state with no mock
data; with nothing running it says so. Same loopback-only server and
Host/origin checks as the rest of the UI.

KDE Plasma gets a minimal system tray (StatusNotifierItem over D-Bus, no Qt
dependency):

```sh
uv sync --extra tray               # installs dbus-next
uv run listn tray                  # starts the engine if none is running on the port
uv run listn tray --no-server      # only attach to an already-running engine
```

The tray menu is exactly: recording status, Start/Stop, Pause/Resume,
Bookmark this moment, the two
target pickers (radio picks; the system picker holds whole sinks and
individual application streams in one group, so exactly one is checked;
session defaults labelled as suggestions), a
Show captions toggle (off by default; opt-in local caption window), a
global-shortcut status line (start/stop and bookmark are configurable
`Ctrl+Alt+R` / `Ctrl+Alt+B` registered through the desktop's
GlobalShortcuts portal — whatever it grants or refuses is shown there), Open
privacy settings (listn's own settings page — PipeWire has no macOS-style per-app permission list,
and the tray does not pretend otherwise), and Quit
(which stops an active recording first). Failure is shown as failure with the
recorder's own cause (dead track, refused start, engine not answering) —
never a stale "recording". The D-Bus contract is covered by tests against a
real session bus and the Tk caption window under Xvfb; interaction with a
running plasmashell panel is not yet exercised. See
[Linux desktop: captions and global shortcuts](docs/LINUX_DESKTOP.md) for
the feature contract, privacy rules and the target-Plasma manual test.

`listn live` and the web UI's single-input mode record **one** PortAudio
input. A PipeWire/PulseAudio monitor may be selectable if exposed
by PortAudio; routing, permissions, device changes and audio quality need
verification on the target desktop. Do not interpret its labels as reliable
speaker identity. Per-app capture and meeting detection are not
implemented (the tray hides those controls rather than faking them). Launch
at login works on Linux through the web Settings toggle described below and
natively on macOS; the tray menu itself still offers no launch-at-login
control. Do not use the macOS
management script on Linux. GPU acceleration is unverified; MLX-based
notes are unavailable and the existing extractive fallback is used. See
[operations](docs/OPERATIONS.md) for validation steps.

### Speaker labels and known voices (Linux)

The diarization and known-voice code is shared Python, so Linux uses the same
rules and the same tests as macOS:

- **CPU, no GPU assumed.** The speaker model is placed on MPS only when Apple
  Silicon is present; otherwise it runs on the CPU, and a failure on another
  device retries on the CPU.
- **Offline after the cache.** `uv sync --extra diarize` installs the
  dependency. The pyannote pipeline is a gated model, so nothing is downloaded
  until you add a Hugging Face token and label a session; Settings shows
  *ready* and *will download once* as separate facts, and every later load reads
  the local cache with the hub offline. No token, no cache, or a refused model
  each produce their own message — never a silent no-op.
- **The transcript never depends on it.** Without the extra, the token, or the
  model, a recording still finishes with its transcript; the session is marked
  *skipped* with the reason and can be labelled later.
- **Enrolment reads the Others track only.** Naming or correcting a speaker
  slices the speaker-isolated system track. A session that only has the
  microphone track is refused with that reason rather than learned from, and
  **Me** is never a voice to learn. A recording imported as a single track has
  no isolated track, so its own audio is what a correction slices.
- **Ambiguous stays generic.** A voice print is used only when it clears the
  threshold *and* leads the next profile; otherwise the line keeps its
  `Speaker N` label.
- **The profiles are yours.** They live in `$XDG_DATA_HOME/listn/speakers`
  (or `~/.local/share/listn/speakers`). Settings lists them with *Forget*
  (profile and clips removed) and *Export voice prints (JSON)* — the numbers
  only, no audio and no local paths.

This is qualified by the automated suite with synthetic audio and stub models.
Actual CPU inference with a cached pyannote model has **not** been run on this
machine: the model is gated and has not been downloaded here.

### Desktop integration (Linux): menu entry, login, and the Arch package

Everything here is user scope: files land under `$XDG_DATA_HOME` and
`$XDG_CONFIG_HOME` (the `~/.local/share` and `~/.config` defaults), nothing
needs root, and nothing is written outside your home directory. It works like
the macOS management workflow — one script, named commands, `--dry-run` for
all of them — and it refuses to run on macOS (`scripts/listn-manage.sh` is
the macOS installer and is not used on Linux).

| Task | Command |
|---|---|
| Install the app menu entry and icon (KDE app menu: "listn") | `scripts/listn-desktop.sh install` |
| Also start the tray at login | `scripts/listn-desktop.sh install --autostart` |
| See what is installed and what each entry launches | `scripts/listn-desktop.sh status` |
| Remove the menu entry, icon and login entry | `scripts/listn-desktop.sh uninstall` |

The menu entry runs `listn ui` (the local web UI on this machine). The
autostart entry runs `listn tray`, which attaches to an engine that is already
running or starts one on the port if nothing is answering — so a login start
never doubles up on a running server. The login entry is an install-time
choice made by `install --autostart`, never a tray control: the tray has no
launch-at-login control, that macOS control stays hidden. `uninstall` removes
the login entry too, whether or not you installed it. Re-running `install`
refreshes the entries in place — including a login entry you already opted
into — so an upgrade never leaves a stale command behind, and it never
creates a login entry without `--autostart`.

The web UI's **Settings → Launch at login** toggle does the same job without
the installer. On Linux it writes `$XDG_CONFIG_HOME/autostart/listn.desktop`
(`~/.config/autostart/listn.desktop` when `XDG_CONFIG_HOME` is unset) with
the launcher that is running plus `tray`, and switching it off removes that
file again. It only ever removes a file carrying listn's `# written by listn`
marker — an entry you wrote yourself at that path is never deleted, and
turning the toggle on over one asks you to move it first. The toggle reports
what is on disk, not the saved setting, so deleting the file by hand turns it
off. Nothing outside your home directory is written.

The script launches the `listn` command it finds: `$LISTN_CMD` if you set it,
this checkout's `.venv/bin/listn`, then `listn` on `PATH`. Set `LISTN_CMD`
when your command is something else, for example
`LISTN_CMD="uv run --directory $PWD listn"`. Installed entries carry the
resolved absolute path, so they work from the app menu regardless of that
session's `PATH`.

**Arch Linux / CachyOS package.** `packaging/arch/PKGBUILD` builds from the
repository's release tag (`v<version>`) and installs read-only code only:

```sh
cd packaging/arch
makepkg -si
```

The newest release tag must match `pkgver` (currently `v0.5.0`): when a new
tag is cut, bump `pkgver` and refresh the measured `sha256sums` with
`updpkgsums`. The test suite stays red until the two match, and `makepkg`
fails loudly on a missing or changed source either way.

Upgrade = rebuild (after the `pkgver`/`pkgrel` bump) and
`sudo pacman -U listn-*.pkg.tar.zst`; remove with `sudo pacman -R listn`.
Removal takes only the package's own files: your environment
(`~/.local/share/listn/env`), data and transcripts (`~/.local/share/listn`),
any model cache, and a login entry you copied in are left in place — the
package's `.install` message repeats this at install, upgrade and removal.

pacman installs the project tree in `/usr/share/listn`, the launcher in
`/usr/bin/listn`, and the menu entry with its icon. The launcher lets `uv`
build **your** environment on first run from the shipped `uv.lock`, into
`~/.local/share/listn/env`: that run downloads the locked dependencies once
(and a Python 3.10–3.13 interpreter only if the system one is outside
listn's supported range — the same thing `uv sync` does); afterwards it runs
locally, and no state outside `$HOME` is ever written. Declared depends:
`python`, `uv`, `portaudio`, `pipewire`, `wireplumber`; the optional `tray`
extra (dbus-next) is pulled into your environment when you run `listn tray`.
Starting the tray at login stays an install-time choice — copy the shipped
template into your own autostart directory (the web **Settings → Launch at
login** toggle writes its own `listn.desktop` there instead; the two entries
are independent, so keep one):

```sh
install -Dm644 /usr/share/listn/packaging/listn-tray.desktop \
        "$HOME/.config/autostart/listn-tray.desktop"
```

Remove that file to stop; the package's `.install` message repeats both
commands. Desktop-entry correctness, the installer's XDG writes and these
documented commands are covered by `uv run pytest tests/test_desktop_packaging.py`,
which also runs the isolated-HOME shell lifecycle test
(`tests/shell/test_desktop_lifecycle.sh`: install, explicit login opt-in,
re-install, uninstall, user data left alone) and checks the PKGBUILD's
`package()` output against the pinned tag source. The clean Arch
build/install/upgrade/uninstall lifecycle runs through
`scripts/arch_pkg_check.sh` — natively where `makepkg` exists, in an
archlinux container via docker/podman, otherwise an honest SKIP — and CI
runs it in an `archlinux` container
(`.github/workflows/arch-package.yml`). These are shell and container
checks, not a Plasma login test: app-menu visibility, the tray in a real
panel and autostart firing at login still need a manual pass on a desktop.

## Quick start

On an Apple Silicon Mac with macOS 14.2 or later:

```bash
xcode-select --install          # Command Line Tools (Swift, codesign), if you don't have them
brew install uv                 # https://brew.sh first, if you don't have Homebrew
git clone git@github.com:margosmark/listn.git
cd listn
scripts/listn-manage.sh first-run
```

`first-run` checks the prerequisites, installs the Python environment, runs the tests, builds `Listn.app`, copies it to `/Applications` and opens it. Click the waveform icon in the menu bar, start a recording, and allow **Microphone** and **System Audio Recording** when macOS asks.

For speaker labels, also do the one-time [Hugging Face setup](#speaker-labels): accept the model terms, make a read token, and paste it in Settings.

## Managing listn

Everything goes through `scripts/listn-manage.sh` (or the same names with `make`, for example `make install`). Run it with no arguments for the full list. Every command can be run again safely, and `--dry-run` prints what it would do without changing anything.

| Task | Command |
|---|---|
| Check prerequisites | `scripts/listn-manage.sh doctor` |
| Install or refresh the Python environment | `scripts/listn-manage.sh setup` (add `--extras mlx,llm` to leave out speaker labels, or `--extras none` for the minimum) |
| Run the tests | `scripts/listn-manage.sh test` |
| Build `macos/Listn.app` only | `scripts/listn-manage.sh build` |
| Build and install into `/Applications` | `scripts/listn-manage.sh install` |
| Pull the latest code and reinstall | `scripts/listn-manage.sh update` |
| Start, quit, restart the app | `scripts/listn-manage.sh start`, `stop`, `restart` |
| See what's installed and running | `scripts/listn-manage.sh status` |
| Follow a log | `scripts/listn-manage.sh logs` (engine), `logs capture`, `logs selftest` |
| Make macOS ask for permissions again | `scripts/listn-manage.sh reset-permissions` |
| Keep permissions across rebuilds | `scripts/listn-manage.sh signing-cert` (once) |
| Remove the app | `scripts/listn-manage.sh uninstall` |

### Install and update

`install` builds with `macos/scripts/build-app.sh`, verifies the signature, quits the running app, replaces `/Applications/Listn.app` with `ditto`, and opens the new one. `install --no-build` installs the app you already built without rebuilding it.

`update` refuses to run with uncommitted changes, then does `git pull --ff-only`, `setup`, `test`, and `install`. Pass `--skip-tests` to save a minute.

The installed app is self-contained: it carries its own Python and packages, so it keeps working if you move or delete the checkout. Models stay in `~/.cache/huggingface` and are not part of the app.

### Status

`status` prints the checkout version and commit, the installed and built app versions, who signed the app and whether the signature is valid, whether the menu bar app and the engine (`http://127.0.0.1:8765/api/status`) are running, the speaker-label check (`listn speakers --check`: pyannote installed, token present, model downloaded), the size of the data folder and model cache, and how many sessions you have.

### Permissions and signing

macOS ties the Microphone and System Audio Recording grants to the app's signature. By default the app is ad-hoc signed, and every rebuild has a new signature, so after an install the old grants no longer match and recordings can come back silent. Because of that, `install` resets both grants whenever the build is ad-hoc signed, and macOS asks again on your next recording. Allow both.

To stop that, create a signing certificate once:

```bash
scripts/listn-manage.sh signing-cert
scripts/listn-manage.sh install --reset-permissions
```

`signing-cert` makes a self-signed code-signing certificate called `listn local` in your login keychain (macOS may ask for your password to trust it). It never leaves the Mac. From then on `build` and `install` sign with it automatically, the grants survive rebuilds, and `install` stops resetting them. The `--reset-permissions` on the first install is needed once, because the signature changed from ad-hoc. To sign with something else, set `LISTN_SIGN_IDENTITY`. To go back to ad-hoc, delete the certificate in Keychain Access.

If you ever need macOS to ask again (you clicked Don't Allow, or recordings are silent), run `scripts/listn-manage.sh reset-permissions` and start a recording. You can also check by hand under System Settings, Privacy & Security, in **Microphone** and in **Screen & System Audio Recording**.

### Uninstall

```bash
scripts/listn-manage.sh uninstall                    # quits and removes the app, keeps your data
scripts/listn-manage.sh uninstall --purge-data       # also deletes recordings, transcripts, voice prints (asks first)
scripts/listn-manage.sh uninstall --purge-models     # also deletes ~/.cache/huggingface (asks first)
```

`uninstall` also clears the permission grants. Your recordings are only deleted with `--purge-data`, and only after you type `y` (or pass `--yes`). The Hugging Face cache can hold models other tools use, so `--purge-models` deletes the whole folder only after asking. The checkout itself is never touched.

### Where to change things

| What | Where |
|---|---|
| Install location | `LISTN_INSTALL_DIR` (default `/Applications`) |
| Data folder | `LISTN_DATA_DIR` (default `~/Library/Application Support/listn`) |
| Signing identity | `LISTN_SIGN_IDENTITY` (default `listn local` if it exists, else ad-hoc) |
| Build without speaker labels | `LISTN_WITHOUT_SPEAKERS=1 scripts/listn-manage.sh install` |
| App points at the checkout's `.venv` instead of bundling Python | `LISTN_STANDALONE=0 scripts/listn-manage.sh install` |
| Hugging Face token, models, thresholds, retention | Settings in the web UI (**Open library**, then Settings) |

---

## Install (macOS)

`scripts/listn-manage.sh first-run` does all of this. These are the manual steps.

You need Python 3.10–3.13, Swift 6 (the Command Line Tools are enough; a full Xcode install is not), and [Homebrew](https://brew.sh). This was written for Apple Silicon. An M5 Pro with 48 GB is plenty for the default live model.

```bash
brew install uv
cd listn
uv sync --extra mlx --extra llm --extra diarize   # GPU Whisper, notes model, speaker labels
```

`--extra diarize` adds pyannote.audio 4 for telling the people on the Others track apart (see [Speaker labels](#speaker-labels)). Leave it out and everything else works; Others just stays one speaker.

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
uv run listn diarize <session-id>   # split Others into Speaker 1, 2, … (again)
uv run listn speakers               # known voices; --reenroll, --check
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

When a recording or file transcription finishes, listn writes notes into `sessions/<id>/summary.json`: a summary, key points, decisions, action items (owner, task, and a due date when one was said), and open questions. Each item carries the time it was said, and the library shows them under *In the recording* with a link that plays the audio from there. Owners are **Me** (you), a renamed speaker, or someone named in the meeting. Turn this off with *Summarize when a recording ends*, or run it again from the library. The notes are editable; notes you edited are not replaced by the automatic run after a recording or speaker labelling (use *Regenerate*, or `listn summarize --force`).

How the notes are made: the transcript is folded into numbered speaker turns, and the model is asked for one JSON object whose items cite the line they came from. A reply that is not valid JSON is retried once. Meetings longer than about 25 minutes of speech are split into overlapping parts, each part is summarized, and a final pass writes the overall summary and merges duplicate decisions and actions. Items whose cited line does not support them, and action items with no commitment near their line, are dropped.

The default model is `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` (about 17 GB), via `mlx-lm` (`uv sync --extra mlx --extra llm`). It is a mixture-of-experts model with about 3B active parameters, so it writes notes about as fast as a 7B model (roughly 10–25 s for a 5–40 minute meeting and about 80 s for an hour on an M5 Pro) but is much better at attributing commitments and avoiding invented items. It is loaded only while notes are being written and dropped after five idle minutes. On a Mac with less memory set `mlx-community/Qwen2.5-7B-Instruct-4bit` (4.3 GB) in Settings. The weights download on first use and then load offline. Without MLX, listn still fills the same file from the transcript itself (decisions and commitments matched from the wording) and labels that pass `extractive`.

All model calls run on one dedicated thread. MLX streams are per thread, and running the model from the after-recording thread used to fail with `There is no Stream(gpu, 0) in current thread`, which silently left most sessions with extractive notes.

```bash
uv run listn summarize <session-id>
```

## Ask your meetings

The library search box matches keywords and, separately, a local embedding of each transcript line. The embedding is a hashed mix of words and character trigrams stored in `search-index.json`. It does not download a model. Lines are re-embedded only when that session's transcript changes.

**Ask your meetings** answers from those lines with the same local notes model. Each answer cites the session and the timestamp; choosing a citation opens the recording at that line. Without MLX the answer quotes the closest lines and the citations still seek the audio.

A settings file left over from an older run keeps whatever model you saved. In the web UI, set the model to `large-v3-turbo` and the backend to Auto (or MLX).

Backends:

- **faster-whisper** (always installed): CTranslate2 int8 on CPU, with voice-activity detection. On a machine where CTranslate2 sees an NVIDIA GPU but the CUDA user-space libraries are missing (a default CachyOS install has none), the first transcription reports **CUDA libraries unavailable — transcribing on CPU** and finishes on CPU instead of failing; `CUDA_VISIBLE_DEVICES=""` is no longer needed for that. Only load/availability failures (missing library, missing or unusable driver, no CUDA-capable device) take this path — a fault on a working GPU, such as an out-of-memory or illegal-memory-access error, is reported as itself rather than relabelled.
- **mlx** (`--extra mlx`): Apple Silicon GPU. This is the backend the menu bar setup is meant to use.
- **openai-whisper** (`--extra openai`): the original implementation, for comparison.

Live audio is cut on a pause (about 0.45 seconds of quiet) instead of a fixed clock, and never runs longer than the chunk length in Settings. On a 10-second test with two utterances, one of them crossing the old 6-second mark, fixed chunks cut 1 of the 2 utterances and held the first for 4.6 seconds after it ended. Pause chunking cut 0 of the 2 and closed each one 0.5 seconds after it ended. That comparison is `tests/test_vad.py`.

Names and jargon go in Settings, one per line. They are Whisper's initial prompt on every backend. faster-whisper also receives them as hotwords.

Live chunks are a bit less accurate than a whole file. For an important meeting, open it in the library and **Retranscribe**.

## How capture is split

The menu bar sends two PCM streams over the socket. Track 0 is system audio (or the one app you picked) and is labelled **Others**. Track 1 is the microphone and is labelled **Me**. Each track is resampled, saved, and transcribed on its own, so the labels do not depend on a diarization model. That is the reliable split for a call.

After you stop, pyannote splits the system track into Speaker 1, Speaker 2, … and Me is left alone (see [Speaker labels](#speaker-labels)). Rename any of them in the library.

ScreenCaptureKit is not used. It can capture app audio, but the permission is screen recording and the API is built around video. Process taps are the audio API for this, they have been in macOS since 14.2, and 26.7 has them.

The tap is private and **unmuted**, so the meeting keeps playing through your speakers. Nothing is muted and nothing is exclusive.

## Speaker labels

With a Hugging Face token in Settings, listn labels who said what on the **Others** track when a recording finishes. The **Me** track is never touched; it is already you.

- The model is [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) on pyannote.audio 4. If it will not load, listn falls back to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). It runs on the Apple GPU (MPS) when it can and on the CPU otherwise. A one-hour call takes a few minutes; the library shows the progress and you can cancel it.
- Each Others line gets the speaker who talks most during it. A line with nobody detected stays **Others**.
- *Split Others into speakers after stopping* in Settings is on by default once a token is saved or the model is already downloaded. Turn it off there.
- **Label speakers** on a saved session (or `listn diarize <id>`) runs it again, for example on recordings from before 0.5.

Setup, once:

1. Install the extra: `uv sync --extra mlx --extra llm --extra diarize`. The Mac app built by `build-app.sh` includes it.
2. Accept the terms on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). For the 3.1 fallback, also accept [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
3. Create a read token at <https://huggingface.co/settings/tokens> and paste it in Settings (or `export HF_TOKEN=...`).

The token is only used for that first download (about 33 MB into `~/.cache/huggingface`). After that the model loads from the cache with `HF_HUB_OFFLINE` set, even if you delete the token. The token is never written to logs, notes or error messages. pyannote's usage metrics (`PYANNOTE_METRICS_ENABLED`) are switched off before it is imported. Audio and transcripts never leave the Mac.

Without the extra, the token, or the download, recordings still finish with Me / Others. Settings and the session page say what is missing, and `listn speakers --check` prints the same thing.

## Known voices

Named speakers are recognised in later meetings. listn keeps a voice print per person, made with the WeSpeaker ResNet34 speaker-embedding model that community-1 uses (the 256-number embedding pyannote compares voices with). Profiles live in `speakers/<name>.json`, and the clips they were made from are in `speakers/clips/<name>/`. Both stay on the Mac.

- **Naming a speaker enrols them.** In a session, click a speaker chip (Speaker 1, …) and type the name. That renames the speaker, then up to 3 of their longest lines are added to the profile. The next meeting labels them automatically.
- Typing a name in **This is…** on one line also adds that clip.
- In Settings, **Known voices** lists the profiles, lets you forget one, and can re-learn all of them.

**Threshold: 0.55 cosine similarity, plus a 0.08 lead over the runner-up.** Calibrated on 40 LibriSpeech speakers with the real model. Two different people score a median of 0.08 (99th percentile about 0.4, worst 0.58); the same person scores a median of 0.82 on 15-second clean samples, and at 0.55 only 0.06% of different-person pairs pass. Over a call-quality line every score drops and strangers drift closer to each other, so a match must also beat the second-best profile by 0.08. With 3 clips per person taken from that same kind of line, this named all 20 known speakers correctly and gave a name to none of 20 strangers (15-second samples; 16 of 20 on 4-second samples, still with no stranger named). A clean recording enrolled in Settings and matched over a call line does worse (16 of 20, one stranger named), so prefer naming people from a meeting. The old spectrum fingerprint at its 0.72 threshold accepted 87% of different-speaker pairs. Raise the slider (0.3–0.9) if strangers get named, lower it if known people are missed. Each person is claimed by at most one speaker per meeting.

When pyannote is not installed, listn falls back to the old 64-band spectrum fingerprint (threshold 0.72, `spectrum_threshold` in settings) for live matching, so 0.4 profiles keep working.

Upgrading from 0.4: profiles are re-learned with the new model from their saved clips, or from lines already named after that person in the library. A profile with neither is marked *needs re-learning*; name that person once in a session to fix it. The vectors from each method are stored separately and never compared with each other.

During a live recording, Others chunks are matched as they arrive. Turn that off with *Match known voices on the Others track*.

The menu bar can show a floating captions window (**Live captions**). It stays above other windows, including full-screen spaces, and does not take keyboard focus. It is off until you turn it on.

## Privacy, pause, and a self-contained app

**Retention.** Settings can delete audio after N days (`0` keeps it). The transcript, notes, and bookmarks stay. A session that is still recording is left alone. The library says when the audio is gone.

**Pause.** Pause stops transcribing and writes silence, then Resume leaves a *Paused* line in the transcript so the gap is visible and the later lines still line up with the audio. The menu bar has the same control. Sleep pauses a recording; wake reconnects and resumes it.

**Battery.** Under 40% on battery, a new recording uses the `small` Whisper model and skips the notes model. Plugging in, or turning the saver off, keeps the model you chose. The saved settings are not rewritten.

**Disk.** Recording will not start when the data volume has less than 2 GB free (change `min_free_gb` in settings). If space runs out during a long meeting, listn stops and keeps what it already wrote.

**Icon and packaging.** `macos/scripts/build-app.sh` builds `Listn.app` with the Command Line Tools and by default copies a private Python environment into `Contents/Resources/python`. The base interpreter is found by resolving `.venv/bin/python` through its symlinks (uv's `cpython-3.13-…` folder is a symlink to the patch release). Its `bin` and `lib` go to `python/prefix`, the venv's packages go into `prefix/lib/python3.X/site-packages`, and `listn` is copied in so nothing points back at the checkout. Bytecode is compiled at build time, and the launcher sets `PYTHONDONTWRITEBYTECODE` so nothing is written into the signed bundle at run time. Every Mach-O file is signed, then the app, then the whole bundle is verified. `LISTN_STANDALONE=0` points at `.venv/bin/listn` instead. The diarize extra is required for the standalone app: the script stops if `.venv` lacks pyannote.audio, and after bundling runs `listn speakers --check` from the bundled Python (an import check only, no model and no network). `LISTN_WITHOUT_SPEAKERS=1` builds without it. The script prints `du -sh` and writes `Contents/Resources/SIZE.txt` before signing.

What that size is: the Swift app plus the interpreter and the installed wheels. Without the MLX extra that is often a few hundred MB. With `uv sync --extra mlx --extra llm` it is commonly 1–3 GB, because the MLX wheels are large. The diarize extra adds about 65 MB of wheels, roughly 230 MB installed (pandas, scikit-learn, matplotlib, torchaudio, lightning and pyannote; torch itself is already there for mlx-whisper). Whisper `large-v3-turbo` (about 1.6 GB) and the notes model (Qwen3-30B-A3B-Instruct 4-bit, about 17 GB, or Qwen2.5-7B-Instruct 4-bit, about 4.3 GB) stay in `~/.cache/huggingface` and are not inside the app. This Linux checkout cannot produce the signed Mac bundle, so confirm the number on the Mac with the script.

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
speakers/<name>.json            # voice prints, one per method
speakers/clips/<name>/*.wav     # the clips they were made from
```

Legacy and file recordings are a single `audio.wav` (or the imported file). Recordings are 16 kHz 16-bit WAV. Turn off *Keep the audio recording* in the web UI if you only want text. Models are cached in `~/.cache/huggingface`.

The socket path has to fit in a Unix socket address (104 bytes on macOS, including the terminating NUL). The default path does. If `LISTN_DATA_DIR` is so deep that `capture.sock` would not fit, the engine listens on `$TMPDIR/listn-<uid>/capture.sock` when that is short enough, otherwise `/tmp/listn-<uid>/capture.sock`. That directory is mode 0700 and the socket is mode 0600. `server.json` stays in the data folder and records the real socket path.

## Backup

Everything personal is in the data folder: recordings, transcripts, notes, settings (including your Hugging Face token) and voice prints. Quit listn first, then copy it:

```bash
scripts/listn-manage.sh stop
ditto "$HOME/Library/Application Support/listn" ~/listn-backup-$(date +%Y%m%d)
```

To restore, quit listn, put the folder back at the same path, and start it again. Time Machine also covers this folder. Models do not need backing up; they download again. Keep backups private: they hold your meetings.

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
  diarize.py      pyannote 4 speaker labels, community-1 then 3.1
  library.py      sessions, search, export
  speakers.py     voice prints (WeSpeaker, spectrum fallback)
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
- **Not sure what state things are in.** Run `scripts/listn-manage.sh status`. It shows the installed version, whether the app and engine are running, the signature, and the speaker-label check.
- **Recordings went silent right after an update.** The rebuild changed the ad-hoc signature. Run `scripts/listn-manage.sh reset-permissions`, record again and allow both prompts, or set up `signing-cert` so it stops happening.
- **No speaker labels.** Run `scripts/listn-manage.sh status` and read the speaker-label lines. Usually the token is missing in Settings, or the model terms on Hugging Face were not accepted for the account the token belongs to.
- **FFmpeg / `libavutil` errors from torchcodec during speaker labelling.** Harmless: the labels still come out, because listn hands pyannote the audio itself rather than having torchcodec decode the file.
- **`update` says you have uncommitted changes.** Commit or `git stash` them, then run it again.
- **Port 8765 is taken.** Quit the other listn (`engine.log` will say so). The menu bar will not attach to a pre-0.3 server that has no capture socket.

MIT License.
