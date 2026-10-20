# listn

Private, local-only transcription of your Mac's system audio (Zoom, Meet, Teams, browser, anything you can hear) and of audio files, using Whisper running on your own machine.

- **Live**: capture system audio through BlackHole and watch the transcript stream in as people talk. The recording is saved alongside it.
- **Files**: transcribe WAV, MP3, M4A, FLAC, OGG, or video files, with progress.
- **Library**: search every transcript, play the audio from any line, rename speakers, and export to TXT, Markdown, SRT, VTT, or JSON.
- **Speaker labels** (optional): pyannote diarization over the whole recording.
- **Web UI and CLI**: the same engine runs behind both.

**Local only.** The UI server binds to `127.0.0.1` and refuses requests from other host names or cross-site pages. There are no cloud APIs and no telemetry (Hugging Face and pyannote usage reporting are switched off). The only network access is the one-time model download. After that, listn runs offline.

---

## Install (macOS)

You need Python 3.10–3.13 and [Homebrew](https://brew.sh).

```bash
# 1. BlackHole: a virtual audio device that lets listn hear system audio
brew install blackhole-2ch            # then restart your Mac (or: sudo killall coreaudiod)

# 2. listn
brew install uv                       # fast Python package manager
git clone <this repo> listn && cd listn
uv sync                               # creates .venv with the pinned versions from uv.lock

# Optional extras
uv sync --extra mlx                   # Apple Silicon GPU backend (recommended on M-series Macs)
uv sync --extra diarize               # speaker labels (pulls in PyTorch, ~1 GB)
uv sync --extra mlx --extra diarize   # both
```

Prefer plain pip? `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .`
(`requirements-all.txt` pins every optional extra too.)

### Route audio so you can hear it *and* listn can capture it (one time)

1. Open **Audio MIDI Setup** (Applications → Utilities).
2. Click **+** → **Create Multi-Output Device**, then tick your speakers/headphones **and** BlackHole 2ch. Put your speakers first and enable *Drift Correction* on BlackHole.
3. In **System Settings → Sound → Output**, choose the Multi-Output Device. Or route just your meeting app by picking it as that app's speaker.

To capture **your own microphone as well** (both sides of a call), create an **Aggregate Device** in Audio MIDI Setup with BlackHole 2ch and your mic, and pick it as listn's input. listn opens every channel and mixes only the ones carrying sound.

The first time you record, macOS asks for microphone permission for your terminal app. This is needed even for BlackHole. If you missed the prompt, enable it under System Settings → Privacy & Security → Microphone.

---

## Run

```bash
uv run listn            # opens the UI at http://127.0.0.1:8765
```

The Live tab checks your setup and tells you what's missing: BlackHole not installed, no Multi-Output Device, or system output not routed to BlackHole. **Test input** shows per-channel levels so you can confirm listn hears your meeting before you rely on it.

### CLI

```bash
uv run listn devices                      # list inputs + setup checks
uv run listn devices --test               # 2-second level check on the chosen device
uv run listn live                         # record + live transcript in the terminal (Ctrl+C to stop)
uv run listn live --device "Aggregate" --model medium --language en --speakers
uv run listn transcribe call.m4a --format srt --output call.srt
uv run listn transcribe *.mp3 --output transcripts/
uv run listn list                         # recent sessions
uv run listn list "quarterly budget"      # search transcripts
uv run listn export <session-id> --format md
uv run listn path                         # where your data lives
```

`--device` takes a name, part of a name, or an index. listn stores the device by name, because macOS renumbers devices whenever you plug in headphones. With no device set, it picks BlackHole, then an aggregate device, then your default input. Settings chosen in the UI (model, language, device) also apply to the CLI.

---

## Models and backends

| Model | Download | Notes |
|---|---|---|
| `tiny` / `base` | 75 / 145 MB | Fastest; fine for clear speech |
| `small` (default) | 480 MB | Good balance for live use |
| `medium` | 1.5 GB | Noticeably better on accents and jargon |
| `large-v3-turbo` | 1.6 GB | Near large-v3 accuracy, much faster; great with MLX |
| `large-v3` | 3 GB | Most accurate, slowest |

Backends (Settings → Transcription backend):

- **faster-whisper** (default, always installed): CTranslate2 int8 on CPU. About 4x faster than the original openai-whisper, uses much less memory, doesn't need PyTorch, and includes voice-activity detection that removes most "Thanks for watching!" hallucinations on silence.
- **mlx** (`--extra mlx`): runs on the Apple Silicon GPU. `auto` picks it when installed on an M-series Mac.
- **openai-whisper** (`--extra openai`): the original implementation, kept for comparison.

Live transcription works on short chunks (6 s by default, adjustable), so accuracy is a bit lower than on a whole file. For important meetings, open the session in the Library and click **Retranscribe**, optionally with a bigger model, or **Label speakers**. Both run over the full recording.

## Speaker labels

1. `uv sync --extra diarize`
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), and create a read token at <https://huggingface.co/settings/tokens>.
3. Paste the token in Settings, or `export HF_TOKEN=...`. It's needed only for the first download. The token is saved in `settings.json` with owner-only permissions.

Speakers are identified over the whole recording after you stop, so labels stay consistent. You can rename "Speaker 1" to a real name in the Library, and exports use your names.

## Where your data lives

`~/Library/Application Support/listn/` (override with `LISTN_DATA_DIR`):

```
settings.json
sessions/<id>/meta.json  transcript.json  audio.wav|.mp3|...
```

Nothing is written to the repo or the current directory unless you pass `--output`. Recordings are 16 kHz mono WAV, about 115 MB per hour. Turn off *Keep the audio recording* if you only want text. Models are cached in `~/.cache/huggingface`.

---

## Development

```bash
uv sync
uv run pytest           # hardware-free: fake sounddevice + fake Whisper
```

Layout:

```
listn/
  cli.py          `listn` entry point (ui, devices, live, transcribe, list, export, path)
  server.py       FastAPI app + localhost guard; serves listn/web/ (plain HTML/CSS/JS, no build step)
  engine.py       live sessions (capture → resample → WAV + chunker → Whisper), file job queue, event bus
  capture.py      opening input streams with channel/rate fallbacks, level test
  devices.py      device discovery, name-based selection, BlackHole setup guidance
  audio.py        mixing, anti-aliased resampling, quiet-point chunking, WAV writer, file decoding
  transcriber.py  faster-whisper / MLX / openai-whisper backends, hallucination filter
  diarize.py      pyannote pipeline + speaker assignment
  library.py      session storage, search, export formats
```

## Troubleshooting

- **Transcript is empty / Test input says "No signal"**: system output isn't going through BlackHole. Choose the Multi-Output Device as output, and check the meeting app's own speaker setting.
- **"Could not open 'Aggregate Device'"**: in Audio MIDI Setup, make sure every sub-device uses the same sample rate and one of them is the clock source.
- **Live lags behind**: the UI shows "N chunks waiting". Pick a smaller model, install the MLX backend, or raise the chunk length.
- **Device missing after installing BlackHole**: restart Core Audio (`sudo killall coreaudiod`) and click **Rescan**.

MIT License.
