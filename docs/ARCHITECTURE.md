# Architecture

This describes the repository as inspected at `4dd2c40`. The README and source remain authoritative.

## Current system

```text
macOS Core Audio capture + menu bar (macos/Sources/ListnMenu/)
       | system track 0 / mic track 1, framed PCM + commands
       v
Unix socket ingest (listn/ingest.py, listn/ipc.py)
       v
Python engine (listn/engine.py) -> transcription (listn/transcriber.py)
       |                         -> optional diarization / speaker matching
       v
Local library, recordings and transcripts (listn/library.py, listn/config.py)
       ^
CLI (listn/cli.py) and localhost web UI/API (listn/server.py, listn/web/)
```

The menu-bar capture application is macOS-specific and uses Core Audio. Its audio tracks distinguish microphone (`Me`) from system audio (`Others`); individual speakers on the latter require optional diarization. The Python package also offers audio-file transcription and a legacy input-device recording path. See the README for supported behavior and macOS requirements.

The web server is restricted to loopback, rejects non-loopback Host headers and checks origin for state-changing requests. The ingest socket is private to the user (0600 socket, 0700 data directory; an alternate short socket path is possible). These are security boundaries, not a substitute for a full multi-user authentication system. Data defaults to `~/Library/Application Support/listn` on macOS and `$XDG_DATA_HOME/listn` or `~/.local/share/listn` elsewhere; `LISTN_DATA_DIR` overrides it.

## Linux / KDE direction

The Python file and web workflow has an initial Linux path, tested with synthetic speech; CachyOS/KDE hardware remains unverified. Start with CachyOS/Arch and KDE Plasma. Evaluate PipeWire capture for microphone and system output on distinct tracks and a KDE tray controller. A bounded PipeWire capture prototype (`listn/pipewire.py`, tests in `tests/test_pipewire.py`) records the two tracks to separate files and exposes the engine's `push_audio` seam; a minimal KDE tray client (`listn/tray.py`, `listn/tray_sni.py`, StatusNotifierItem over D-Bus, no Qt) drives the recorder through the loopback API. Both are unverified on real hardware and inside a real plasmashell panel, and are documented in [LINUX_CAPTURE](LINUX_CAPTURE.md). Reuse the existing engine, local library and web UI where practical; evaluate whether the existing IPC contract is appropriate rather than presuming it is complete for Linux. A dashboard can extend the existing web UI for recording state, devices, transcripts and diagnostics; its design and scope are not yet approved.

Linux system/per-app capture, GPU acceleration and live transcription performance are **unverified**, and the tray has not been exercised inside a real KDE panel. `faster-whisper` is a core dependency; MLX extras target Apple Silicon. The local generative notes backend currently uses MLX; without it the existing extractive fallback applies. Avoid claiming Linux generative parity until a real local backend is built and tested.
