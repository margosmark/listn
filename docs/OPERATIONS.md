# Operations and development

## Existing macOS workflow

The supported installation and management commands are in [README](../README.md). `scripts/listn-manage.sh` and the Makefile manage macOS app builds and installation; inspect the command before running it. The system-audio tap requires macOS 14.2 or later. Real audio and permission verification needs a Mac, not just unit tests.

## Python development

The project declares Python `>=3.10,<3.14` and uses `uv.lock`. From the repository root, with `uv` installed:

```sh
uv sync --group dev
uv run pytest
uv run python -m listn --help
```

Optional backends and macOS install instructions are documented in README. The commands above may download packages; model use may download model weights. A successful unit test does not establish that a microphone, system-audio capture, GPU, or real meeting works. Keep test data isolated and do not publish personal audio or tokens.

For local experiments, `LISTN_DATA_DIR` overrides the default storage directory. On macOS the default is `~/Library/Application Support/listn`; elsewhere it is `$XDG_DATA_HOME/listn` or `~/.local/share/listn`. The local web service is loopback-only; do not expose it to the LAN as a shortcut.

## Linux / KDE status

The README documents the initial Linux Python CLI and local web workflow. Synthetic speech has been transcribed on Linux with faster-whisper CPU; CachyOS/KDE and physical devices have not been tested. The live input path is single-input only and does not provide separate system and microphone tracks. The current macOS management script is not a Linux installer.

Before implementing Linux capture, establish the target Plasma session, PipeWire/PulseAudio setup, input/output devices and GPU on the intended machine. Verify real microphone and playback samples separately, changing default devices, silence/failure reporting, pause/resume and private storage; record the test environment and actual results. Tests must not rely on real user recordings.

The PipeWire capture prototype is exercised with `uv run python -m listn.pipewire [--dry-run|--duration N]` and `uv run pytest tests/test_pipewire.py`; observed PipeWire behaviour and the exact limits of the current verification (headless container, no audio hardware, no KDE session) are recorded in [LINUX_CAPTURE](LINUX_CAPTURE.md). Run it on the target machine before treating Linux capture as working.

The desktop integration (menu entry, optional tray at login, PKGBUILD) has its own checks: `uv run pytest tests/test_desktop_packaging.py` parses both `.desktop` files, executes `scripts/listn-desktop.sh install --autostart` / `status` / `uninstall` against a throwaway `$HOME` and asserts only the XDG directories change, sources the PKGBUILD to check its declared depends, and asserts the README documents the same commands the files run. On a desktop, confirm by hand: `scripts/listn-desktop.sh status`, then open the KDE app menu and look for **listn** (`desktop-file-validate packaging/*.desktop` and `shellcheck scripts/listn-desktop.sh` also run in the suite when installed). The PKGBUILD builds only from a release tag the repository does not have yet; nothing in it runs at build time, and a real `makepkg` run still needs an Arch machine.

The dashboard is validated the same way: `uv run pytest tests/test_dashboard.py` covers the API contract (empty state, live, processing, failed, track separation, loopback/origin checks), and a manual pass against a running server — `uv run listn ui --no-browser`, then open `/dashboard` or `curl http://127.0.0.1:<port>/api/dashboard` — shows the same engine state. It reads state only; it never records, writes or sends anything itself.
