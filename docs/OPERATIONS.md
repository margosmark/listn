# Operations and development

## Existing macOS workflow

The supported installation and management commands are in [README](../README.md). `scripts/listn-manage.sh` and the Makefile manage macOS app builds and installation; inspect the command before running it. The system-audio tap requires macOS 14.2 or later. Real audio and permission verification needs a Mac, not just unit tests.

## Python development

The project declares Python `>=3.11,<3.14` and uses `uv.lock`. From the repository root, with `uv` installed:

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

The desktop integration (menu entry, optional tray at login, PKGBUILD) has its own checks: `uv run pytest tests/test_desktop_packaging.py` parses both `.desktop` files, executes `scripts/listn-desktop.sh install --autostart` / `status` / `uninstall` against a throwaway `$HOME`, sources the PKGBUILD to check its declared depends and against the newest release tag (currently `v0.5.0`, which must equal `pkgver`; the pinned `sha256sums` is re-measured against the archive GitHub actually serves), runs the real `package()` against that pinned tag archive and asserts the exact file set it owns — `/usr` only, no `/etc`, no venv or caches — checks the `.install` install/upgrade/removal messages, and asserts the README documents the same commands the files run. `tests/shell/test_desktop_lifecycle.sh` (run by the same pytest file, and standalone with `sh`) walks install → explicit login opt-in → re-install/upgrade → status → uninstall → dry-run against an isolated `$HOME`, proving uninstall removes only listn's own files while transcripts, settings, the model cache, the user's uv environment and other applications' entries survive, and that no system XDG directory changes. `scripts/arch_pkg_check.sh` performs the clean Arch lifecycle end to end — makepkg from the tag source with checksum verification, `pacman -U`, first run (`listn --version` building the user's environment), one-engine-at-login (`listn tray` attaches to a running engine instead of starting a second), a pkgrel-bump rebuild and upgrade, then `pacman -R` with retention assertions — natively where `makepkg` exists, re-executed in an `archlinux` container via docker/podman, or SKIP with exit 0 and no writes where neither is available; `.github/workflows/arch-package.yml` runs it in an `archlinux:latest` container alongside the shell lifecycle test. `desktop-file-validate packaging/*.desktop` and `shellcheck scripts/*.sh` also run in the suite when installed.

All of the above are shell and container checks. None of them starts plasmashell, shows the app menu, displays the tray panel or logs a session in: on a desktop, confirm by hand that `scripts/listn-desktop.sh status` is truthful, the KDE app menu shows **listn**, the tray appears after `listn tray`, and — only if you copied the autostart template — exactly one engine starts at login. A real `makepkg` run outside CI still needs an Arch machine (or `scripts/arch_pkg_check.sh` with docker/podman); nothing in the PKGBUILD runs at build time.

The Settings → Launch at login toggle (the XDG autostart entry) has its own checks: `uv run pytest tests/test_autostart.py` covers writing `$XDG_CONFIG_HOME/autostart/listn.desktop` with the running launcher's `Exec`, the marker-guarded delete (an entry listn did not write is never removed), idempotent re-writes, refusal to write outside `$HOME`, macOS staying out of it, and status that follows the file on disk instead of the saved setting. It also checks both escaping layers on a hostile launcher path (spaces, quotes, `$()`, backticks and backslashes) through `desktop-file-validate` and a real GLib/GIO round trip, every desktop-entry reserved character quoted on its own (a lone `'` in the install prefix used to leave the Exec unparseable — the entry then died at login while Settings still showed ON), a real GIO launch through an apostrophe-only prefix delivering the exact argv and `PYTHONPATH`, and that a symlink planted at the fixed temp path is never followed while a stale temp is cleared (`O_EXCL`/`O_NOFOLLOW`). On a desktop, flip the toggle in Settings, check `~/.config/autostart/listn.desktop`, delete it by hand and confirm the toggle reads off again.

The dashboard is validated the same way: `uv run pytest tests/test_dashboard.py` covers the API contract (empty state, live, processing, failed, track separation, loopback/origin checks), and a manual pass against a running server — `uv run listn ui --no-browser`, then open `/dashboard` or `curl http://127.0.0.1:<port>/api/dashboard` — shows the same engine state. It reads state only; it never records, writes or sends anything itself.

Linux meeting detection and tray prompts (see [LINUX_MEETINGS](LINUX_MEETINGS.md)) are fixture-tested with `uv run pytest tests/test_meeting_watch.py`: supported apps from `pw-dump`-shaped fixtures, browser/huddle title rules, false positives, polling dedup, consent (no start without a click or an explicitly enabled auto-record), dismiss/snooze/end offer, and prompt metadata without window titles. A real meeting app, XWayland titles and the prompt as rendered by plasmashell still need a manual pass on the target desktop.
