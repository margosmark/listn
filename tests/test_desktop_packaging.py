"""Linux desktop integration: .desktop entries, XDG autostart, PKGBUILD, docs.

Everything here is checkable without a KDE session: the entries are parsed as
text, the installer script is executed against a throwaway $HOME, the PKGBUILD
is sourced by bash, and the README is asserted to document the same commands
the files actually run.
"""

from __future__ import annotations

import configparser
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from listn import cli
from listn.tray import TrayStatus, build_menu

ROOT = Path(__file__).resolve().parents[1]
MENU_DESKTOP = ROOT / "packaging" / "listn.desktop"
AUTOSTART_DESKTOP = ROOT / "packaging" / "listn-tray.desktop"
ICON = ROOT / "macos" / "Resources" / "AppIcon.iconset" / "icon_128x128.png"
INSTALLER = ROOT / "scripts" / "listn-desktop.sh"
PKGDIR = ROOT / "packaging" / "arch"
PKGBUILD = PKGDIR / "PKGBUILD"
LAUNCHER = PKGDIR / "listn-launcher.sh"
README = ROOT / "README.md"


def read_desktop(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str  # type: ignore[method-assign]  # desktop keys are case-sensitive
    parser.read(path, encoding="utf-8")
    assert "Desktop Entry" in parser.sections(), path
    return dict(parser["Desktop Entry"])


def cli_subcommands() -> set[str]:
    import argparse

    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("listn CLI has no subcommands")


# ---------------------------------------------------------------- the entries

def test_menu_entry_keys_are_correct():
    entry = read_desktop(MENU_DESKTOP)
    assert entry["Type"] == "Application"
    assert entry["Version"] == "1.0"
    assert entry["Name"] == "listn"
    assert entry["Terminal"] == "false"
    assert entry["Icon"] == "listn"
    # The repo entry must stay relocatable: no absolute path to any machine.
    assert not entry["Exec"].startswith("/")
    assert "hermes" not in MENU_DESKTOP.read_text(encoding="utf-8")
    assert "/home/" not in MENU_DESKTOP.read_text(encoding="utf-8")
    # Valid category list: main category, trailing semicolon, no empty slot.
    cats = entry["Categories"]
    assert cats.endswith(";") and ";;" not in cats
    assert "AudioVideo" in cats.split(";")
    assert entry["Comment"] and entry["Keywords"].endswith(";")


def test_menu_entry_launches_a_real_cli_subcommand():
    entry = read_desktop(MENU_DESKTOP)
    argv = entry["Exec"].split()
    assert argv[0] == "listn"
    assert argv[1:] == ["ui"]
    assert set(argv[1:]) <= cli_subcommands()


def test_autostart_entry_is_the_tray():
    entry = read_desktop(AUTOSTART_DESKTOP)
    assert entry["Type"] == "Application"
    assert entry["Exec"] == "listn tray"
    assert entry["Terminal"] == "false"
    assert entry["NoDisplay"] == "true"        # never show up in app menus
    assert entry["X-GNOME-Autostart-enabled"] == "true"
    assert entry["Icon"] == "listn"
    assert not entry["Exec"].startswith("/")
    text = AUTOSTART_DESKTOP.read_text(encoding="utf-8")
    assert "hermes" not in text and "/home/" not in text
    assert "tray" in cli_subcommands() and "ui" in cli_subcommands()


def test_install_time_choice_not_a_tray_toggle():
    """The autostart file is an install-time choice: the tray has no
    launch-at-login control (that macOS control stays hidden)."""
    tray_text = (ROOT / "listn" / "tray.py").read_text(encoding="utf-8")
    first_quote = tray_text.index('"""')
    last_quote = tray_text.index('"""', first_quote + 3)
    docstring, code = tray_text[:last_quote], tray_text[last_quote:].lower()
    # Anything about login/autostart outside the honest "does not exist"
    # module docstring would be a control creeping into the tray.
    assert "autostart" not in code
    assert "login" not in code
    assert "launch-at-login" in docstring.lower()  # documented as absent

    labels = [item.label for item in
              build_menu(TrayStatus(), [], [], None, None)]
    for label in labels:
        low = label.lower()
        assert "login" not in low and "autostart" not in low
        assert "start at" not in low and "startup" not in low


# ------------------------------------------------------------------ the docs

def test_readme_documents_the_same_commands_the_files_run():
    readme = README.read_text(encoding="utf-8")
    menu_exec = read_desktop(MENU_DESKTOP)["Exec"]
    autostart_exec = read_desktop(AUTOSTART_DESKTOP)["Exec"]

    # Install / uninstall in the README Linux section, mirroring the macOS
    # listn-manage.sh command table.
    assert "scripts/listn-desktop.sh install" in readme
    assert "scripts/listn-desktop.sh uninstall" in readme
    assert "scripts/listn-desktop.sh status" in readme
    assert "install --autostart" in readme

    # The documented commands are the commands the entries actually run.
    menu_doc = re.search(r"The\s+menu entry runs `([^`]+)`", readme)
    autostart_doc = re.search(r"The\s+autostart entry runs `([^`]+)`", readme)
    assert menu_doc, "README must document what the menu entry runs"
    assert autostart_doc, "README must document what the autostart entry runs"
    assert menu_doc.group(1) == menu_exec
    assert autostart_doc.group(1) == autostart_exec

    # The PKGBUILD path is documented, honestly, including that it builds from
    # a release tag the repository does not have yet.
    assert "packaging/arch/PKGBUILD" in readme
    assert "makepkg" in readme
    assert "no tags" in readme

    # The installer renders each entry with exactly the documented subcommand.
    installer = INSTALLER.read_text(encoding="utf-8")
    assert 'render "$desktop_src" "$bin" ui' in installer
    assert 'render "$autostart_src" "$bin" tray' in installer


def test_readme_says_autostart_is_user_scope_and_not_a_tray_toggle():
    readme = README.read_text(encoding="utf-8")
    section = readme[readme.index("## Linux (CachyOS / KDE)"):readme.index("## Quick start")]
    section = re.sub(r"\s+", " ", section)
    assert "user scope" in section.lower()
    assert "install-time choice" in section
    assert "no launch-at-login control" in section


# --------------------------------------------------------------- the script

def test_installer_syntax_and_macos_guard():
    assert INSTALLER.stat().st_mode & 0o111, "installer must be executable"
    check = subprocess.run(["sh", "-n", str(INSTALLER)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    text = INSTALLER.read_text(encoding="utf-8")
    assert '[ "$(uname -s)" = "Darwin" ]' in text
    assert "scripts/listn-manage.sh" in text
    # Standard XDG paths, no machine-specific absolute path.
    assert '${XDG_DATA_HOME:-$HOME/.local/share}/applications' in text
    assert '${XDG_CONFIG_HOME:-$HOME/.config}/autostart' in text
    assert "/home/" not in text and "hermes" not in text


def run_installer(args: list[str], home: Path, listn_cmd: Path | str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("XDG_DATA_HOME", None)
    env.pop("XDG_CONFIG_HOME", None)
    env.update({"HOME": str(home), "LISTN_CMD": str(listn_cmd)})
    return subprocess.run(["sh", str(INSTALLER), *args],
                          capture_output=True, text=True, env=env, cwd=ROOT)


def test_installer_writes_status_and_removes_entries(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_bin = tmp_path / "bin" / "listn"
    fake_bin.parent.mkdir()
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(0o755)

    # install --autostart: menu entry, icon and autostart entry, user scope.
    result = run_installer(["install", "--autostart"], home, fake_bin)
    assert result.returncode == 0, result.stderr

    menu = home / ".local/share/applications/listn.desktop"
    icon = home / ".local/share/icons/hicolor/128x128/apps/listn.png"
    autostart = home / ".config/autostart/listn-tray.desktop"
    assert menu.is_file() and icon.is_file() and autostart.is_file()

    # Only the Exec line differs from the shipped template.
    src_lines = MENU_DESKTOP.read_text(encoding="utf-8").splitlines()
    out_lines = menu.read_text(encoding="utf-8").splitlines()
    diffs = [(a, b) for a, b in zip(src_lines, out_lines) if a != b]
    assert len(src_lines) == len(out_lines)
    assert len(diffs) == 1
    assert diffs[0][0] == "Exec=listn ui"
    assert diffs[0][1] == f'Exec="{fake_bin}" ui'

    assert read_desktop(autostart)["Exec"] == f'"{fake_bin}" tray'
    assert icon.read_bytes() == ICON.read_bytes()

    # status reports the login-time choice truthfully.
    status = run_installer(["status"], home, fake_bin)
    assert status.returncode == 0
    assert "installed:" in status.stdout
    assert "tray starts at login" in status.stdout

    # uninstall removes everything it wrote.
    gone = run_installer(["uninstall"], home, fake_bin)
    assert gone.returncode == 0, gone.stderr
    assert not menu.exists() and not icon.exists() and not autostart.exists()
    status = run_installer(["status"], home, fake_bin)
    assert "tray does not start at login" in status.stdout


def test_installer_without_autostart_leaves_login_alone(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = run_installer(["install"], home, "/usr/bin/listn")
    assert result.returncode == 0, result.stderr
    assert (home / ".local/share/applications/listn.desktop").is_file()
    assert not (home / ".config/autostart/listn-tray.desktop").exists()
    status = run_installer(["status"], home, "/usr/bin/listn")
    assert "tray does not start at login" in status.stdout


def test_installer_dry_run_writes_nothing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = run_installer(["install", "--autostart", "--dry-run"], home, "/usr/bin/listn")
    assert result.returncode == 0, result.stderr
    assert "dry-run:" in result.stdout
    assert list(home.iterdir()) == []


# ---------------------------------------------------------------- PKGBUILD

def source_pkgbuild() -> dict[str, str]:
    script = (
        'source PKGBUILD || exit 1\n'
        'printf "pkgver=%s\\n" "$pkgver"\n'
        'printf "install=%s\\n" "$install"\n'
        'printf "depends=%s\\n" "${depends[*]}"\n'
        'printf "optdepends=%s\\n" "${optdepends[*]}"\n'
        'printf "source=%s\\n" "${source[0]}"\n'
        'printf "sums=%s\\n" "${sha256sums[0]}"\n'
    )
    result = subprocess.run(["bash", "-c", script], cwd=PKGDIR,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def test_pkgbuild_depends_are_honest_and_match_pyproject():
    fields = source_pkgbuild()
    assert set(fields["depends"].split()) == {
        "python", "uv", "portaudio", "pipewire", "wireplumber",
    }
    assert "tray" in fields["optdepends"]
    # The packaged version is the project's version.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert match, "pyproject must declare a version"
    version = match.group(1)
    assert fields["pkgver"] == version
    assert fields["install"] == "listn.install"


def test_pkgbuild_builds_from_the_tag_and_says_so():
    fields = source_pkgbuild()
    # Source is the release tag archive of this repo ...
    version = fields["pkgver"]
    assert fields["source"] == (
        f"listn-{version}.tar.gz::"
        f"https://github.com/margosmark/listn/archive/refs/tags/v{version}.tar.gz"
    )
    # ... and the checksum gap (no tags exist yet) is stated, not hidden.
    text = PKGBUILD.read_text(encoding="utf-8")
    assert fields["sums"] == "SKIP"
    assert "updpkgsums" in text
    assert "no tags yet" in text
    result = subprocess.run(["git", "ls-remote", "--tags", "origin"],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"cannot reach origin: {result.stderr.strip()}")
    assert result.stdout.strip() == "", (
        "a tag exists now: refresh sha256sums with updpkgsums and update the "
        "PKGBUILD comment and this test")


def test_pkgbuild_ships_only_user_scope_autostart_and_no_root_state():
    text = PKGBUILD.read_text(encoding="utf-8")
    # Syntax must be valid bash.
    assert subprocess.run(["bash", "-n", str(PKGBUILD)]).returncode == 0
    # Launcher + menu entry + icon are installed; the tray-at-login template is
    # only shipped for the user to copy, never into the system autostart dir.
    assert 'install -Dm755 packaging/arch/listn-launcher.sh "$pkgdir/usr/bin/listn"' in text
    assert '"$pkgdir/usr/share/applications/listn.desktop"' in text
    assert '"$pkgdir/usr/share/icons/hicolor/128x128/apps/listn.png"' in text
    assert '"$pkgdir/usr/share/listn/packaging/listn-tray.desktop"' in text
    assert "$pkgdir/etc/xdg" not in text
    # No venv or other state is built as root: package() must not run uv.
    pkg_body = text.split("package()", 1)[1]
    assert "uv sync" not in pkg_body and "uv run" not in pkg_body


def test_launcher_keeps_state_in_the_user_home():
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "UV_PROJECT_ENVIRONMENT" in text
    assert "${XDG_DATA_HOME:-$HOME/.local/share}/listn/env" in text
    assert "--frozen" in text
    # The tray extra is pulled in only by the tray command.
    assert '--frozen --extra tray listn "$@"' in text
    assert "/usr/share/listn" in text


# ------------------------------------------------------ optional real tools

@pytest.mark.skipif(shutil.which("desktop-file-validate") is None,
                    reason="desktop-file-utils not installed")
@pytest.mark.parametrize("path", [MENU_DESKTOP, AUTOSTART_DESKTOP])
def test_desktop_files_pass_desktop_file_validate(path):
    result = subprocess.run(["desktop-file-validate", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_installer_passes_shellcheck():
    result = subprocess.run(["shellcheck", "-s", "sh", str(INSTALLER)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
