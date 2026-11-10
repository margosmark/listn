"""Linux desktop integration: .desktop entries, XDG autostart, PKGBUILD, docs.

Everything here is checkable without a KDE session: the entries are parsed as
text, the installer script is executed against a throwaway $HOME (plus a
standalone POSIX shell lifecycle script under tests/shell/), the PKGBUILD is
sourced by bash and its package() run against the pinned tag archive, and the
README is asserted to document the same commands the files actually run.

What this file cannot check is a real Plasma login: app-menu visibility, the
tray panel appearing in the system tray, and a session autostart actually
firing stay manual acceptance (docs/OPERATIONS.md).
"""

from __future__ import annotations

import configparser
import hashlib
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from listn import __version__, autostart, cli
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
              build_menu(TrayStatus(), [], [], [], None, None)]
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
    # a release tag and how the checksum is refreshed when a new one is cut.
    assert "packaging/arch/PKGBUILD" in readme
    assert "makepkg" in readme
    assert "updpkgsums" in readme

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


def tag_version_key(name: str):
    """Total order over tag names like '0.5.0' / '0.10.0-rc1': numeric runs
    compare as numbers, everything else as text."""
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"[._-]", name.lstrip("v")))


def origin_tags() -> list[str]:
    result = subprocess.run(["git", "ls-remote", "--tags", "--refs", "origin"],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"cannot reach origin: {result.stderr.strip()}")
    return sorted((line.split("refs/tags/")[1]
                   for line in result.stdout.splitlines()
                   if "refs/tags/" in line), key=tag_version_key)


def test_pkgbuild_builds_from_the_tag_and_says_so():
    fields = source_pkgbuild()
    # Source is the release tag archive of this repo ...
    version = fields["pkgver"]
    assert fields["source"] == (
        f"listn-{version}.tar.gz::"
        f"https://github.com/margosmark/listn/archive/refs/tags/v{version}.tar.gz"
    )
    # ... and the checksum is measured, not a SKIP gap; the comment says how
    # to refresh it for the next tag.
    text = PKGBUILD.read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{64}", fields["sums"]), (
        "sha256sums must be the measured hash of the tag archive: bump "
        "pkgver and run updpkgsums after cutting a tag")
    assert "updpkgsums" in text
    # Tripwire for the NEXT tag: this stays red until pkgver tracks the
    # newest tag on origin and updpkgsums has re-measured its archive.
    tags = origin_tags()
    assert tags, "no tags on origin - the PKGBUILD builds from one"
    assert f"v{version}" == tags[-1], (
        f"origin's newest tag is {tags[-1]} but the PKGBUILD builds "
        f"v{version}: bump pkgver, run updpkgsums, and update this test")


@pytest.fixture(scope="session")
def tag_archive(tmp_path_factory) -> Path:
    """The published tag archive the PKGBUILD downloads, fetched once.

    The repo is private, so the fetch borrows git's stored credentials the
    way makepkg's own curl config does. Without credentials (or curl) the
    dependent tests skip rather than guess at a source.
    """
    fields = source_pkgbuild()
    url = fields["source"].split("::", 1)[1]
    curl = shutil.which("curl") or ""
    if not curl:
        pytest.skip("curl not installed")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    env.pop("GIT_ASKPASS", None)
    cred = subprocess.run(["git", "credential", "fill"], cwd=ROOT, text=True,
                          input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, env=env)
    secrets = dict(line.split("=", 1) for line in cred.stdout.splitlines()
                   if "=" in line)
    if (cred.returncode != 0 or not secrets.get("username")
            or not secrets.get("password")):
        pytest.skip("no stored GitHub credentials to fetch the archive")
    login = f'{secrets["username"]}:{secrets["password"]}'
    login = login.replace("\\", "\\\\").replace('"', '\\"')
    archive = tmp_path_factory.mktemp("tag-archive") / "tag.tar.gz"
    fetch = subprocess.run([curl, "-fsSL", "-K", "-", "-o", str(archive), url],
                           input=f'user = "{login}"\n', text=True,
                           capture_output=True, cwd=ROOT, env=env)
    if fetch.returncode != 0:
        pytest.skip(f"cannot fetch the tag archive: {fetch.stderr.strip()}")
    return archive


def test_pkgbuild_sha256_is_the_hash_of_the_published_tag_archive(tag_archive):
    """The sum must be what GitHub actually serves for the tag archive the
    PKGBUILD downloads - a moved or regenerated tag goes red here before
    makepkg fails on the user's machine.
    """
    fields = source_pkgbuild()
    digest = hashlib.sha256(tag_archive.read_bytes()).hexdigest()
    assert digest == fields["sums"], (
        f"GitHub serves sha256 {digest} for v{fields['pkgver']} but the "
        f"PKGBUILD pins {fields['sums']}: run updpkgsums")


def test_pkgbuild_package_fileset_is_exact_and_stateless(tag_archive, tmp_path):
    """package() output is exactly the read-only app files: nothing under
    /etc, no venv, no caches, no build-time state - the "clean install"
    contract checked by running the real package() against the pinned tag
    source in a throwaway pkgdir (no root, no system writes).
    """
    fields = source_pkgbuild()
    version = fields["pkgver"]
    # We are packaging the pinned bytes, not some other tree.
    assert hashlib.sha256(tag_archive.read_bytes()).hexdigest() == fields["sums"]

    src_root = tmp_path / "src"
    src_root.mkdir()
    subprocess.run(["tar", "-xzf", str(tag_archive), "-C", str(src_root)],
                   check=True, capture_output=True)
    source_dir = src_root / f"listn-{version}"
    assert source_dir.is_dir(), sorted(p.name for p in src_root.iterdir())

    pkgdir = tmp_path / "pkg"
    # The packager's flow: the repo's current PKGBUILD against the tagged
    # source it downloads - so source the worktree PKGBUILD, not the tag's.
    run = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(PKGBUILD))} && package"],
        cwd=src_root, env=dict(os.environ, pkgdir=str(pkgdir)),
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    files = sorted(p.relative_to(pkgdir).as_posix()
                   for p in pkgdir.rglob("*") if p.is_file())
    expected = [
        "usr/bin/listn",
        "usr/share/applications/listn.desktop",
        "usr/share/icons/hicolor/128x128/apps/listn.png",
        "usr/share/listn/README.md",
        "usr/share/listn/pyproject.toml",
        "usr/share/listn/uv.lock",
        "usr/share/listn/packaging/listn-tray.desktop",
    ]
    expected += [f"usr/share/listn/listn/{p.relative_to(source_dir / 'listn').as_posix()}"
                 for p in sorted((source_dir / "listn").rglob("*")) if p.is_file()]
    assert files == sorted(expected)

    # No runtime/build state ships: system config dirs, caches, envs, git.
    assert not [f for f in files if f.startswith(("etc/", "var/", "home/"))]
    assert not [f for f in files
                if "__pycache__" in f or ".git" in f or "egg-info" in f
                or "/env/" in f or f.endswith(".pyc")]

    # The packaged files are byte-identical to the source they came from,
    # and the tag's own version matches the pkgver being built.
    assert ((pkgdir / "usr/bin/listn").read_bytes()
            == (source_dir / "packaging/arch/listn-launcher.sh").read_bytes())
    assert ((pkgdir / "usr/share/applications/listn.desktop").read_bytes()
            == (source_dir / "packaging/listn.desktop").read_bytes())
    tag_pyproject = (source_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'^version = "{re.escape(version)}"', tag_pyproject,
                     re.M), (f"tag v{version} source does not declare version "
                             f"{version}: pkgver and source disagree")


def test_install_file_messages_cover_the_lifecycle():
    """pacman's .install must tell the truth about the three hooks: how to
    opt in to login, that upgrade repeats the guidance, and that removal
    leaves the user's environment, data and login entry alone."""
    text = (PKGDIR / "listn.install").read_text(encoding="utf-8")
    for hook in ("post_install", "post_upgrade", "post_remove"):
        assert re.search(rf"^{hook}\(\)", text, re.M), hook
    # Upgrade reprints the install guidance rather than going silent.
    assert re.search(r"post_upgrade\(\) \{\s*\n\s*post_install\b", text)
    # Opt-in is the shipped template copied into the user's own autostart dir.
    assert "/usr/share/listn/packaging/listn-tray.desktop" in text
    assert ".config/autostart/listn-tray.desktop" in text
    # Removal promises retention, in the user's own paths.
    remove = text.split("post_remove()", 1)[1]
    assert "~/.local/share/listn/env" in remove
    assert "~/.local/share/listn" in remove
    assert ".config/autostart/" in remove
    assert "left in place" in remove
    # It never tells pacman's user to delete their own data for them.
    assert "rm -rf" not in remove


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
    # Read-only /usr/share/listn: sync the locked deps only, never build the
    # project (setuptools would need to write into the source tree), and run
    # the module from there with PYTHONPATH.
    assert "--frozen --no-install-project --no-dev" in text
    # F2: the module runs with the cwd kept off sys.path (-P on >= 3.11, the
    # runpy shim below on the 3.10 floor).
    assert '"$env/bin/python" -P -m listn "$@"' in text
    assert 'PYTHONPATH="$project' in text
    # The tray extra is pulled in only by the tray command.
    assert "--no-install-project --no-dev --extra tray" in text
    assert text.count("--extra tray") == 1
    assert "/usr/share/listn" in text


def test_launcher_gates_the_safe_path_flag_on_the_python_floor():
    """F2: neither ``-P`` nor ``PYTHONSAFEPATH`` exists on Python 3.10.

    The floor in pyproject.toml is 3.10, so the launcher probes the
    interpreter it is about to ``exec`` and falls back to a runpy shim —
    byte-identical to the one ``listn/autostart.py`` renders into its Exec
    line, so the two shipped launch paths cannot drift apart.
    """
    text = LAUNCHER.read_text(encoding="utf-8")
    probe = ('if "$env/bin/python" -c '
             "'import sys;sys.exit(sys.version_info < (3, 11))'")
    assert probe in text, "the -P branch is not gated on the interpreter"
    shim = re.search(r'-c "(.+)" "\$@"', text)
    assert shim, "the launcher has no runpy shim fallback for Python 3.10"
    assert shim.group(1) == autostart.SAFE_RUN_SHIM
    # The shim passes through shell quoting and (in autostart) both
    # desktop-entry escaping layers, so none of their metacharacters.
    assert not set(autostart.SAFE_RUN_SHIM) & set('"\\`$%')


def test_launcher_ignores_a_planted_listn_package_in_its_cwd(tmp_path):
    """F2: ``cd <dir-with-planted-listn-package> && listn --version`` prints
    the real version.

    The shipped script runs as-is (only ``uv sync`` is stubbed — the sync is
    not what this is about) from a directory holding a hostile ``listn/``
    package, with PYTHONPATH pointing at the real tree the way the packaged
    PYTHONPATH does. The control run, without the fix, must load the planted
    package — otherwise the test could not detect the hijack at all.
    """
    planted = tmp_path / "cwd"
    (planted / "listn").mkdir(parents=True)
    (planted / "listn" / "__init__.py").write_text(
        'print("PLANTED_PKG_LOADED_INSTEAD")\n', encoding="utf-8")
    (planted / "listn" / "__main__.py").write_text(
        'import sys\nprint("PLANTED_PKG_LOADED_INSTEAD")\nsys.exit(3)\n',
        encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)

    env = {**os.environ,
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "UV_PROJECT_ENVIRONMENT": str(env_dir),
           "PYTHONPATH": str(ROOT)}
    control = subprocess.run([sys.executable, "-m", "listn", "--version"],
                             cwd=planted, env=env, capture_output=True, text=True,
                             timeout=120)
    assert "PLANTED_PKG_LOADED_INSTEAD" in control.stdout, control.stdout

    run = subprocess.run(["sh", str(LAUNCHER), "--version"], cwd=planted, env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == f"listn {__version__}"
    assert "PLANTED_PKG_LOADED_INSTEAD" not in run.stdout + run.stderr


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



# ------------------------------------- isolated-HOME shell lifecycle test

SHELL_LIFECYCLE = ROOT / "tests" / "shell" / "test_desktop_lifecycle.sh"


def test_isolated_home_shell_lifecycle_passes():
    """The standalone POSIX shell script walks the whole lifecycle -
    install, explicit login opt-in, upgrade re-install, status, uninstall,
    dry-run, platform guard - against a throwaway $HOME, and proves
    uninstall removed only listn's files while user data (transcripts,
    settings, model cache, environment) and other apps' files survived.
    """
    assert SHELL_LIFECYCLE.is_file(), "committed shell lifecycle test missing"
    result = subprocess.run(["sh", str(SHELL_LIFECYCLE)],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout + result.stderr
    checks = sum(1 for line in result.stdout.splitlines()
                 if line.startswith("ok "))
    assert checks >= 40, f"only {checks} lifecycle checks ran:\n{result.stdout}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None,
                    reason="shellcheck not installed")
def test_shell_lifecycle_passes_shellcheck():
    result = subprocess.run(["shellcheck", "-s", "sh", str(SHELL_LIFECYCLE)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


# ---------------------------------------- one engine at login (`listn tray`)

class _QuietTrayItem:
    """Stand-in for the D-Bus StatusNotifierItem: construct, run once, exit.

    cmd_tray's engine ownership logic is what is under test here, not the
    session bus - a real D-Bus tray belongs to a Plasma login test.
    """

    def __init__(self, controller):
        self.controller = controller

    async def run(self):
        return None


def _listening_socket(port: int | None = None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port if port is not None else 0))
    sock.listen(4)
    return sock


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _must_not_start(*_args, **_kwargs):
    raise AssertionError("start_embedded_server called but an engine answers / "
                         "must not start one")


def test_login_tray_attaches_to_a_running_engine(monkeypatch, capsys, data_dir):
    """The login entry (`listn tray`) must never start a second engine: with
    one already answering on the port it attaches and starts nothing.
    """
    from listn import cli, tray as tray_mod
    import listn.tray_sni as sni_mod

    listener = _listening_socket()
    port = listener.getsockname()[1]
    monkeypatch.setattr(tray_mod, "start_embedded_server", _must_not_start)
    monkeypatch.setattr(sni_mod, "StatusNotifierItem", _QuietTrayItem)
    try:
        args = cli.build_parser().parse_args(["tray", "--port", str(port)])
        assert cli.cmd_tray(args) == 0
        out = capsys.readouterr().out
        assert f"attached to the engine on 127.0.0.1:{port}" in out
        assert "engine started" not in out
    finally:
        listener.close()


def test_tray_starts_one_engine_then_a_second_tray_attaches(
        monkeypatch, capsys, data_dir):
    """With nothing answering, the first tray starts the one engine; a tray
    started afterwards (user opens the menu while autostart already ran)
    attaches to it instead of duplicating it.
    """
    from listn import cli, tray as tray_mod
    import listn.tray_sni as sni_mod

    monkeypatch.setattr(sni_mod, "StatusNotifierItem", _QuietTrayItem)
    port = _free_port()
    starts: list[int] = []
    holder: dict[str, socket.socket] = {}

    class _OwnedServer:
        should_exit = False

    def fake_start(p: int):
        holder["sock"] = _listening_socket(p)
        starts.append(p)
        thread = threading.Thread(target=lambda: None)
        thread.start()   # cmd_tray joins it on the way out
        return _OwnedServer(), thread

    monkeypatch.setattr(tray_mod, "start_embedded_server", fake_start)
    try:
        args = cli.build_parser().parse_args(["tray", "--port", str(port)])
        assert cli.cmd_tray(args) == 0
        out = capsys.readouterr().out
        assert f"engine started on 127.0.0.1:{port}" in out
        assert starts == [port]

        assert cli.cmd_tray(args) == 0
        out = capsys.readouterr().out
        assert f"attached to the engine on 127.0.0.1:{port}" in out
        assert starts == [port], "second tray must not start another engine"
    finally:
        holder.get("sock", socket.socket()).close()


def test_tray_no_server_refuses_instead_of_starting_one(
        monkeypatch, capsys, data_dir):
    """`--no-server` (attach-only) with nothing answering fails honestly and
    starts nothing - the login entry never silently spawns an engine twice.
    """
    from listn import cli, tray as tray_mod

    monkeypatch.setattr(tray_mod, "start_embedded_server", _must_not_start)
    args = cli.build_parser().parse_args(
        ["tray", "--no-server", "--port", str(_free_port())])
    assert cli.cmd_tray(args) == 1
    err = capsys.readouterr().err
    assert "not answering" in err
    assert "listn ui" in err


# ------------------------------------------------- Arch lifecycle CI script

ARCH_CHECK = ROOT / "scripts" / "arch_pkg_check.sh"
ARCH_WORKFLOW = ROOT / ".github" / "workflows" / "arch-package.yml"


def test_arch_lifecycle_check_script_is_sound():
    """scripts/arch_pkg_check.sh exists, parses, and carries the lifecycle
    assertions and the honest skip/Plasma wording the story requires."""
    assert ARCH_CHECK.is_file(), "Arch lifecycle check script missing"
    assert subprocess.run(["sh", "-n", str(ARCH_CHECK)]).returncode == 0
    assert ARCH_CHECK.stat().st_mode & 0o111, "script must be executable"
    text = ARCH_CHECK.read_text(encoding="utf-8")
    # Every lifecycle stage is asserted, not just run.
    for needle in ("makepkg -sf", "pacman -U", "pacman -R",
                   "/etc/xdg/autostart", "left in place",
                   "attached to the engine", "post_upgrade",
                   "retained through uninstall"):
        assert needle in text, f"missing lifecycle assertion: {needle}"
    # Honest boundaries: skip mode says what was NOT validated, and the
    # script never claims to be a Plasma login test.
    assert "SKIP: no makepkg and no docker/podman" in text
    assert "NOT a Plasma login test" in text
    # Opt-in only: removal of the login entry is never automatic.
    assert "without opt-in" in text


@pytest.mark.skipif(shutil.which("shellcheck") is None,
                    reason="shellcheck not installed")
def test_arch_lifecycle_check_passes_shellcheck():
    result = subprocess.run(["shellcheck", "-s", "sh", str(ARCH_CHECK)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_arch_lifecycle_check_skips_cleanly_without_an_arch_userland():
    """Where neither makepkg nor a container runtime exists, the check must
    exit 0 with an honest SKIP and write nothing - not fail, not pretend."""
    if (shutil.which("makepkg") or shutil.which("docker")
            or shutil.which("podman")):
        pytest.skip("Arch userland or container runtime present: "
                    "run scripts/arch_pkg_check.sh for the full lifecycle")
    result = subprocess.run(["sh", str(ARCH_CHECK)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP" in result.stdout
    assert "NOT validated" in result.stdout


def test_arch_workflow_runs_the_checks_in_an_arch_container():
    import yaml

    wf = yaml.safe_load(ARCH_WORKFLOW.read_text(encoding="utf-8"))
    jobs = wf["jobs"]
    pkg = jobs["pkg-lifecycle"]
    assert pkg["container"] == "archlinux:latest"
    steps = yaml.dump(pkg["steps"])
    assert "arch_pkg_check.sh" in steps
    assert "GITHUB_TOKEN" in steps, "private tag archive needs the token"
    assert "actions/checkout" in steps
    desktop = yaml.dump(jobs["desktop-lifecycle"]["steps"])
    assert "tests/shell/test_desktop_lifecycle.sh" in desktop
    # The workflow itself says what it is not.
    assert "NOT a Plasma login test" in ARCH_WORKFLOW.read_text(encoding="utf-8")
