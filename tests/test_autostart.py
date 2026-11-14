"""Settings → Launch at login on Linux: the XDG autostart entry.

Every test runs against a throwaway ``$HOME``: nothing here may touch the
real one, and nothing may be written outside it. The status assertions are
about the file on disk, never about the stored bool alone.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from listn import __version__, autostart
from listn.config import Settings
from listn.engine import Engine
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD

FOREIGN = (
    "[Desktop Entry]\n"
    "Type=Application\n"
    "Name=mine\n"
    "Exec=/home/someone/bin/other tray\n"
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fresh $HOME with no XDG override — where the entry must land."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return h


@pytest.fixture
def entry(home):
    return home / ".config/autostart/listn.desktop"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """A packaged-style absolute console script in argv[0], like /usr/bin/listn."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "listn"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(exe), "ui"])
    return exe


@pytest.fixture
def hostile_launcher(monkeypatch, tmp_path):
    """A launcher path that stresses both escaping layers: a space, a quote,
    ``$()``, a backtick and a backslash — all legal in a Linux filename."""
    odd_dir = tmp_path / 'we"ird $(x) `tick` \\ dir'
    odd_dir.mkdir()
    exe = odd_dir / "listn"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(exe), "ui"])
    return exe


@pytest.fixture
def client(data_dir, fake_factory, home, launcher):
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


KEYFILE_ESCAPES = {"\\": "\\", "s": " ", "n": "\n", "t": "\t", "r": "\r"}


def keyfile_string(raw: str) -> str:
    """What GLib's ``g_key_file_get_string`` hands back for a raw value.

    The key-file *string* layer runs before any consumer reads the entry:
    ``\\s \\n \\t \\r \\\\`` are the only escapes, and any other backslash
    sequence makes the value un-interpretable — GLib then errors and the
    entry loads with no command at all. An invalid escape here fails the
    test for the same reason GLib would refuse it.
    """
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\":
            i += 1
            if i >= len(raw) or raw[i] not in KEYFILE_ESCAPES:
                raise AssertionError(f"invalid key-file escape in {raw!r}")
            out.append(KEYFILE_ESCAPES[raw[i]])
        else:
            out.append(raw[i])
        i += 1
    return "".join(out)


def read_entry(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str  # desktop keys are case-sensitive
    parser.read(path, encoding="utf-8")
    assert "Desktop Entry" in parser.sections(), path
    # configparser returns the raw bytes of each value; GLib applies the
    # key-file string layer on top, and that is the layer desktop-file-validate,
    # GIO and GNOME see first — so assertions here work at that layer.
    return {k: keyfile_string(v) for k, v in parser["Desktop Entry"].items()}


#: A real GLib/GIO reader, run by an interpreter that has PyGObject (the venv
#: does not): key-file string layer, Exec parsing, and what GIO makes of it.
GLIB_PROBE = """
import json
import sys

import gi

gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gio

path = sys.argv[1]
out = {}
try:
    keyfile = GLib.KeyFile()
    keyfile.load_from_file(path, GLib.KeyFileFlags.NONE)
    out["exec"] = keyfile.get_string("Desktop Entry", "Exec")
    _ok, argv = GLib.shell_parse_argv(out["exec"])
    out["argv"] = list(argv)
except GLib.Error as exc:
    out["error"] = str(exc)
info = Gio.DesktopAppInfo.new_from_filename(path)
out["cmdline"] = None if info is None else info.get_commandline()
print(json.dumps(out))
"""


def has_gi(candidate: str) -> bool:
    """Whether this interpreter can import PyGObject (the venv cannot)."""
    try:
        probe = subprocess.run([candidate, "-c", "import gi"],
                               capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def decode_exec(value: str) -> list[str]:
    """Desktop-entry ``Exec`` tokenizer, written from the spec for round trips.

    Arguments split on unquoted whitespace; ``"`` groups; ``\\`` makes the next
    character literal anywhere. There is no shell in the spec either.
    """
    args: list[str] = []
    buf: list[str] = []
    quoted = escape = started = False
    for ch in value:
        if escape:
            buf.append(ch)
            escape = False
            started = True
        elif ch == "\\":
            escape = started = True
        elif ch == '"':
            quoted = not quoted
            started = True
        elif ch in " \t" and not quoted:
            if started:
                args.append("".join(buf))
                buf, started = [], False
        else:
            buf.append(ch)
            started = True
    assert not escape, "Exec ends mid-escape"
    if started:
        args.append("".join(buf))
    return args


# ------------------------------------------------------------------- create


def test_toggle_on_writes_the_entry_under_home(client, home, entry, launcher):
    r = client.put("/api/settings", json={"launch_at_login": True})
    assert r.status_code == 200
    assert r.json()["launch_at_login"] is True

    assert entry.is_file()
    assert stat.S_IMODE(entry.stat().st_mode) == 0o644
    assert entry.resolve().is_relative_to(home.resolve())
    # The entry is the only file this writes, and it is inside $HOME.
    assert [p for p in home.rglob("*") if p.is_file()] == [entry]

    text = entry.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith(autostart.MARKER)

    desktop = read_entry(entry)
    assert desktop["Type"] == "Application"
    assert desktop["Terminal"] == "false"
    assert desktop["NoDisplay"] == "true"
    # Exactly the running launcher plus `tray`: no shell, no extra words.
    assert desktop["Exec"] == f"{launcher} tray"


def test_toggle_on_validates_as_a_desktop_entry(client, entry):
    if shutil.which("desktop-file-validate") is None:
        pytest.skip("desktop-file-utils not installed")
    client.put("/api/settings", json={"launch_at_login": True})
    result = subprocess.run(["desktop-file-validate", str(entry)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exec_is_quoted_with_desktop_entry_rules(client, monkeypatch, tmp_path, entry):
    """Spaces group with quotes; `$` and `%` are escaped. No shell involved."""
    spaced = tmp_path / "ap p"
    spaced.mkdir()
    exe = spaced / "listn$bin"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(exe), "ui"])

    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200
    desktop = read_entry(entry)
    expected = '"' + str(exe).replace("$", "\\$") + '" tray'
    assert desktop["Exec"] == expected
    # The value must be parseable back into exactly two arguments.
    assert decode_exec(desktop["Exec"]) == [str(exe), "tray"]


def test_toggle_on_validates_a_hostile_launcher_path(client, entry, hostile_launcher):
    """F1: space + ``"`` + ``$()`` + backtick + backslash must validate.

    The key-file string layer runs before the Exec layer, so a path that is
    only escaped for Exec (``\\"``, ``\\$``, ``\\` ``) is an invalid key-file
    value — GLib then refuses it and desktop-file-validate fails too.
    """
    if shutil.which("desktop-file-validate") is None:
        pytest.skip("desktop-file-utils not installed")
    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200
    result = subprocess.run(["desktop-file-validate", str(entry)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hostile_launcher_path_round_trips_through_glib(client, entry, hostile_launcher):
    """F1: real GLib must read Exec back and GIO must still see a command."""
    interpreters = [c for c in (sys.executable, "/usr/bin/python3") if has_gi(c)]
    if not interpreters:
        pytest.skip("PyGObject not installed for any interpreter")
    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200

    result = subprocess.run([interpreters[0], "-c", GLIB_PROBE, str(entry)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert "error" not in data, data["error"]          # key-file string layer is valid
    assert data["argv"] == [str(hostile_launcher), "tray"], data   # Exec layer too
    # GIO loads the entry with its command (before the fix: cmdline=None,
    # so the entry existed but could never autostart).
    assert data["cmdline"] is not None, data
    assert data["cmdline"] == data["exec"], data


# ------------------------------------------------------------------ disable


def test_toggle_off_removes_the_entry(client, entry):
    client.put("/api/settings", json={"launch_at_login": True})
    assert entry.is_file()

    r = client.put("/api/settings", json={"launch_at_login": False})
    assert r.status_code == 200
    assert r.json()["launch_at_login"] is False
    assert not entry.exists()


def test_toggle_off_with_no_file_is_a_noop(client, entry):
    r = client.put("/api/settings", json={"launch_at_login": False})
    assert r.status_code == 200
    assert r.json()["launch_at_login"] is False
    assert not entry.exists()          # nothing was created to "turn off"


def test_toggle_off_never_deletes_a_foreign_entry(client, entry):
    entry.parent.mkdir(parents=True)
    entry.write_text(FOREIGN, encoding="utf-8")

    r = client.put("/api/settings", json={"launch_at_login": False})
    assert r.status_code == 200
    assert entry.read_text(encoding="utf-8") == FOREIGN   # untouched, byte for byte
    # …and it does not count as listn's own entry, so the status is off.
    assert r.json()["launch_at_login"] is False


def test_toggle_on_refuses_to_overwrite_a_foreign_entry(client, entry, data_dir):
    entry.parent.mkdir(parents=True)
    entry.write_text(FOREIGN, encoding="utf-8")

    r = client.put("/api/settings", json={"launch_at_login": True})
    assert r.status_code == 400
    assert "not written by listn" in r.json()["detail"]
    assert entry.read_text(encoding="utf-8") == FOREIGN
    # The refusal did not persist a setting that never took effect.
    assert Settings.load().launch_at_login is False
    assert client.get("/api/settings").json()["launch_at_login"] is False


# --------------------------------------------------------------- idempotent


def test_toggle_on_twice_writes_the_same_file_once(client, entry):
    client.put("/api/settings", json={"launch_at_login": True})
    first = entry.stat()
    content = entry.read_text(encoding="utf-8")

    r = client.put("/api/settings", json={"launch_at_login": True})
    assert r.status_code == 200
    second = entry.stat()
    assert entry.read_text(encoding="utf-8") == content
    assert (second.st_ino, second.st_mtime_ns) == (first.st_ino, first.st_mtime_ns)


# ------------------------------------------------------- temp file hardening


def test_planted_symlink_at_the_temp_path_is_not_followed(client, home, entry, tmp_path):
    """F2: a symlink at the fixed temp name is never opened, moved or chmod'd.

    Whoever can write ``~/.config/autostart`` could point ``listn.desktop.tmp``
    at a victim file; the write must remove that link (not its target) and
    still land the entry as a regular file.
    """
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    victim.chmod(0o600)
    tmp = entry.with_name(entry.name + ".tmp")
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp.symlink_to(victim)

    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200

    # The victim keeps its bytes and its mode.
    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o600
    # The entry still lands: a real file, 0644, with a valid Exec.
    assert entry.is_file() and not entry.is_symlink()
    assert stat.S_IMODE(entry.stat().st_mode) == 0o644
    assert read_entry(entry)["Exec"].endswith(" tray")
    # Nothing of the temp file survives, and it is the only file written.
    assert not tmp.is_symlink() and not tmp.exists()
    assert [p for p in home.rglob("*") if p.is_file()] == [entry]
    assert client.get("/api/settings").json()["launch_at_login"] is True


def test_a_stale_regular_temp_file_is_cleared_before_writing(client, entry):
    """O_EXCL needs the fixed name cleared first: a leftover temp must not fail the toggle."""
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.with_name(entry.name + ".tmp")
    tmp.write_text("leftover from a crashed run\n", encoding="utf-8")

    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200
    assert entry.is_file()
    assert not tmp.exists()
    assert read_entry(entry)["Type"] == "Application"


# ------------------------------------------------------------- disk is truth


def test_status_follows_the_file_not_the_stored_bool(client, entry):
    assert client.put("/api/settings", json={"launch_at_login": True}).status_code == 200
    assert client.get("/api/settings").json()["launch_at_login"] is True
    assert client.get("/api/status").json()["settings"]["launch_at_login"] is True

    # Hand-deleted file: the stored bool is still True, the status is off.
    entry.unlink()
    assert Settings.load().launch_at_login is True
    assert client.get("/api/settings").json()["launch_at_login"] is False
    assert client.get("/api/status").json()["settings"]["launch_at_login"] is False

    # Hand-written listn entry: on again, without a PUT.
    entry.write_text(autostart.render(), encoding="utf-8")
    assert client.get("/api/settings").json()["launch_at_login"] is True

    # A foreign entry at the same path reads as off.
    entry.write_text(FOREIGN, encoding="utf-8")
    assert client.get("/api/settings").json()["launch_at_login"] is False


# ------------------------------------------------------------------- macOS


def test_macos_leaves_the_toggle_to_the_native_app(client, home, entry, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    r = client.put("/api/settings", json={"launch_at_login": True})
    assert r.status_code == 200
    assert r.json()["launch_at_login"] is True      # the stored bool still rules
    assert not entry.exists()
    assert [p for p in home.rglob("*") if p.is_file()] == []   # nothing written

    # Even a listn-marked file at the path is not read or touched on macOS.
    entry.parent.mkdir(parents=True)
    entry.write_text(autostart.render(), encoding="utf-8")
    assert client.get("/api/settings").json()["launch_at_login"] is True
    entry.unlink()
    assert client.get("/api/settings").json()["launch_at_login"] is True
    assert Settings.load().launch_at_login is True


# ------------------------------------------------------------ paths and scope


def test_xdg_config_home_is_honoured_when_inside_home(client, home, monkeypatch):
    xdg = home / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    client.put("/api/settings", json={"launch_at_login": True})
    assert (xdg / "autostart/listn.desktop").is_file()
    assert not (home / ".config/autostart/listn.desktop").exists()


def test_refuses_to_write_outside_home(client, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "elsewhere"))
    r = client.put("/api/settings", json={"launch_at_login": True})
    assert r.status_code == 400
    assert "outside $HOME" in r.json()["detail"]
    assert not (tmp_path / "elsewhere").exists()


def test_autostart_file_stays_under_home(home, entry):
    assert autostart.autostart_file() == entry


def test_source_adds_no_network_no_subprocess_no_shell():
    """Local-only by construction: no sockets, no child processes, no shell."""
    import ast

    tree = ast.parse(Path(autostart.__file__).read_text(encoding="utf-8"))
    imported = set()
    spawned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in {
                    "system", "popen", "Popen", "run", "call", "check_call",
                    "check_output", "spawn", "spawnl", "spawnv", "execv"}:
                spawned.append(fn.attr)
            if any(kw.arg == "shell" for kw in node.keywords):
                spawned.append("shell=True")
    assert not imported & {"socket", "subprocess", "urllib", "requests",
                           "http", "asyncio", "ftplib", "smtplib", "paramiko"}
    assert not spawned, spawned
    # Standard library and local file handling only.
    assert imported <= {"os", "shutil", "sys", "__future__", "pathlib"}


# -------------------------------------------------------------- launcher argv


def test_launcher_argv_prefers_the_running_console_script(tmp_path, monkeypatch):
    exe = tmp_path / "listn"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(exe), "ui"])
    assert autostart.launcher_argv() == [str(exe)]

    # A bare name is resolved through PATH.
    monkeypatch.setattr(sys, "argv", ["listn", "ui"])
    monkeypatch.setenv("PATH", str(tmp_path))
    assert autostart.launcher_argv() == [str(exe)]

    # `python -m listn` relaunches the same way: `env PYTHONPATH=` carries the
    # running package's own source root (the packaged venv is synced with
    # --no-install-project and holds only the dependencies, and a login
    # session has no PYTHONPATH — without the prefix the entry dies with
    # ModuleNotFoundError), and the login cwd never lands ahead of the real
    # package on sys.path (F2): -P on >= 3.11, the runpy shim on 3.10.
    main_py = tmp_path / "site" / "listn" / "__main__.py"
    main_py.parent.mkdir(parents=True)
    main_py.write_text("import sys\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(main_py), "ui"])
    prefix = [autostart.ENV_PROGRAM, f"PYTHONPATH={autostart.source_root()}"]
    if sys.version_info >= (3, 11):
        assert autostart.launcher_argv() == [
            *prefix, sys.executable, "-P", "-m", "listn"]
    else:
        assert autostart.launcher_argv() == [
            *prefix, sys.executable, "-c", autostart.SAFE_RUN_SHIM]

    # A non-listn argv[0] (pytest, an interpreter) never lands in Exec: the
    # listn command on PATH wins, and the bare command name is the last resort.
    other = tmp_path / "pytest"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    other.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(other), "-q"])
    assert autostart.launcher_argv() == [str(exe)]
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert autostart.launcher_argv() == ["listn"]


def test_launcher_argv_resolves_a_relative_invocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / "listn"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "argv", ["bin/listn", "ui"])
    assert autostart.launcher_argv() == [str(exe)]


# -------------------------------------------------------- the cwd is hostile


def test_safe_module_argv_gates_on_the_python_floor(monkeypatch):
    """F2: `-P` needs 3.11; the 3.10 floor gets the runpy shim instead.

    ``PYTHONSAFEPATH`` is not an option for the floor — it arrived in 3.11
    alongside ``-P``, so a 3.10 interpreter would silently ignore it.
    """
    monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
    assert autostart.safe_module_argv("/usr/bin/python3") == [
        "/usr/bin/python3", "-P", "-m", "listn"]

    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    assert autostart.safe_module_argv("/usr/bin/python3") == [
        "/usr/bin/python3", "-c", autostart.SAFE_RUN_SHIM]

    # The shim travels through two desktop-entry escaping layers, so it may
    # hold none of the characters either layer rewrites, and no line break
    # (which would split the Exec value in half).
    assert not set(autostart.SAFE_RUN_SHIM) & set('"\\`$%')
    assert "\n" not in autostart.SAFE_RUN_SHIM
    # It is a script, not a statement soup: it must at least compile.
    compile(autostart.SAFE_RUN_SHIM, "<safe-run-shim>", "exec")


def test_module_argv_carries_the_running_source_root_on_pythonpath():
    """The packaged venv never contains the project itself.

    ``uv sync --no-install-project`` (``packaging/arch/listn-launcher.sh``:
    a read-only ``/usr/share/listn`` cannot be built into a wheel) puts only
    the *dependencies* into ``~/.local/share/listn/env``, and the desktop
    session runs the login entry with no ``PYTHONPATH`` — so the entry must
    carry the root of the package that was running when the toggle flipped,
    the way the launcher exports it, or ``python -m listn`` cannot import
    ``listn`` at all at login.
    """
    root = Path(autostart.source_root())
    assert root.is_absolute() and root == root.resolve()   # stable: no cwd drift
    assert (root / "listn" / "__init__.py").is_file()      # importable from here alone
    # Derived from this module's own location — the package actually running.
    assert root == Path(autostart.__file__).resolve().parent.parent
    # env + PYTHONPATH lead; the safe invocation (F2) follows unchanged.
    for version, safe in (((3, 11, 0, "final", 0),
                           ["/venv/bin/python", "-P", "-m", "listn"]),
                          ((3, 10, 0, "final", 0),
                           ["/venv/bin/python", "-c", autostart.SAFE_RUN_SHIM])):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "version_info", version)
            assert autostart.module_argv("/venv/bin/python") == [
                autostart.ENV_PROGRAM, f"PYTHONPATH={autostart.source_root()}", *safe]


def test_rendered_exec_from_a_planted_home_prints_the_real_version(home, monkeypatch):
    """F2: the login Exec must ignore a planted ``$HOME/listn/`` in its cwd.

    The entry is rendered the way a packaged install renders it (argv[0] is
    the package's ``__main__.py``), tokenized back with the desktop-entry
    rules, and executed from a ``$HOME`` that contains a hostile ``listn/``
    package — the exact login-time setup. ``tray`` is swapped for
    ``--version`` so the test never starts a server; the launcher prefix it
    runs is byte-identical either way.
    """
    repo = Path(autostart.__file__).resolve().parents[1]
    main_py = repo / "listn" / "__main__.py"
    monkeypatch.setattr(sys, "argv", [str(main_py), "tray"])

    argv = decode_exec(autostart.render().split("Exec=", 1)[1].splitlines()[0])
    assert argv == [*autostart.launcher_argv(), "tray"]
    argv[-1] = "--version"

    planted = home / "listn"
    planted.mkdir()
    (planted / "__init__.py").write_text('print("PLANTED_PKG_LOADED_INSTEAD")\n',
                                         encoding="utf-8")
    (planted / "__main__.py").write_text('print("PLANTED_PKG_LOADED_INSTEAD")\n',
                                         encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(repo)}
    # Control: without the safe path the planted package wins — the F2 repro.
    control = subprocess.run([sys.executable, "-m", "listn", "--version"],
                             cwd=home, env=env, capture_output=True, text=True,
                             timeout=120)
    assert "PLANTED_PKG_LOADED_INSTEAD" in control.stdout, control.stdout

    run = subprocess.run(argv, cwd=home, env=env, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == f"listn {__version__}"
    assert "PLANTED_PKG_LOADED_INSTEAD" not in run.stdout + run.stderr


@pytest.mark.parametrize("floor", [(3, 11), (3, 10)],
                         ids=["safe-path-flag", "py310-shim"])
def test_rendered_packaged_exec_prints_the_real_version_in_a_clean_environment(
        home, monkeypatch, tmp_path, floor):
    """The packaged login entry must import listn with no session PYTHONPATH.

    A packaged install builds the per-user venv with ``uv sync
    --no-install-project``, so listn itself is never in that venv — only the
    launcher exports ``PYTHONPATH=/usr/share/listn``, and the desktop session
    runs the rendered entry with no environment of its own: the reported bug
    is ``ModuleNotFoundError: No module named 'listn'`` at login.

    The entry is rendered the way a packaged install renders it (the venv's
    interpreter, argv[0] the package's ``__main__.py``), tokenized back with
    the desktop-entry rules, and executed from a plain ``$HOME`` in an
    environment with ``PYTHONPATH`` removed — the login-time setup. Control:
    the entry as it was *before* this fix (same interpreter, no PYTHONPATH)
    fails with exactly that error; with the fix, the real ``listn
    <version>`` runs. Both shipped Exec shapes are exercised: ``-P`` (3.11+)
    and the 3.10 runpy shim.
    """
    real_version = tuple(sys.version_info[:2])
    if floor >= (3, 11) and real_version < (3, 11):
        pytest.skip("this interpreter is the 3.10 floor: it cannot run -P")

    # The packaged venv, bare: `--no-install-project` means the project (and
    # every wheel of it) never enters it, only the locked dependencies do —
    # a fresh venv with nothing in its site-packages models exactly that.
    env_dir = tmp_path / "packaged-env"
    made = subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                           str(env_dir)],
                          capture_output=True, text=True, timeout=300)
    assert made.returncode == 0, made.stderr
    packaged_python = env_dir / "bin" / "python"
    assert packaged_python.is_file()

    # Render as the packaged install does: its venv interpreter, started via
    # `python -m listn` (argv[0] is the package's __main__.py), gated the way
    # the real rendering interpreter would be.
    main_py = Path(autostart.__file__).resolve().parent / "__main__.py"
    monkeypatch.setattr(sys, "executable", str(packaged_python))
    monkeypatch.setattr(sys, "argv", [str(main_py), "tray"])
    monkeypatch.setattr(sys, "version_info", (*floor, 0, "final", 0))

    argv = decode_exec(autostart.render().split("Exec=", 1)[1].splitlines()[0])
    assert argv == [*autostart.launcher_argv(), "tray"]
    assert argv[:2] == [autostart.ENV_PROGRAM,
                        f"PYTHONPATH={autostart.source_root()}"]
    argv[-1] = "--version"

    # The login environment: no PYTHONPATH, plain $HOME, cwd = $HOME.
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    clean["HOME"] = str(home)

    # Control: the entry as it was before the fix — the same interpreter with
    # no PYTHONPATH — is exactly the reported failure.
    control = subprocess.run(
        [*autostart.safe_module_argv(str(packaged_python)), "--version"],
        cwd=home, env=clean, capture_output=True, text=True, timeout=120)
    assert control.returncode != 0, control.stdout
    # `python -m` reports it through runpy as ImportError, the shim as
    # ModuleNotFoundError; both carry the same wording as the bug report.
    assert "No module named" in control.stderr and "listn" in control.stderr, \
        control.stderr

    # The rendered entry runs in exactly that environment and prints the
    # real version of the package that was running when it was rendered.
    run = subprocess.run(argv, cwd=home, env=clean, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == f"listn {__version__}"


def test_the_rendered_module_entry_validates_as_a_desktop_entry(tmp_path, monkeypatch):
    """The -P/shim Exec must stay a legal desktop-entry command line."""
    if shutil.which("desktop-file-validate") is None:
        pytest.skip("desktop-file-utils not installed")
    main_py = tmp_path / "site" / "listn" / "__main__.py"
    main_py.parent.mkdir(parents=True)
    main_py.write_text("import sys\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(main_py), "tray"])

    entry = tmp_path / "listn.desktop"
    entry.write_text(autostart.render(), encoding="utf-8")
    result = subprocess.run(["desktop-file-validate", str(entry)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_rendered_module_entry_round_trips_through_glib(tmp_path, monkeypatch):
    """Real GLib must read the `env PYTHONPATH=…` Exec back argv-for-argv.

    The module form now leads with two extra tokens (`env`,
    `PYTHONPATH=<source root>`), and the `PYTHONPATH=` value travels through
    both desktop-entry escaping layers like every other argument: GLib's
    shell_parse_argv has to hand back exactly the argv the entry encodes, or
    GIO would exec something other than what was written.
    """
    interpreters = [c for c in (sys.executable, "/usr/bin/python3") if has_gi(c)]
    if not interpreters:
        pytest.skip("PyGObject not installed for any interpreter")
    main_py = tmp_path / "site" / "listn" / "__main__.py"
    main_py.parent.mkdir(parents=True)
    main_py.write_text("import sys\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(main_py), "tray"])

    entry = tmp_path / "listn.desktop"
    entry.write_text(autostart.render(), encoding="utf-8")
    result = subprocess.run([interpreters[0], "-c", GLIB_PROBE, str(entry)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert "error" not in data, data["error"]          # key-file string layer is valid
    assert data["argv"] == [*autostart.launcher_argv(), "tray"], data
    assert data["cmdline"] is not None, data           # GIO still sees a command
    assert data["cmdline"] == data["exec"], data


# ------------------------------------------------------------------ the UI


def test_linux_settings_shows_the_toggle(client):
    """The toggle is wired up on Linux now, so the UI must not hide it."""
    js = client.get("/static/app.js").text
    assert '["#offer-meetings", "#auto-record", "#launch-at-login"]' not in js
    assert js.count('["#offer-meetings", "#auto-record"]') == 2
    html = client.get("/").text
    assert 'data-setting="launch_at_login"' in html
