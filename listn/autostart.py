"""Launch at login on Linux: the Settings toggle's XDG autostart entry.

``PUT /api/settings {"launch_at_login": true}`` writes
``$XDG_CONFIG_HOME/autostart/listn.desktop`` (default
``~/.config/autostart/listn.desktop``) so the tray starts at login; ``false``
removes that entry again. The file carries a ``# written by listn`` marker,
so switching off only ever deletes a file listn wrote itself — an entry you
wrote by hand that merely occupies the path is left alone, and reads back as
"off".

Everything here is local and inert: no network, no subprocess, nothing but
files under ``$HOME``. The Exec line is the running console script plus
``tray``, quoted with the desktop-entry rules (no shell ever parses it) and
carrying no token, transcript or other private data; the ``python -m listn``
form re-launches with the cwd kept off ``sys.path`` (``-P`` / the 3.10 shim),
so a stray ``~/listn/`` cannot shadow the packaged code at login, and it
carries the running package's own source root on ``PYTHONPATH``: the
packaged install syncs the venv with ``--no-install-project`` (the project
itself never enters it — only the launcher exports PYTHONPATH), and a login
session runs the entry with no PYTHONPATH of its own, so without that prefix
``import listn`` dies with ``ModuleNotFoundError`` and no tray ever starts.

macOS is out of scope on purpose: the native menu bar app owns this toggle
there (AGENTS.md — preserve macOS behaviour), so every entry point here is a
no-op unless ``sys.platform`` starts with ``linux``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

#: Line prefix that marks the entry as listn's own; deleting requires it.
MARKER = "# written by listn"

ENTRY_NAME = "listn.desktop"

#: ``python -P`` (Python >= 3.11): never prepend the launching directory to
#: ``sys.path``. ``python -m`` otherwise puts cwd *first*, ahead of
#: ``PYTHONPATH``, so a planted ``listn/`` in it would run instead of the
#: installed package (F2 — the CWD import hijack).
SAFE_PATH_FLAG = "-P"

#: The floor is Python 3.11 (``pyproject.toml`` ``requires-python``); anything
#: below it (a system 3.10, say) has neither ``-P`` nor ``PYTHONSAFEPATH`` —
#: both arrived in 3.11 — and runs this ``python -c`` shim instead: drop every ``sys.path`` entry that
#: resolves to the cwd, then run ``listn`` exactly as ``-m`` would. It holds
#: no ``"``, ``\``, ``$``, backtick or ``%``, so both desktop-entry escaping
#: layers pass it through as one argument. Keep it byte-identical to the
#: fallback line in ``packaging/arch/listn-launcher.sh`` (a test compares).
SAFE_RUN_SHIM = (
    "import os,runpy,sys;"
    "sys.path[:]=[p for p in sys.path if os.path.realpath(p or os.curdir)!=os.getcwd()];"
    "runpy.run_module('listn',run_name='__main__',alter_sys=True)"
)

#: How the module form sets ``PYTHONPATH`` for the entry. Desktop-entry
#: ``Exec`` has no shell (nothing ever parses it), so ``VAR=value cmd`` is
#: not an assignment but a bogus program name — ``env`` is the portable
#: program that performs it. Absolute, so it is never resolved through a
#: session ``PATH`` listn does not choose at login; coreutils ships it in
#: ``/usr/bin`` on every Linux desktop, and only Linux ever writes this entry.
ENV_PROGRAM = "/usr/bin/env"


class AutostartError(RuntimeError):
    """The autostart entry cannot be written or removed safely."""


def supported() -> bool:
    """True where listn applies this toggle itself (Linux, never macOS)."""
    return sys.platform.startswith("linux")


def _within(path: Path, home: Path) -> bool:
    """Lexical containment, so a symlinked ``~/.config`` stays legal."""
    try:
        path.relative_to(home)
    except ValueError:
        return False
    return True


def autostart_file() -> Path:
    """Absolute path of the entry, always lexically inside ``$HOME``."""
    if not supported():
        raise AutostartError("launch at login is managed natively on this platform")
    home_raw = os.environ.get("HOME", "").strip()
    if not home_raw:
        raise AutostartError("HOME is not set, so the autostart entry has nowhere to go")
    home = Path(os.path.normpath(os.path.expanduser(home_raw)))
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        base = Path(os.path.normpath(os.path.expanduser(xdg)))
        if not base.is_absolute():
            base = home / ".config"   # a relative XDG_CONFIG_HOME is ignored
    else:
        base = home / ".config"
    if not _within(base, home):
        raise AutostartError(
            f"XDG_CONFIG_HOME points at {base}, outside $HOME — "
            "listn only writes inside your home directory")
    return base / "autostart" / ENTRY_NAME


def quote_exec_arg(arg: str) -> str:
    """Quote one argument for the ``Exec`` line *as written to the file*.

    Two layers stack, and the outer one is applied first when the entry is
    read: GLib's key-file *string* layer (only ``\\s \\n \\t \\r \\\\`` are
    escapes — any other backslash makes the whole value un-interpretable,
    so the entry loads with no command at all), then the ``Exec`` layer
    (``"`` grouping, backslash escapes for ``\\ " $ ` `` and ``%%`` for a
    literal percent). So this quotes for ``Exec`` and then doubles every
    backslash for the string layer; GLib undoes exactly those two, in
    reverse. No shell parses either layer, so nothing here is shell
    interpolation.
    """
    if arg and not any(c in arg for c in ' \t"\\`$%'):
        return arg              # no backslash in the argument: the layers are a no-op
    escaped = (arg.replace("\\", "\\\\").replace('"', '\\"')
               .replace("$", "\\$").replace("`", "\\`").replace("%", "%%"))
    return f'"{escaped}"'.replace("\\", "\\\\")   # key-file string layer


def _python() -> str | None:
    exe = sys.executable or ""
    return exe if exe and Path(exe).is_file() else None


def safe_module_argv(python: str) -> list[str]:
    """``python`` plus how to run ``listn`` with the cwd kept off ``sys.path``.

    The interpreter that renders this is the same one that will run the entry
    at login, so testing ``sys.version_info`` here is the right gate: ``-P``
    where it exists, the ``runpy`` shim below 3.11.
    """
    if sys.version_info >= (3, 11):
        return [python, SAFE_PATH_FLAG, "-m", "listn"]
    return [python, "-c", SAFE_RUN_SHIM]


def source_root() -> str:
    """The directory ``import listn`` must search to find what is running now.

    Derived from this module's own file — the package that is executing as
    the toggle flips — never from the cwd or from ``PYTHONPATH``: whatever
    was running then is exactly what the login entry must start again
    (``/usr/share/listn`` on a packaged install, the checkout in development,
    the venv's ``site-packages`` when the project was installed into it).
    """
    return str(Path(__file__).resolve().parent.parent)


def module_argv(python: str) -> list[str]:
    """Full ``Exec`` argv for the ``python -m listn`` form.

    Leads with ``env PYTHONPATH=<source root>`` because the packaged venv
    only ever holds the *dependencies*: the launcher builds it with
    ``uv sync --no-install-project`` (a read-only ``/usr/share/listn`` cannot
    be built into a wheel) and exports ``PYTHONPATH`` itself, while the
    desktop session runs the entry with no environment of its own — without
    the prefix, ``python -m listn`` at login raises ``ModuleNotFoundError``
    and the tray never starts. The variable is set outright, not appended
    to: the entry carries exactly the tree that was running, and nothing
    inherited from a session listn does not control. Then
    :func:`safe_module_argv` (F2: the login cwd stays off ``sys.path``).
    """
    return [ENV_PROGRAM, f"PYTHONPATH={source_root()}", *safe_module_argv(python)]


def launcher_argv() -> list[str]:
    """How to start the tray again: the console script running right now.

    Covers the packaged launcher (``/usr/bin/listn``), a source checkout
    (``uv run listn`` → ``.venv/bin/listn``) and ``python -m listn`` — the
    ``-m`` form is re-rendered through :func:`module_argv` (the running
    package's source root on PYTHONPATH, so a venv that was synced with
    ``--no-install-project`` can still import it at login, plus the cwd kept
    off ``sys.path`` — F2); falls back to the bare ``listn`` command when
    nothing better is known.
    """
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    file: Path | None = None
    if argv0:
        p = Path(argv0).expanduser()
        if p.is_absolute():
            file = p if p.is_file() else None
        elif os.sep in argv0 or (os.altsep and os.altsep in argv0):
            # A relative invocation: the login session's cwd won't be ours.
            resolved = p.resolve()
            file = resolved if resolved.is_file() else None
    if file is not None and file.name == "__main__.py" and "listn" in file.parts:
        # `python -m listn …` puts the package's __main__.py in argv[0].
        py = _python()
        if py:
            return module_argv(py)
    if file is not None and file.name.startswith("listn"):
        return [str(file)]
    if len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2].split(".")[0] == "listn":
        py = _python()
        if py:
            return module_argv(py)
    found = shutil.which("listn")
    if found:
        return [found]
    return ["listn"]


def render() -> str:
    """The full desktop entry for the launcher that is running right now."""
    exec_line = " ".join(quote_exec_arg(a) for a in [*launcher_argv(), "tray"])
    if any(c in exec_line for c in "\n\r"):
        raise AutostartError("the launcher path contains a line break; refusing to write a broken entry")
    return "\n".join([
        f"{MARKER} — Settings → Launch at login manages this file; delete it to opt out.",
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=listn",
        "Comment=Start the listn tray and its engine at login",
        f"Exec={exec_line}",
        "Icon=listn",
        "Terminal=false",
        "NoDisplay=true",
        "StartupNotify=false",
        "X-GNOME-Autostart-enabled=true",
        "",
    ])


def _marked(text: str) -> bool:
    return any(line.startswith(MARKER) for line in text.splitlines())


def _read(path: Path) -> str:
    """Entry contents, strict — an unreadable file is never assumed to be ours."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AutostartError(f"{path} cannot be read ({exc}); listn will not touch it") from exc


def is_enabled() -> bool:
    """Disk truth: is there a listn-owned entry right now?"""
    if not supported():
        return False
    try:
        path = autostart_file()
    except AutostartError:
        return False
    if path.is_symlink() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _marked(text)


def set_enabled(enabled: bool) -> None:
    """Create (``True``) or remove (``False``) the entry, idempotently.

    Raises :class:`AutostartError` rather than overwrite or delete anything
    listn cannot prove it owns.
    """
    if not supported():
        return                      # macOS: the native app owns this toggle
    path = autostart_file()
    if enabled:
        exists = path.exists() or path.is_symlink()   # is_symlink catches a dangling link
        content = render()
        if exists:
            if path.is_symlink() or not path.is_file():
                raise AutostartError(
                    f"{path} exists and is not a regular file; listn will not touch it")
            existing = _read(path)
            if not _marked(existing):
                raise AutostartError(
                    f"{path} was not written by listn — move or delete it yourself, "
                    "then flip the toggle again")
            if existing == content:
                return              # idempotent: already exactly this file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            try:
                # O_EXCL|O_NOFOLLOW: never open through whatever sits at the
                # fixed temp name. A stale temp — or a symlink planted there —
                # is unlinked first, which removes that link itself and never
                # its target; if anything reappears in the gap, O_EXCL fails
                # the write instead of following it.
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    os.fchmod(fh.fileno(), 0o644)   # 0644 whatever the umask, no path followed
                    fh.write(content)
                os.replace(tmp, path)
            finally:
                if tmp.is_symlink() or tmp.exists():
                    tmp.unlink()
        except OSError as exc:
            raise AutostartError(f"could not write {path} ({exc})") from exc
        return
    if path.is_symlink() or not path.exists():
        return                      # nothing there (or something we must not follow)
    if not path.is_file():
        return                      # a directory at the path is not our entry
    try:
        existing = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return                      # can't confirm the marker → never delete
    if not _marked(existing):
        return                      # foreign entry: left exactly as it was
    path.unlink()
