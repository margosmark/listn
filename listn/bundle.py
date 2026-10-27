"""Assemble a Python tree that travels inside listn.app.

The menu bar app then runs ``python -m listn`` from that tree. It does not
need the repo checkout or uv. Model weights stay in ``~/.cache/huggingface``
and are not copied.

The base interpreter is found by resolving ``<venv>/bin/python`` through its
symlinks (uv's ``cpython-3.13-…`` directory is itself a symlink to the
patch release). Its prefix (``bin`` and ``lib``) is copied with symlinks
dereferenced, the venv's site-packages go into that prefix's own
``site-packages`` so ``.pth`` files still run, and ``listn`` is copied from
the project so an editable install does not point back at the checkout.

Every Mach-O file is listed in ``macho.txt`` so the build script can sign
each one before it seals the app.
"""

from __future__ import annotations

import compileall
import os
import re
import shutil
from pathlib import Path

SKIP_DIR_NAMES = {"__pycache__", "test", "tests", "ensurepip", "idlelib", "turtledemo"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
MACHO_MAGIC = {
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
}
_VERSION_DIR = re.compile(r"^python3\.\d+$")


def read_home(venv: Path) -> Path | None:
    cfg = Path(venv) / "pyvenv.cfg"
    try:
        lines = cfg.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            return Path(value.strip())
    return None


def base_interpreter(venv: Path) -> Path:
    """The real interpreter binary behind the venv, with every symlink resolved."""
    venv = Path(venv)
    for name in ("python3", "python"):
        candidate = venv / "bin" / name
        if candidate.exists():
            real = candidate.resolve()
            if real.is_file() and real.parent.name == "bin":
                return real
    home = read_home(venv)
    if home is not None:
        home = home.resolve()
        bindir = home if home.name == "bin" else home / "bin"
        for name in ("python3", "python"):
            candidate = bindir / name
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not find the interpreter behind {venv}. Run `uv sync` first.")


def base_prefix(venv: Path) -> Path:
    return base_interpreter(venv).parent.parent


def stdlib_dir_name(prefix: Path) -> str:
    lib = Path(prefix) / "lib"
    found = [p.name for p in lib.iterdir() if p.is_dir() and _VERSION_DIR.match(p.name)] if lib.is_dir() else []
    names = sorted(found, key=lambda name: int(name.split(".")[1]))
    if not names:
        raise FileNotFoundError(f"No lib/python3.X under {prefix}.")
    return names[-1]


def launcher_text() -> str:
    return """#!/bin/bash
# Runs the interpreter shipped inside this app. No repo checkout and no uv.
# Bytecode is compiled at build time; writing it here would break the signature.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHOME="$ROOT/prefix"
unset PYTHONPATH
exec "$ROOT/prefix/bin/python3" -m listn "$@"
"""


def _keep(relative: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return False
    return relative.suffix not in SKIP_SUFFIXES


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = src.resolve() if src.is_symlink() else src
    if target.is_file():
        shutil.copy2(target, dest)


def _copy_tree(src: Path, dest: Path, skip_top: set[str] | None = None) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        if skip_top and relative.parts and relative.parts[0] in skip_top:
            continue
        if not _keep(relative):
            continue
        target = dest / relative
        if path.is_dir() and not path.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _copy_file(path, target)


def _editable_hook(path: Path, project: Path) -> bool:
    name = path.name.lower()
    if "editable" in name or name.startswith("_listn") or name.startswith("listn"):
        return True
    if path.suffix == ".pth":
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return False
        return str(project) in text
    return False


def is_macho(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) in MACHO_MAGIC
    except OSError:
        return False


def directory_bytes(path: Path) -> int:
    total = 0
    for child in Path(path).rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def format_size(nbytes: int) -> str:
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.0f} MB"
    return f"{nbytes / 1_000:.0f} KB"


def assemble(venv: Path, project: Path, dest: Path, compile_bytecode: bool = True) -> dict:
    """Copy the interpreter, the venv's packages, and listn into ``dest``."""
    venv = Path(venv)
    project = Path(project).resolve()
    dest = Path(dest)
    package = project / "listn"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"No listn package at {package}.")
    interpreter = base_interpreter(venv)
    prefix = interpreter.parent.parent
    version_dir = stdlib_dir_name(prefix)
    if dest.exists():
        shutil.rmtree(dest)
    prefix_dest = dest / "prefix"

    _copy_file(interpreter, prefix_dest / "bin" / "python3")
    (prefix_dest / "bin" / "python3").chmod(0o755)
    stdlib = prefix / "lib" / version_dir
    lib_dest = prefix_dest / "lib"
    _copy_tree(prefix / "lib", lib_dest, skip_top={version_dir, "pkgconfig"})
    _copy_tree(stdlib, lib_dest / version_dir, skip_top={"site-packages", "config-" + version_dir[6:] + "-darwin"})

    site_dest = lib_dest / version_dir / "site-packages"
    site_dest.mkdir(parents=True, exist_ok=True)
    for site in sorted((venv / "lib").glob("python3.*/site-packages")):
        for child in site.iterdir():
            if _editable_hook(child, project):
                continue
            target = site_dest / child.name
            if child.is_dir():
                _copy_tree(child, target)
            elif child.is_file():
                _copy_file(child, target)
    if (site_dest / "listn").exists():
        shutil.rmtree(site_dest / "listn")
    _copy_tree(package, site_dest / "listn")

    launcher = dest / "bin" / "listn"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(launcher_text())
    launcher.chmod(0o755)

    if compile_bytecode:
        compileall.compile_dir(str(lib_dest / version_dir), quiet=2, workers=0)

    machos = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and not p.is_symlink() and is_macho(p))
    (dest / "macho.txt").write_text("".join(f"{item}\n" for item in machos))

    size = directory_bytes(dest)
    (dest / "SIZE.txt").write_text(
        f"{format_size(size)}\n"
        "The app's Python environment only. Whisper, the notes model and the pyannote "
        "speaker models stay in ~/.cache/huggingface and are not included.\n"
    )
    return {"bytes": size, "label": format_size(size), "launcher": str(launcher), "macho": machos,
            "interpreter": str(interpreter)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Copy a private Python environment into listn.app")
    parser.add_argument("--venv", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    try:
        report = assemble(Path(args.venv), Path(args.project), Path(args.dest))
    except (OSError, ValueError) as exc:
        print(f"Could not bundle Python: {exc}", file=sys.stderr)
        return 1
    print(f"Interpreter: {report['interpreter']}")
    print(f"Bundled Python environment: {report['label']}, {len(report['macho'])} Mach-O files (models not included)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
