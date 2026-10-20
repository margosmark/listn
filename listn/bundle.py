"""Assemble a Python tree that travels inside listn.app.

The menu bar app then runs ``python -m listn`` from that tree. It does not
need the repo checkout or uv. Model weights stay in ``~/.cache/huggingface``
and are not copied.

``pyvenv.cfg`` records the base interpreter as ``home``. That directory is
copied (the binary is dereferenced, so a symlink out of the tree is not what
the app executes). Site-packages are copied from the venv, and the ``listn``
package is copied from the project so an editable install does not point
back at the checkout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SKIP_DIR_NAMES = {"__pycache__", "test", "tests", "ensurepip"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def read_base_prefix(venv: Path) -> Path:
    cfg = Path(venv) / "pyvenv.cfg"
    for line in cfg.read_text().splitlines():
        if line.startswith("home ="):
            home = line.split("=", 1)[1].strip()
            if home:
                return Path(home)
    raise ValueError(f"{cfg} has no home entry, so the base interpreter is unknown.")


def launcher_text() -> str:
    return """#!/bin/bash
# Runs the interpreter shipped inside this app. No repo checkout and no uv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHOME="$ROOT/prefix"
export PYTHONPATH="$ROOT/site-packages"
exec "$ROOT/prefix/bin/python3" -m listn "$@"
"""


def _keep(relative: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return False
    return relative.suffix not in SKIP_SUFFIXES


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = src.resolve() if src.is_symlink() else src
    shutil.copy2(target, dest)


def _copy_tree(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        if not _keep(relative):
            continue
        target = dest / relative
        if path.is_dir() and not path.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() or path.is_symlink():
            _copy_file(path, target)


def directory_bytes(path: Path) -> int:
    total = 0
    for child in Path(path).rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_size(nbytes: int) -> str:
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.0f} MB"
    return f"{nbytes / 1_000:.0f} KB"


def assemble(venv: Path, project: Path, dest: Path) -> dict:
    """Copy the interpreter and listn into ``dest``. Returns a size report."""
    venv = Path(venv)
    project = Path(project)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    prefix = read_base_prefix(venv)
    prefix_dest = dest / "prefix"
    bin_src = prefix / "bin"
    if not bin_src.is_dir():
        raise FileNotFoundError(f"No interpreter bin directory at {bin_src}.")
    for name in ("python3", "python"):
        candidate = bin_src / name
        if candidate.exists():
            _copy_file(candidate, prefix_dest / "bin" / "python3")
            break
    else:
        raise FileNotFoundError(f"No python3 in {bin_src}.")
    lib = prefix / "lib"
    _copy_tree(lib, prefix_dest / "lib")
    framework = prefix / "Python"
    if framework.exists():
        _copy_file(framework, prefix_dest / "Python")

    site_dest = dest / "site-packages"
    site_dest.mkdir(parents=True)
    lib_root = venv / "lib"
    if lib_root.is_dir():
        for site in lib_root.glob("python*/site-packages"):
            for child in site.iterdir():
                if child.suffix in {".pth", ".egg-link"} or child.name.endswith(".egg-link"):
                    continue
                target = site_dest / child.name
                if child.is_dir():
                    _copy_tree(child, target)
                elif child.is_file():
                    _copy_file(child, target)
    package = project / "listn"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"No listn package at {package}.")
    _copy_tree(package, site_dest / "listn")

    launcher = dest / "bin" / "listn"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(launcher_text())
    launcher.chmod(0o755)
    size = directory_bytes(dest)
    note = (
        f"{format_size(size)}\n"
        "This is the app and its Python environment only. "
        "Whisper and the notes model stay in ~/.cache/huggingface and are not included.\n"
    )
    (dest / "SIZE.txt").write_text(note)
    return {"bytes": size, "label": format_size(size), "launcher": str(launcher)}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Copy a private Python environment into listn.app")
    parser.add_argument("--venv", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    report = assemble(Path(args.venv), Path(args.project), Path(args.dest))
    print(f"Bundled Python environment: {report['label']} (models not included)")


if __name__ == "__main__":
    main()
