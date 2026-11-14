#!/bin/sh
# Arch package launcher for listn (installed as /usr/bin/listn).
#
# The code lives in /usr/share/listn (read-only, owned by pacman), and building
# a wheel from a read-only tree fails: setuptools must create listn.egg-info
# next to pyproject.toml. So the launcher never installs the project. `uv sync
# --no-install-project` brings the locked dependencies (--no-dev) into a
# per-user virtual environment, and Python imports the module straight out of
# /usr/share/listn via PYTHONPATH. Running listn writes nothing outside $HOME:
# no root-owned venv, no root-owned state.
#
# The tray extra (dbus-next) is only needed by `listn tray`, so only that
# command syncs it in; every other command stays on the base dependency set.

set -eu

project=/usr/share/listn
env="${UV_PROJECT_ENVIRONMENT:-${XDG_DATA_HOME:-$HOME/.local/share}/listn/env}"
export UV_PROJECT_ENVIRONMENT="$env"

if [ "${1:-}" = "tray" ]; then
    uv sync --project "$project" --frozen --no-install-project --no-dev --extra tray
else
    uv sync --project "$project" --frozen --no-install-project --no-dev
fi

# `python -m` puts cwd first on sys.path, ahead of PYTHONPATH: a planted
# listn/ in whatever directory this runs from would execute instead of the
# packaged code (F2, the CWD import hijack). -P drops cwd — but it only exists
# on Python >= 3.11, and anything older than the 3.11 floor (pyproject.toml)
# has neither -P nor PYTHONSAFEPATH (both arrived in 3.11). So pre-3.11 runs the same
# runpy shim listn/autostart.py renders into its Exec line: strip every
# sys.path entry that resolves to the cwd, then run the module as -m would.
if "$env/bin/python" -c 'import sys;sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
    PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}" exec "$env/bin/python" -P -m listn "$@"
fi
PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}" exec "$env/bin/python" -c "import os,runpy,sys;sys.path[:]=[p for p in sys.path if os.path.realpath(p or os.curdir)!=os.getcwd()];runpy.run_module('listn',run_name='__main__',alter_sys=True)" "$@"
