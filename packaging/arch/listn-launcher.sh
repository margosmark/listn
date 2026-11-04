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

PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}" exec "$env/bin/python" -m listn "$@"
