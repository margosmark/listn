#!/bin/sh
# Arch package launcher for listn (installed as /usr/bin/listn).
#
# The code lives in /usr/share/listn (read-only, owned by pacman). The Python
# environment is built per user by uv from the shipped uv.lock, so running
# listn writes nothing outside $HOME - no root-owned venv, no root-owned state.
#
# The tray extra (dbus-next) is only needed by `listn tray`, so that one
# command syncs it in; every other command stays on the base dependency set.

project=/usr/share/listn
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${XDG_DATA_HOME:-$HOME/.local/share}/listn/env}"

if [ "${1:-}" = "tray" ]; then
    exec uv run --project "$project" --frozen --extra tray listn "$@"
fi
exec uv run --project "$project" --frozen listn "$@"
