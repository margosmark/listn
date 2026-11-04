#!/bin/sh
# listn desktop integration for Linux (KDE and other freedesktop sessions):
# a menu entry, an optional XDG autostart entry for the tray, and the icon.
#
# User scope only. Everything is written under $XDG_DATA_HOME and
# $XDG_CONFIG_HOME (their ~/.local/share and ~/.config defaults); nothing
# needs root and nothing is written outside $HOME. The macOS equivalent is
# scripts/listn-manage.sh - do not run either script on the other platform.
#
# Usage:
#   scripts/listn-desktop.sh install [--autostart] [--dry-run]
#   scripts/listn-desktop.sh uninstall [--dry-run]
#   scripts/listn-desktop.sh status
#
#   install           menu entry + icon (KDE app menu shows "listn")
#   --autostart       also start the tray at login (install-time choice; the
#                     tray has no launch-at-login control of its own)
#   uninstall         remove both, whether or not --autostart was used
#   status            show what is installed and which command each entry runs
#   --dry-run         print what would change without writing anything
#
# What the entries run: $LISTN_CMD if set, otherwise this checkout's
# .venv/bin/listn, otherwise `listn` on PATH. A $LISTN_CMD containing spaces
# is used verbatim, so quote it yourself when it needs quoting.

set -eu

usage() {
    cat <<'EOF'
usage: listn-desktop.sh <install [--autostart] [--dry-run] | uninstall [--dry-run] | status>

  install           install the app menu entry and icon (user scope)
  --autostart       with install: also start the tray at login
  uninstall         remove the menu entry, icon and autostart entry
  status            show what is installed and what it launches
  --dry-run         print what would change without writing

Environment:
  LISTN_CMD         command the entries launch (default: this checkout's
                    .venv/bin/listn, then `listn` on PATH)
  XDG_DATA_HOME     default ~/.local/share
  XDG_CONFIG_HOME   default ~/.config
EOF
}

if [ "$(uname -s)" = "Darwin" ]; then
    echo "listn-desktop.sh is the Linux installer; on macOS use scripts/listn-manage.sh." >&2
    exit 2
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(CDPATH='' cd -- "$script_dir/.." && pwd -P)

apps_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons_dir="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/128x128/apps"
autostart_dir="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"

desktop_src="$repo_dir/packaging/listn.desktop"
autostart_src="$repo_dir/packaging/listn-tray.desktop"
icon_src="$repo_dir/macos/Resources/AppIcon.iconset/icon_128x128.png"

menu_entry="$apps_dir/listn.desktop"
icon_entry="$icons_dir/listn.png"
autostart_entry="$autostart_dir/listn-tray.desktop"

action="${1:-}"
if [ -z "$action" ]; then
    usage >&2
    exit 1
fi
shift

want_autostart=no
dry_run=no
for arg in "$@"; do
    case "$arg" in
        --autostart) want_autostart=yes ;;
        --dry-run) dry_run=yes ;;
        -h|--help) usage; exit 0 ;;
        *) echo "listn-desktop.sh: unknown option: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

run() {
    # run <description> <command...>
    if [ "$dry_run" = yes ]; then
        echo "dry-run: $*"
    else
        "$@"
    fi
}

resolve_cmd() {
    if [ -n "${LISTN_CMD:-}" ]; then
        printf '%s' "$LISTN_CMD"
        return 0
    fi
    if [ -x "$repo_dir/.venv/bin/listn" ]; then
        printf '%s' "$repo_dir/.venv/bin/listn"
        return 0
    fi
    if command -v listn >/dev/null 2>&1; then
        command -v listn
        return 0
    fi
    return 1
}

render() {
    # render <template> <command> <subcommand>: keep the template byte for
    # byte except Exec=, which points at the resolved listn command. A single
    # path is quoted; a command with spaces is passed through verbatim.
    template="$1"
    bin="$2"
    sub="$3"
    case "$bin" in
        *' '*) exec_line="$bin $sub" ;;
        *) exec_line="\"$bin\" $sub" ;;
    esac
    awk -v repl="Exec=$exec_line" '/^Exec=/ { print repl; next } { print }' "$template"
}

refresh_databases() {
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$apps_dir" >/dev/null 2>&1 || true
    fi
}

install_entries() {
    if ! bin=$(resolve_cmd); then
        cat >&2 <<EOF
listn-desktop.sh: no listn command found.
Run \`uv sync\` in this checkout, or set LISTN_CMD, for example:
  LISTN_CMD="\$PWD/.venv/bin/listn" scripts/listn-desktop.sh install
EOF
        return 1
    fi
    for src in "$desktop_src" "$autostart_src" "$icon_src"; do
        if [ ! -f "$src" ]; then
            echo "listn-desktop.sh: missing $src" >&2
            return 1
        fi
    done

    echo "listn command: $bin"
    run mkdir -p "$apps_dir" "$icons_dir"
    if [ "$dry_run" = yes ]; then
        echo "dry-run: write $menu_entry (Exec=\"$bin\" ui)"
        echo "dry-run: copy $icon_src -> $icon_entry"
    else
        render "$desktop_src" "$bin" ui > "$menu_entry"
        cp "$icon_src" "$icon_entry"
        echo "installed: $menu_entry"
        echo "installed: $icon_entry"
    fi

    if [ "$want_autostart" = yes ]; then
        run mkdir -p "$autostart_dir"
        if [ "$dry_run" = yes ]; then
            echo "dry-run: write $autostart_entry (Exec=\"$bin\" tray)"
        else
            render "$autostart_src" "$bin" tray > "$autostart_entry"
            echo "installed: $autostart_entry (tray starts at login)"
        fi
    fi
    refresh_databases
}

uninstall_entries() {
    for target in "$menu_entry" "$icon_entry" "$autostart_entry"; do
        if [ -e "$target" ]; then
            run rm -f -- "$target"
            [ "$dry_run" = yes ] || echo "removed: $target"
        else
            echo "absent:  $target"
        fi
    done
    [ "$dry_run" = yes ] || refresh_databases
}

status_entries() {
    for target in "$menu_entry" "$icon_entry" "$autostart_entry"; do
        if [ -f "$target" ]; then
            exec_line=$(awk '/^Exec=/ { print; exit }' "$target")
            echo "installed: $target${exec_line:+  [$exec_line]}"
        else
            echo "absent:    $target"
        fi
    done
    if [ -f "$autostart_entry" ]; then
        echo "tray starts at login (install-time choice, not a tray toggle)"
    else
        echo "tray does not start at login (enable with: install --autostart)"
    fi
}

case "$action" in
    install) install_entries ;;
    uninstall) uninstall_entries ;;
    status) status_entries ;;
    -h|--help|help) usage ;;
    *)
        echo "listn-desktop.sh: unknown command: $action" >&2
        usage >&2
        exit 1
        ;;
esac
