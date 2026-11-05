#!/bin/sh
# Isolated-HOME functional lifecycle test for scripts/listn-desktop.sh.
#
# Runs the real installer end to end against a throwaway $HOME: install,
# login opt-in, re-install (upgrade), status, uninstall, dry-run, the macOS
# guard and the unknown-command error - then proves uninstall removed only
# listn's own files and left user data (transcripts, settings, model cache,
# environment) and other applications' files alone. Nothing is written
# outside the throwaway directory and no system directory changes.
#
# This is a shell-level lifecycle test, NOT a Plasma login test: it does not
# start plasmashell, show the app menu, display the tray panel or log a
# session in. Those remain manual acceptance (docs/OPERATIONS.md).
#
# Usage: sh tests/shell/test_desktop_lifecycle.sh
# Exit: 0 when every check passes, 1 otherwise. Output is TAP-like.

set -u

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd -P)
installer="$repo_root/scripts/listn-desktop.sh"

n=0
failed=0
ok()  { n=$((n + 1)); printf 'ok %d - %s\n' "$n" "$1"; }
bad() { n=$((n + 1)); failed=$((failed + 1)); printf 'not ok %d - %s\n' "$n" "$1"; }

assert_exists() { # <path> <description>
    if [ -e "$1" ]; then ok "$2"; else bad "$2 (missing: $1)"; fi
}
assert_absent() { # <path> <description>
    if [ ! -e "$1" ]; then ok "$2"; else bad "$2 (unexpectedly present: $1)"; fi
}
assert_grep() { # <fixed-string> <file> <description>
    if [ -f "$2" ] && grep -qF "$1" "$2"; then ok "$3"
    else bad "$3 (pattern '$1' not found in $2)"; fi
}
assert_eq() { # <got> <want> <description>
    if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (got '$1', want '$2')"; fi
}
assert_rc() { # <got> <want> <description>
    if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (exit $1, want $2)"; fi
}

if [ ! -f "$installer" ]; then
    echo "not ok - installer not found: $installer" >&2
    exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/listn-desktop-test.XXXXXX")
# Invoked indirectly through the EXIT trap below.
# shellcheck disable=SC2317
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

HOME_DIR="$WORK/home"
BIN_DIR="$WORK/bin"
mkdir -p "$HOME_DIR" "$BIN_DIR"

# Two distinct fake commands so an upgrade that refreshes entries is visible.
CMD_A="$BIN_DIR/listn-a"
CMD_B="$BIN_DIR/listn-b"
for cmd in "$CMD_A" "$CMD_B"; do
    printf '#!/bin/sh\nexit 0\n' > "$cmd"
    chmod 755 "$cmd"
done

# System directories the installer must never touch (read-only snapshot).
sys_snapshot() {
    ls -1 /etc/xdg/autostart 2>/dev/null
    ls -1 /usr/share/applications 2>/dev/null
    ls -1 /usr/share/icons/hicolor/128x128/apps 2>/dev/null
}
sys_before=$(sys_snapshot)

# Run the installer in the isolated home. XDG_* are unset so the defaults
# land inside $HOME; LISTN_CMD pins the command the entries render.
run_installer() { # <listn-cmd> <args...>
    cmd=$1; shift
    env -u XDG_DATA_HOME -u XDG_CONFIG_HOME \
        HOME="$HOME_DIR" LISTN_CMD="$cmd" \
        sh "$installer" "$@"
}

apps="$HOME_DIR/.local/share/applications"
icons="$HOME_DIR/.local/share/icons/hicolor/128x128/apps"
autostart="$HOME_DIR/.config/autostart"
menu="$apps/listn.desktop"
icon="$icons/listn.png"
login="$autostart/listn-tray.desktop"

# ---------------------------------------------------- 1. plain install
out=$WORK/install.log
run_installer "$CMD_A" install > "$out" 2>&1
assert_rc "$?" 0 "install succeeds"
assert_exists "$menu" "install writes the menu entry (user scope)"
assert_exists "$icon" "install writes the icon (user scope)"
assert_absent "$login" "plain install does NOT create a login entry (opt-in only)"
assert_grep "Exec=\"$CMD_A\" ui" "$menu" "menu entry launches 'listn ui' via LISTN_CMD"

status_out=$WORK/status1.log
run_installer "$CMD_A" status > "$status_out" 2>&1
assert_rc "$?" 0 "status succeeds after install"
assert_grep "installed: $menu" "$status_out" "status reports the menu entry installed"
assert_grep "absent:    $login" "$status_out" "status reports no login entry"
assert_grep "tray does not start at login" "$status_out" "status says login is off"

# Every path status prints lives inside the isolated home.
stray=$(grep -E 'installed:|absent:' "$status_out" \
        | grep -vF "$HOME_DIR" || true)
assert_eq "$stray" "" "status only ever names paths under \$HOME"

# ------------------------------------------------- 2. explicit login opt-in
run_installer "$CMD_A" install --autostart > "$out" 2>&1
assert_rc "$?" 0 "install --autostart succeeds"
assert_exists "$login" "--autostart writes the login entry"
assert_grep "Exec=\"$CMD_A\" tray" "$login" "login entry launches 'listn tray'"

status_out=$WORK/status2.log
run_installer "$CMD_A" status > "$status_out" 2>&1
assert_grep "tray starts at login" "$status_out" "status says login is on after opt-in"

# ------------------------------------- 3. upgrade: re-install refreshes entries
# The login entry the user already opted into must follow a changed command,
# or a login would start a stale binary after an upgrade.
run_installer "$CMD_B" install > "$out" 2>&1
assert_rc "$?" 0 "re-install with a new command succeeds"
assert_grep "Exec=\"$CMD_B\" ui" "$menu" "re-install refreshes the menu entry"
assert_grep "Exec=\"$CMD_B\" tray" "$login" "re-install refreshes the existing login entry"
assert_eq "$(find "$apps" -name 'listn.desktop' | wc -l | tr -d ' ')" 1 \
    "exactly one menu entry after re-install (no duplicates)"
assert_eq "$(find "$autostart" -name 'listn-tray.desktop' | wc -l | tr -d ' ')" 1 \
    "exactly one login entry after re-install (no duplicates)"

# ---------------------------------- 4. plant user data and foreign files
data="$HOME_DIR/.local/share/listn"
mkdir -p "$data/transcripts" "$data/env" "$HOME_DIR/.cache/huggingface"
echo "recording-1" > "$data/transcripts/rec1.wav"
echo '{"model": "small"}' > "$data/settings.json"
echo "weights" > "$HOME_DIR/.cache/huggingface/model.bin"
echo "site-packages" > "$data/env/keep.txt"

foreign_menu="$apps/other-app.desktop"
foreign_icon="$icons/other.png"
foreign_login="$autostart/other-app.desktop"
echo "other" > "$foreign_menu"
echo "other" > "$foreign_icon"
echo "other" > "$foreign_login"

# ------------------------------------------------------- 5. uninstall
run_installer "$CMD_B" uninstall > "$out" 2>&1
assert_rc "$?" 0 "uninstall succeeds"
assert_absent "$menu" "uninstall removes listn's menu entry"
assert_absent "$icon" "uninstall removes listn's icon"
assert_absent "$login" "uninstall removes listn's login entry"
assert_exists "$foreign_menu" "uninstall leaves another app's menu entry alone"
assert_exists "$foreign_icon" "uninstall leaves another app's icon alone"
assert_exists "$foreign_login" "uninstall leaves another app's login entry alone"
assert_exists "$data/transcripts/rec1.wav" "uninstall leaves transcripts alone"
assert_exists "$data/settings.json" "uninstall leaves settings alone"
assert_exists "$HOME_DIR/.cache/huggingface/model.bin" "uninstall leaves the model cache alone"
assert_exists "$data/env/keep.txt" "uninstall leaves the user's environment alone"

status_out=$WORK/status3.log
run_installer "$CMD_B" status > "$status_out" 2>&1
assert_rc "$?" 0 "status succeeds after uninstall"
assert_grep "absent:    $menu" "$status_out" "status reports the menu entry gone"
assert_grep "tray does not start at login" "$status_out" "status reports login off after uninstall"

# Uninstall is idempotent: running it again is not an error.
run_installer "$CMD_B" uninstall > "$out" 2>&1
assert_rc "$?" 0 "second uninstall is harmless"

# --------------------------------------------------------- 6. dry-run
before=$(find "$HOME_DIR" | sort)
run_installer "$CMD_A" install --autostart --dry-run > "$out" 2>&1
assert_rc "$?" 0 "dry-run install succeeds"
assert_grep "dry-run:" "$out" "dry-run announces what it would do"
assert_eq "$(find "$HOME_DIR" | sort)" "$before" "dry-run writes nothing"

# ------------------------------------------------- 7. platform and CLI guards
shim="$WORK/shim"
mkdir -p "$shim"
printf '#!/bin/sh\necho Darwin\n' > "$shim/uname"
chmod 755 "$shim/uname"
PATH="$shim:$PATH" env -u XDG_DATA_HOME -u XDG_CONFIG_HOME \
    HOME="$HOME_DIR" LISTN_CMD="$CMD_A" \
    sh "$installer" status > "$out" 2>&1
assert_rc "$?" 2 "installer refuses to run on macOS"
assert_grep "listn-manage.sh" "$out" "macOS refusal points at the macOS script"

run_installer "$CMD_A" frobnicate > "$out" 2>&1
assert_rc "$?" 1 "unknown command fails"

# --------------------------------- 8. no system-wide writes at any point
sys_after=$(sys_snapshot)
assert_eq "$sys_after" "$sys_before" \
    "system XDG autostart/app/icon directories are unchanged"

# ------------------------------------------------------------ summary
if [ "$failed" -eq 0 ]; then
    printf 'PASS %d/%d checks\n' "$n" "$n"
    exit 0
fi
printf 'FAIL %d of %d checks\n' "$failed" "$n" >&2
exit 1
