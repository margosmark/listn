#!/bin/sh
# Clean Arch build / install / upgrade / uninstall check for the listn
# package (packaging/arch/PKGBUILD).
#
# Modes, in order:
#   1. native     - makepkg is on PATH (Arch box, or the archlinux CI
#                   container): full lifecycle below.
#   2. container  - docker/podman available: re-executes this script inside
#                   archlinux:latest with the repo mounted read-only.
#   3. skip       - neither: prints SKIP and exits 0. Nothing is written;
#                   this machine's package lifecycle is NOT validated.
#
# The native lifecycle, all as an unprivileged builder user for build/first
# run and root only for pacman -U/-R (or sudo on a developer box):
#   fetch the pinned tag source -> makepkg -sf -> pacman -U ->
#   assert the owned file set (user-scope only, no /etc) ->
#   opt-in login entry per the .install message + synthetic user data ->
#   first run (`listn --version`, builds the user's uv env) ->
#   one-engine-at-login check (engine up, `listn tray` attaches; needs
#   dbus-run-session, skipped with a note when absent) ->
#   pkgrel-bump rebuild + pacman -U upgrade (post_upgrade message, user
#   state retained) -> pacman -R removal (post_remove message; packaged
#   files gone, user data / model cache / login entry retained).
#
# This is a package lifecycle check, NOT a Plasma login test: it never
# starts plasmashell, never shows the app menu or tray panel and never
# logs a session in. Menu visibility, the tray icon in a real panel and
# autostart firing at login stay manual acceptance (docs/OPERATIONS.md).
#
# Usage: scripts/arch_pkg_check.sh [--help]
# Env:   GITHUB_TOKEN          auth for the private repo's tag archive
#                             (falls back to `gh auth token`, then anonymous)
#        LISTN_CHECK_SKIP_FIRST_RUN=1   skip the uv first-run download
#
# Exit: 0 on PASS or SKIP, 1 on any failed check.

set -eu

usage() {
    cat <<'EOF'
usage: scripts/arch_pkg_check.sh [--help]

Runs the clean Arch package lifecycle: build from the pinned tag source,
install, first run, one-engine-at-login, upgrade, uninstall - asserting
user-scope-only files and user-data retention at each step.

Modes: native (makepkg) | container (docker/podman -> archlinux:latest)
       | skip (exit 0, nothing written).

Environment:
  GITHUB_TOKEN                 credentials for the private tag archive
  LISTN_CHECK_SKIP_FIRST_RUN=1 skip the first-run uv download
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH='' cd -- "$script_dir/.." && pwd -P)
pkgbuild="$repo_root/packaging/arch/PKGBUILD"
[ -f "$pkgbuild" ] || { echo "not found: $pkgbuild" >&2; exit 2; }

failures=0
pass() { printf 'PASS: %s\n' "$1"; }
fail() { failures=$((failures + 1)); printf 'FAIL: %s\n' "$1" >&2; }
note() { printf '%s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- mode select
if ! command -v makepkg >/dev/null 2>&1; then
    runtime=$(command -v docker || command -v podman || true)
    if [ -n "$runtime" ]; then
        note "makepkg not found; re-running in an Arch container via $runtime"
        # shellcheck disable=SC2086  # deliberate word split of the -e pair
        # shellcheck disable=SC2016  # expands inside the container, not here
        exec "$runtime" run --rm \
            ${GITHUB_TOKEN:+-e GITHUB_TOKEN} \
            ${LISTN_CHECK_SKIP_FIRST_RUN:+-e LISTN_CHECK_SKIP_FIRST_RUN} \
            -v "$repo_root:/repo:ro" -w /repo \
            archlinux:latest \
            sh -c 'set -eu
                   b=$(mktemp -d)
                   tar -C /repo --exclude=.git --exclude=.venv -cf - . \
                     | tar -C "$b" -xf -
                   cd "$b"
                   sh scripts/arch_pkg_check.sh'
    fi
    note "SKIP: no makepkg and no docker/podman on this machine."
    note "The Arch build/install lifecycle was NOT validated here; CI runs"
    note "it in an archlinux container (.github/workflows/arch-package.yml)."
    exit 0
fi

# ------------------------------------------------------------ PKGBUILD facts
fields=$(bash -c "source \"$pkgbuild\"; printf '%s\n' \"\$pkgver\" \"\$pkgrel\" \"\${source[0]}\"")
pkgver=$(printf '%s\n' "$fields" | sed -n '1p')
pkgrel=$(printf '%s\n' "$fields" | sed -n '2p')
source0=$(printf '%s\n' "$fields" | sed -n '3p')
fname=${source0%%::*}
url=${source0#*::}
if [ -z "$pkgver" ] || [ -z "$url" ]; then die "cannot parse $pkgbuild"; fi
note "package: listn $pkgver-$pkgrel from $url"

is_root=no
if [ "$(id -u)" -eq 0 ]; then is_root=yes; fi

builder=""
run_as_builder() { "$@"; }
if [ "$is_root" = yes ]; then
    note "installing build tooling (pacman -Sy base-devel sudo curl git dbus)"
    pacman -Syu --noconfirm --needed base-devel sudo curl git dbus shadow
    builder=builder
    id builder >/dev/null 2>&1 || useradd -m -s /bin/sh builder
    printf 'builder ALL=(ALL) NOPASSWD: ALL\n' > /etc/sudoers.d/listn-check
    chmod 440 /etc/sudoers.d/listn-check
    run_as_builder() { sudo -u builder -H "$@"; }
fi
builder_home=$(getent passwd "${builder:-$(id -un)}" | cut -d: -f6)
[ -n "$builder_home" ] || die "cannot resolve the builder home"
note "builder: ${builder:-$(id -un)} (home $builder_home)"

build_dir=$(run_as_builder mktemp -d /tmp/listn-arch-check.XXXXXX)
# Invoked indirectly through the EXIT trap.
# shellcheck disable=SC2317
cleanup() {
    status=$?
    if [ "$status" -eq 0 ] && [ "$failures" -eq 0 ]; then
        rm -rf "$build_dir"
    else
        note "build dir kept for inspection: $build_dir"
    fi
    exit "$status"
}
trap cleanup EXIT
cp "$pkgbuild" "$repo_root/packaging/arch/listn.install" "$build_dir/"

# ------------------------------------------------- fetch the pinned source
if [ ! -f "$build_dir/$fname" ]; then
    token="${GITHUB_TOKEN:-}"
    if [ -z "$token" ] && command -v gh >/dev/null 2>&1; then
        token=$(gh auth token 2>/dev/null || true)
    fi
    fetched=no
    if [ -n "$token" ]; then
        if printf 'user = "x-access-token:%s"\n' "$token" \
                | curl -fsSL -K - -o "$build_dir/$fname" "$url"; then
            fetched=yes
        else
            note "authed fetch failed; trying anonymously"
        fi
    fi
    if [ "$fetched" = no ]; then
        if curl -fsSL -o "$build_dir/$fname" "$url"; then
            fetched=yes
        else
            rm -f "$build_dir/$fname"
            die "cannot fetch $url - set GITHUB_TOKEN (private repo) or run where gh/git credentials exist"
        fi
    fi
    note "fetched $fname"
fi

# ------------------------------------------------------------------- build
cd "$build_dir"
if build_out=$(run_as_builder makepkg -sf --noconfirm 2>&1); then
    printf '%s\n' "$build_out"
    pass "makepkg built from the tagged source ($fname)"
else
    printf '%s\n' "$build_out" >&2
    fail "makepkg build failed"
    exit 1
fi
pkgfile=$(find "$build_dir" -maxdepth 1 -name "listn-$pkgver-$pkgrel-*.pkg.tar.*" | sed -n '1p')
[ -n "$pkgfile" ] || die "built package not found in $build_dir"
note "built: $(basename "$pkgfile")"

# ---------------------------------------------------------------- install
if [ "$is_root" = yes ]; then
    install_out=$(pacman -U --noconfirm "$pkgfile" 2>&1) || true
else
    install_out=$(sudo pacman -U --noconfirm "$pkgfile" 2>&1) || true
fi
printf '%s\n' "$install_out"
if pacman -Q listn >/dev/null 2>&1; then
    pass "pacman -U installed listn $(pacman -Q listn | awk '{print $2}')"
else
    fail "pacman -U did not install listn"
    exit 1
fi
if printf '%s\n' "$install_out" | grep -q "Optional - start the tray at login"; then
    pass "post_install message shows the opt-in command"
else
    fail "post_install message missing the opt-in command"
fi

# ------------------------------------------------- owned file set, no /etc
qlist=$(pacman -Ql listn)
printf '%s\n' "$qlist"
outside_usr=$(printf '%s\n' "$qlist" | awk '$2 !~ /^\/usr\// {print}')
if [ -z "$outside_usr" ]; then
    pass "package owns files under /usr only"
else
    fail "package owns non-/usr files: $outside_usr"
fi
if printf '%s\n' "$qlist" | grep -q " /etc/"; then
    fail "package ships files under /etc"
else
    pass "no files under /etc (nothing system-wide)"
fi
if printf '%s\n' "$qlist" | grep -Eq "__pycache__|\.pyc|/env/|egg-info"; then
    fail "package ships caches or an environment"
else
    pass "no caches, venv or egg-info in the package"
fi
for f in /usr/bin/listn \
         /usr/share/applications/listn.desktop \
         /usr/share/icons/hicolor/128x128/apps/listn.png \
         /usr/share/listn/packaging/listn-tray.desktop \
         /usr/share/listn/uv.lock; do
    if pacman -Qo "$f" >/dev/null 2>&1; then
        pass "installed: $f"
    else
        fail "missing or unowned: $f"
    fi
done

# ---------------------------------------- login stays an explicit opt-in
if [ -e /etc/xdg/autostart/listn-tray.desktop ]; then
    fail "system autostart entry installed without opt-in"
else
    pass "nothing installed into /etc/xdg/autostart"
fi
autostart="$builder_home/.config/autostart/listn-tray.desktop"
if [ -e "$autostart" ]; then
    fail "builder already has a login entry before opt-in"
else
    pass "no login entry before the explicit opt-in"
fi

# Opt in exactly as the .install message tells the user, and plant
# synthetic user data the later steps must prove retained.
run_as_builder install -Dm644 /usr/share/listn/packaging/listn-tray.desktop \
    "$autostart"
run_as_builder mkdir -p "$builder_home/.local/share/listn/transcripts" \
    "$builder_home/.cache/huggingface"
printf 'synthetic\n' | run_as_builder tee \
    "$builder_home/.local/share/listn/transcripts/lifecycle.wav" >/dev/null
printf 'synthetic\n' | run_as_builder tee \
    "$builder_home/.cache/huggingface/model.bin" >/dev/null
if [ -e "$autostart" ]; then
    pass "opt-in command from the .install message creates the login entry"
else
    fail "opt-in command did not create $autostart"
fi

# ------------------------------------------------------------- first run
if [ "${LISTN_CHECK_SKIP_FIRST_RUN:-0}" = 1 ]; then
    note "first run skipped (LISTN_CHECK_SKIP_FIRST_RUN=1)"
else
    if first_out=$(run_as_builder timeout 900 /usr/bin/listn --version 2>&1); then
        printf '%s\n' "$first_out"
        pass "first run: /usr/bin/listn --version (builds the user's env)"
    else
        printf '%s\n' "$first_out" >&2
        fail "first run: /usr/bin/listn --version failed"
    fi
    if [ -x "$builder_home/.local/share/listn/env/bin/python" ]; then
        pass "user environment lives under ~/.local/share/listn/env"
    else
        fail "user environment was not created under ~/.local/share/listn/env"
    fi
    if [ -e /usr/share/listn/listn/__pycache__ ]; then
        fail "first run wrote __pycache__ into /usr/share/listn"
    else
        pass "first run wrote nothing into /usr/share/listn"
    fi
fi

# ------------------------------------- one engine at login (tray attaches)
if command -v dbus-run-session >/dev/null 2>&1 \
        && [ -x "$builder_home/.local/share/listn/env/bin/python" ]; then
    # Started directly (not through the helper function) so $! is the
    # engine's own process and can be killed at the end of the check.
    if [ "$is_root" = yes ]; then
        # shellcheck disable=SC2024  # root-owned log in the build dir, intended
        sudo -u builder -H /usr/bin/listn ui --no-browser \
            >"$build_dir/engine.log" 2>&1 &
    else
        /usr/bin/listn ui --no-browser \
            >"$build_dir/engine.log" 2>&1 &
    fi
    engine_pid=$!
    engine_up=no
    i=0
    while [ "$i" -lt 60 ]; do
        if run_as_builder python -c \
                "import socket; s=socket.create_connection(('127.0.0.1', 8765), 1); s.close()" \
                >/dev/null 2>&1; then
            engine_up=yes
            break
        fi
        i=$((i + 1))
        sleep 1
    done
    if [ "$engine_up" = yes ]; then
        tray_out=$(run_as_builder env PYTHONUNBUFFERED=1 timeout 20 \
            dbus-run-session -- /usr/bin/listn tray 2>&1 || true)
        printf '%s\n' "$tray_out"
        if printf '%s\n' "$tray_out" | grep -q "attached to the engine"; then
            pass "login tray attaches to the running engine (no second engine)"
        else
            fail "tray did not attach to the running engine: $tray_out"
        fi
        if run_as_builder "$builder_home/.local/share/listn/env/bin/python" \
                -c "import dbus_next" 2>/dev/null; then
            pass "tray extra (dbus-next) pulled into the user env on demand"
        else
            fail "dbus-next missing from the user env after listn tray"
        fi
    else
        fail "engine did not come up on 127.0.0.1:8765 (see $build_dir/engine.log)"
        cat "$build_dir/engine.log" >&2 || true
    fi
    kill "$engine_pid" 2>/dev/null || true
    wait "$engine_pid" 2>/dev/null || true
else
    note "SKIP: one-engine-at-login check (dbus-run-session or user env missing)"
fi

# ---------------------------------------------------------------- upgrade
sed -i "s/^pkgrel=.*/pkgrel=99/" "$build_dir/PKGBUILD"
if up_out=$(run_as_builder makepkg -sf --noconfirm 2>&1); then
    printf '%s\n' "$up_out"
    pass "upgrade rebuild (pkgrel=99) succeeded"
else
    printf '%s\n' "$up_out" >&2
    fail "upgrade rebuild failed"
fi
up_pkg=$(find "$build_dir" -maxdepth 1 -name "listn-$pkgver-99-*.pkg.tar.*" | sed -n '1p')
[ -n "$up_pkg" ] || die "upgraded package artifact not found"
if [ "$is_root" = yes ]; then
    up_install=$(pacman -U --noconfirm "$up_pkg" 2>&1) || true
else
    up_install=$(sudo pacman -U --noconfirm "$up_pkg" 2>&1) || true
fi
printf '%s\n' "$up_install"
if pacman -Q listn | grep -q " $pkgver-99$"; then
    pass "pacman -U upgraded in place to $pkgver-99"
else
    fail "upgrade did not land (now: $(pacman -Q listn 2>/dev/null || echo none))"
fi
if printf '%s\n' "$up_install" | grep -q "Optional - start the tray at login"; then
    pass "post_upgrade repeats the install guidance"
else
    fail "post_upgrade message missing"
fi
for kept in "$autostart" \
            "$builder_home/.local/share/listn/env" \
            "$builder_home/.local/share/listn/transcripts/lifecycle.wav" \
            "$builder_home/.cache/huggingface/model.bin"; do
    if [ -e "$kept" ]; then
        pass "retained through upgrade: $kept"
    else
        fail "lost through upgrade: $kept"
    fi
done

# -------------------------------------------------------------- uninstall
if [ "$is_root" = yes ]; then
    rm_out=$(pacman -R --noconfirm listn 2>&1) || true
else
    rm_out=$(sudo pacman -R --noconfirm listn 2>&1) || true
fi
printf '%s\n' "$rm_out"
if pacman -Q listn >/dev/null 2>&1; then
    fail "listn is still installed after pacman -R"
else
    pass "pacman -R removed the package"
fi
if printf '%s\n' "$rm_out" | grep -q "left in place"; then
    pass "post_remove message says user data is left in place"
else
    fail "post_remove message missing"
fi
for gone in /usr/bin/listn \
            /usr/share/listn \
            /usr/share/applications/listn.desktop \
            /usr/share/icons/hicolor/128x128/apps/listn.png; do
    if [ -e "$gone" ]; then
        fail "still present after removal: $gone"
    else
        pass "removed with the package: $gone"
    fi
done
for kept in "$autostart" \
            "$builder_home/.local/share/listn/env" \
            "$builder_home/.local/share/listn/transcripts/lifecycle.wav" \
            "$builder_home/.cache/huggingface/model.bin"; do
    if [ -e "$kept" ]; then
        pass "retained through uninstall: $kept"
    else
        fail "lost through uninstall: $kept"
    fi
done
if [ "$is_root" = yes ]; then
    rm -f /etc/sudoers.d/listn-check
fi

# --------------------------------------------------------------- summary
if [ "$failures" -eq 0 ]; then
    note "PASS: Arch package lifecycle (build/install/first-run/upgrade/uninstall)"
    exit 0
fi
die "$failures lifecycle checks failed"
