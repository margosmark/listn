#!/bin/bash
# listn-manage.sh: set up, build, install, update, check and remove listn on a Mac.
#
#   scripts/listn-manage.sh <command> [options]
#
# Run with no arguments (or `help`) for the command list. Every command is
# safe to run again. Nothing here deletes recordings, transcripts or voice
# prints unless you pass --purge-data and confirm.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Listn.app"
BUILT_APP="$ROOT/macos/$APP_NAME"
INSTALL_DIR="${LISTN_INSTALL_DIR:-/Applications}"
INSTALLED_APP="$INSTALL_DIR/$APP_NAME"
BUNDLE_ID="local.listn.menu"
DATA_DIR="${LISTN_DATA_DIR:-$HOME/Library/Application Support/listn}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CERT_NAME="${LISTN_CERT_NAME:-listn local}"
ENGINE_URL="http://127.0.0.1:${LISTN_PORT:-8765}"
DEFAULT_EXTRAS=(mlx llm diarize)

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

DRY_RUN=0
ASSUME_YES=0

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

say()  { printf '%s\n' "${BOLD}==>${RESET} $*"; }
ok()   { printf '%s\n' "  ${GREEN}ok${RESET}    $*"; }
warn() { printf '%s\n' "  ${YELLOW}warn${RESET}  $*" >&2; }
bad()  { printf '%s\n' "  ${RED}fail${RESET}  $*" >&2; }
die()  { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }

# Run a command, or just print it with --dry-run.
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  [dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

confirm() {
  local prompt="$1"
  [[ "$ASSUME_YES" == "1" ]] && return 0
  [[ -t 0 ]] || die "$prompt Refusing without a terminal; pass --yes to confirm."
  local reply
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" || "$reply" == "yes" ]]
}

require_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || die "This command only works on macOS."
}

app_version() {
  local app="$1"
  /usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$app/Contents/Info.plist" 2>/dev/null || echo "unknown"
}

repo_version() {
  sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1
}

have_cert() {
  security find-identity -p codesigning 2>/dev/null | grep -q "\"$CERT_NAME\""
}

# The identity build-app.sh should sign with: LISTN_SIGN_IDENTITY if set,
# else the "listn local" certificate if it exists, else ad-hoc ("-").
sign_identity() {
  if [[ -n "${LISTN_SIGN_IDENTITY:-}" ]]; then
    printf '%s' "$LISTN_SIGN_IDENTITY"
  elif have_cert; then
    printf '%s' "$CERT_NAME"
  else
    printf '%s' "-"
  fi
}

app_running() {
  pgrep -f "$APP_NAME/Contents/MacOS/Listn" >/dev/null 2>&1
}

engine_running() {
  curl -fsS --max-time 2 "$ENGINE_URL/api/status" >/dev/null 2>&1
}

usage() {
  cat <<USAGE
${BOLD}listn-manage.sh${RESET}: manage the listn app from this checkout ($ROOT)

Usage: scripts/listn-manage.sh <command> [options]

Commands:
  doctor               Check prerequisites (macOS, Apple Silicon, Swift, Homebrew, uv, Python)
  setup                Install the Python environment with uv (extras: ${DEFAULT_EXTRAS[*]})
  test                 Run the Python test suite
  build                Build macos/$APP_NAME (wraps macos/scripts/build-app.sh)
  install              Build, then install into $INSTALL_DIR and relaunch
  update               git pull --ff-only, then setup, test, build and install
  start | stop | restart
                       Open, quit, or restart the installed app
  status               Installed version, running state, engine, speaker models, permissions hints
  logs [name]          Tail a log: engine (default), capture, selftest
  reset-permissions    Clear the Microphone and System Audio grants so macOS asks again
  signing-cert         Create the self-signed "$CERT_NAME" code-signing certificate once,
                       so permissions survive rebuilds
  uninstall            Quit and remove $INSTALLED_APP (keeps your data)
  first-run            doctor, setup, test, build, install: everything for a fresh clone

Options:
  --extras a,b,c       Extras for setup (default: mlx,llm,diarize; use "none" for the minimum)
  --skip-tests         Skip the test step in install, update and first-run
  --no-build           install: copy the existing macos/$APP_NAME without rebuilding
  --reset-permissions  install: always reset the permission grants (default: only when ad-hoc signed)
  --purge-data         uninstall: also delete recordings, transcripts and voice prints in
                       $DATA_DIR (asks first)
  --purge-models       uninstall: also delete downloaded models in $HF_CACHE (asks first;
                       other apps may share this cache)
  --yes                Answer yes to confirmations (needed without a terminal)
  --dry-run            Print what would run without changing anything
  -h, --help           Show this help

Environment:
  LISTN_SIGN_IDENTITY  Signing identity for build (default: "$CERT_NAME" if present, else ad-hoc)
  LISTN_INSTALL_DIR    Where to install (default: /Applications)
  LISTN_DATA_DIR       Data folder (default: ~/Library/Application Support/listn)
  LISTN_WITHOUT_SPEAKERS=1, LISTN_STANDALONE=0
                       Passed through to build-app.sh
USAGE
}

# ---------------------------------------------------------------- doctor

cmd_doctor() {
  say "Checking prerequisites"
  local failed=0

  if [[ "$(uname -s)" == "Darwin" ]]; then
    local ver; ver="$(sw_vers -productVersion)"
    local major minor
    major="${ver%%.*}"; minor="$(echo "$ver" | cut -d. -f2)"; minor="${minor:-0}"
    if (( major > 14 || (major == 14 && minor >= 2) )); then
      ok "macOS $ver (14.2 or later is required for process taps)"
    else
      bad "macOS $ver is too old; process taps need 14.2 or later"; failed=1
    fi
  else
    bad "Not macOS. The engine tests run on Linux, but the app needs a Mac."; failed=1
  fi

  if [[ "$(uname -m)" == "arm64" ]]; then
    ok "Apple Silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m))"
  else
    warn "Not Apple Silicon: MLX (GPU Whisper and the notes model) will not install"
  fi

  if command -v swift >/dev/null 2>&1; then
    ok "Swift: $(swift --version 2>/dev/null | head -1)"
  else
    bad "Swift not found. Install the Command Line Tools: xcode-select --install"; failed=1
  fi
  if command -v codesign >/dev/null 2>&1; then ok "codesign"; else bad "codesign not found (Command Line Tools)"; failed=1; fi

  if command -v brew >/dev/null 2>&1; then
    ok "Homebrew $(brew --version 2>/dev/null | head -1 | awk '{print $2}')"
  else
    warn "Homebrew not found (https://brew.sh). Needed only to install uv the easy way."
  fi

  if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version | awk '{print $2}')"
  else
    bad "uv not found. Install it: brew install uv"; failed=1
  fi

  if command -v uv >/dev/null 2>&1; then
    local py
    if py="$(uv python find '>=3.10,<3.14' 2>/dev/null)"; then
      ok "Python for uv: $py"
    else
      warn "No Python 3.10-3.13 found yet; uv will download one during setup"
    fi
  fi

  if command -v git >/dev/null 2>&1; then ok "git"; else warn "git not found (needed for update)"; fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    local id; id="$(sign_identity)"
    if [[ "$id" == "-" ]]; then
      warn "Builds will be ad-hoc signed, so permissions reset after each rebuild. Run: $0 signing-cert"
    else
      ok "Signing identity: $id"
    fi
  fi

  if (( failed )); then
    die "Fix the items marked fail, then run doctor again."
  fi
  say "Prerequisites look good"
}

# ---------------------------------------------------------------- setup / test

EXTRAS=("${DEFAULT_EXTRAS[@]}")

cmd_setup() {
  command -v uv >/dev/null 2>&1 || die "uv not found. Install it: brew install uv"
  local args=(sync)
  local e
  for e in ${EXTRAS[@]+"${EXTRAS[@]}"}; do
    [[ -n "$e" ]] && args+=(--extra "$e")
  done
  say "Installing the Python environment: uv ${args[*]}"
  (cd "$ROOT" && run uv "${args[@]}")
  ok "Python environment ready in $ROOT/.venv"
}

cmd_test() {
  command -v uv >/dev/null 2>&1 || die "uv not found. Install it: brew install uv"
  say "Running tests"
  (cd "$ROOT" && run uv run pytest -q)
}

# ---------------------------------------------------------------- build / install

cmd_build() {
  require_macos
  [[ -x "$ROOT/.venv/bin/python" ]] || die "No .venv yet. Run: $0 setup"
  local id; id="$(sign_identity)"
  say "Building $APP_NAME (signing: $( [[ "$id" == "-" ]] && echo ad-hoc || echo "$id" ))"
  run env LISTN_SIGN_IDENTITY="$id" "$ROOT/macos/scripts/build-app.sh"
}

quit_app() {
  if app_running; then
    say "Quitting listn"
    run osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
    if [[ "$DRY_RUN" != "1" ]]; then
      local _
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        app_running || break
        sleep 0.5
      done
      if app_running; then
        warn "listn did not quit in 5 seconds; stopping it"
        pkill -f "$APP_NAME/Contents" || true
        sleep 1
      fi
    fi
  fi
  stop_engine
}

engine_pids() {
  pgrep -f "$APP_NAME/Contents/Resources/python.* -m listn ui" 2>/dev/null || true
}

# The menu bar app starts the engine but an engine can outlive it, and a new
# app attaches to whatever engine is already listening. Stop it so the next
# launch runs the code that was just installed.
stop_engine() {
  local pids; pids="$(engine_pids)"
  [[ -n "$pids" ]] || return 0
  say "Stopping the listn engine"
  # shellcheck disable=SC2086
  run kill $pids || true
  if [[ "$DRY_RUN" != "1" ]]; then
    local _
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [[ -n "$(engine_pids)" ]] || return 0
      sleep 0.5
    done
    warn "The engine did not stop in 5 seconds; forcing it"
    # shellcheck disable=SC2046
    kill -9 $(engine_pids) 2>/dev/null || true
  fi
}

reset_permissions() {
  say "Resetting Microphone and System Audio Recording grants for $BUNDLE_ID"
  run tccutil reset Microphone "$BUNDLE_ID" || warn "tccutil Microphone reset failed"
  run tccutil reset AudioCapture "$BUNDLE_ID" || warn "tccutil AudioCapture reset failed"
  echo "  macOS will ask for both again on the next recording. Allow both."
}

RESET_PERMS=0
NO_BUILD=0
SKIP_TESTS=0

cmd_install() {
  require_macos
  if [[ "$NO_BUILD" != "1" ]]; then
    cmd_build
  fi
  [[ -d "$BUILT_APP" || "$DRY_RUN" == "1" ]] || die "No $BUILT_APP. Run: $0 build"
  if [[ "$DRY_RUN" != "1" ]]; then
    codesign --verify --deep --strict "$BUILT_APP" || die "$BUILT_APP does not verify; rebuild it."
  fi

  local old_version="none"
  [[ -d "$INSTALLED_APP" ]] && old_version="$(app_version "$INSTALLED_APP")"

  quit_app
  say "Installing into $INSTALLED_APP (was: $old_version)"
  local sudo=()
  [[ -w "$INSTALL_DIR" ]] || sudo=(sudo)
  run ${sudo[@]+"${sudo[@]}"} rm -rf "$INSTALLED_APP"
  run ${sudo[@]+"${sudo[@]}"} ditto "$BUILT_APP" "$INSTALLED_APP"

  local id; id="$(sign_identity)"
  if [[ "$RESET_PERMS" == "1" || "$id" == "-" ]]; then
    reset_permissions
  fi

  say "Starting listn"
  run open "$INSTALLED_APP"
  [[ "$DRY_RUN" == "1" ]] || ok "Installed $(app_version "$INSTALLED_APP"). Look for the waveform icon in the menu bar."
}

cmd_update() {
  require_macos
  command -v git >/dev/null 2>&1 || die "git not found"
  say "Pulling the latest code"
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
    die "You have uncommitted changes in $ROOT. Commit or stash them first."
  fi
  run git -C "$ROOT" pull --ff-only
  cmd_setup
  [[ "$SKIP_TESTS" == "1" ]] || cmd_test
  cmd_install
}

cmd_first_run() {
  cmd_doctor
  cmd_setup
  [[ "$SKIP_TESTS" == "1" ]] || cmd_test
  cmd_install
  cat <<NEXT

Next steps:
  1. Click the menu bar icon and start a recording. Allow Microphone and System Audio Recording.
  2. For speaker labels: accept the terms on https://huggingface.co/pyannote/speaker-diarization-community-1,
     create a read token at https://huggingface.co/settings/tokens, and paste it in Settings.
  3. Check everything with: $0 status
NEXT
}

# ---------------------------------------------------------------- start / stop / status

cmd_start() {
  require_macos
  [[ -d "$INSTALLED_APP" ]] || die "listn is not installed. Run: $0 install"
  if app_running; then ok "listn is already running"; return; fi
  run open "$INSTALLED_APP"
  ok "Started"
}

cmd_stop() {
  require_macos
  if app_running || [[ -n "$(engine_pids)" ]]; then quit_app; ok "Stopped"; else ok "listn is not running"; fi
}

cmd_restart() { cmd_stop; cmd_start; }

cmd_status() {
  require_macos
  say "listn status"
  echo "  checkout        $ROOT (version $(repo_version))"
  if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "  git             $(git -C "$ROOT" log -1 --format='%h %s' 2>/dev/null)"
  fi
  if [[ -d "$INSTALLED_APP" ]]; then
    echo "  installed app   $INSTALLED_APP (version $(app_version "$INSTALLED_APP"))"
    local auth
    auth="$(codesign -dv "$INSTALLED_APP" 2>&1 | sed -n 's/^Authority=//p' | head -1)"
    echo "  signed by       ${auth:-ad-hoc}"
    if codesign --verify --deep --strict "$INSTALLED_APP" 2>/dev/null; then
      echo "  signature       valid"
    else
      echo "  signature       ${RED}INVALID${RESET} (reinstall)"
    fi
  else
    echo "  installed app   not installed (run: $0 install)"
  fi
  if [[ -d "$BUILT_APP" ]]; then
    echo "  built app       $BUILT_APP (version $(app_version "$BUILT_APP"))"
  fi
  if app_running; then echo "  menu bar app    running"; else echo "  menu bar app    not running"; fi
  if engine_running; then
    echo "  engine          listening on $ENGINE_URL"
  else
    echo "  engine          not reachable at $ENGINE_URL"
  fi

  local listn_bin=""
  if [[ -x "$INSTALLED_APP/Contents/Resources/python/bin/listn" ]]; then
    listn_bin="$INSTALLED_APP/Contents/Resources/python/bin/listn"
  elif [[ -x "$ROOT/.venv/bin/listn" ]]; then
    listn_bin="$ROOT/.venv/bin/listn"
  fi
  if [[ -n "$listn_bin" ]]; then
    echo "  speaker labels  (from $listn_bin speakers --check)"
    "$listn_bin" speakers --check 2>/dev/null | sed 's/^/                  /' || echo "                  check failed"
  fi

  echo "  data folder     $DATA_DIR ($(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}' || echo 'missing'))"
  if [[ -d "$DATA_DIR/sessions" ]]; then
    echo "  sessions        $(find "$DATA_DIR/sessions" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  fi
  echo "  model cache     $HF_CACHE ($(du -sh "$HF_CACHE" 2>/dev/null | awk '{print $1}' || echo 'missing'))"
  if [[ "$(sign_identity)" == "-" ]]; then
    echo "  signing         ad-hoc (permissions reset after each rebuild)"
  else
    echo "  signing         $(sign_identity)"
  fi
  echo
  echo "  Permissions: System Settings > Privacy & Security > Microphone, and"
  echo "  Screen & System Audio Recording > System Audio Recording Only, should list listn."
}

cmd_logs() {
  local which="${1:-engine}"
  local file
  case "$which" in
    engine)   file="$DATA_DIR/engine.log" ;;
    capture)  file="$DATA_DIR/capture.log" ;;
    selftest) file="$DATA_DIR/selftest.log" ;;
    *) die "Unknown log '$which'. Use engine, capture or selftest." ;;
  esac
  [[ -f "$file" ]] || die "No $file yet."
  say "$file (Ctrl-C to stop)"
  tail -n 50 -f "$file"
}

# ---------------------------------------------------------------- signing certificate

cmd_signing_cert() {
  require_macos
  if have_cert; then
    ok "The \"$CERT_NAME\" code-signing certificate already exists"
    return
  fi
  say "Creating a self-signed code-signing certificate \"$CERT_NAME\" in your login keychain"
  echo "  It never leaves this Mac and is only used to sign Listn.app, so privacy"
  echo "  grants keep matching after rebuilds. macOS may ask for your login password."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] openssl req ... ; security import ... ; security add-trusted-cert ..."
    return
  fi
  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/listn-cert.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN
  cat > "$tmp/cert.cnf" <<CNF
[ req ]
distinguished_name = dn
prompt = no
x509_extensions = ext
[ dn ]
CN = $CERT_NAME
[ ext ]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
CNF
  local pass; pass="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24)"
  /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -config "$tmp/cert.cnf" -keyout "$tmp/key.pem" -out "$tmp/cert.pem" 2>/dev/null \
    || die "openssl could not create the certificate"
  /usr/bin/openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
    -name "$CERT_NAME" -out "$tmp/cert.p12" -passout "pass:$pass" 2>/dev/null \
    || die "openssl could not package the certificate"
  security import "$tmp/cert.p12" -k "$HOME/Library/Keychains/login.keychain-db" \
    -P "$pass" -T /usr/bin/codesign >/dev/null \
    || die "Importing into the login keychain failed"
  security add-trusted-cert -r trustRoot -p codeSign \
    -k "$HOME/Library/Keychains/login.keychain-db" "$tmp/cert.pem" \
    || warn "Could not mark the certificate as trusted for code signing; signing may still work"
  if have_cert; then
    ok "Created \"$CERT_NAME\". The next build and install sign with it automatically."
    echo "  Run '$0 install --reset-permissions' once so macOS asks for the grants under the new signature."
  else
    die "The certificate was imported but codesign cannot see it. Check Keychain Access for \"$CERT_NAME\"."
  fi
}

# ---------------------------------------------------------------- uninstall

PURGE_DATA=0
PURGE_MODELS=0

cmd_uninstall() {
  require_macos
  quit_app
  if [[ -d "$INSTALLED_APP" ]]; then
    say "Removing $INSTALLED_APP"
    local sudo=()
    [[ -w "$INSTALL_DIR" ]] || sudo=(sudo)
    run ${sudo[@]+"${sudo[@]}"} rm -rf "$INSTALLED_APP"
  else
    ok "$INSTALLED_APP is not installed"
  fi
  reset_permissions

  if [[ "$PURGE_DATA" == "1" ]]; then
    if [[ -d "$DATA_DIR" ]]; then
      echo
      warn "This deletes ALL recordings, transcripts, notes and voice prints in:"
      warn "  $DATA_DIR ($(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}'))"
      if confirm "Delete it permanently?"; then
        run rm -rf "$DATA_DIR"
        ok "Deleted $DATA_DIR"
      else
        ok "Kept $DATA_DIR"
      fi
    fi
  else
    ok "Kept your data in $DATA_DIR (use --purge-data to delete it)"
  fi

  if [[ "$PURGE_MODELS" == "1" ]]; then
    if [[ -d "$HF_CACHE" ]]; then
      warn "This deletes the whole Hugging Face cache $HF_CACHE ($(du -sh "$HF_CACHE" 2>/dev/null | awk '{print $1}')),"
      warn "including models other tools may use. They download again when needed."
      if confirm "Delete it?"; then
        run rm -rf "$HF_CACHE"
        ok "Deleted $HF_CACHE"
      else
        ok "Kept $HF_CACHE"
      fi
    fi
  else
    ok "Kept downloaded models in $HF_CACHE (use --purge-models to delete them)"
  fi
  echo "  The checkout in $ROOT is untouched. Delete it yourself if you want."
}

# ---------------------------------------------------------------- main

main() {
  local cmd="${1:-help}"
  [[ $# -gt 0 ]] && shift
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --extras)
        [[ $# -ge 2 ]] || die "--extras needs a value"
        if [[ "$2" == "none" ]]; then EXTRAS=(); else IFS=',' read -r -a EXTRAS <<< "$2"; fi
        shift 2 ;;
      --extras=*)
        local v="${1#--extras=}"
        if [[ "$v" == "none" ]]; then EXTRAS=(); else IFS=',' read -r -a EXTRAS <<< "$v"; fi
        shift ;;
      --skip-tests) SKIP_TESTS=1; shift ;;
      --no-build) NO_BUILD=1; shift ;;
      --reset-permissions) RESET_PERMS=1; shift ;;
      --purge-data) PURGE_DATA=1; shift ;;
      --purge-models) PURGE_MODELS=1; shift ;;
      --yes|-y) ASSUME_YES=1; shift ;;
      --dry-run|-n) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      -*) die "Unknown option $1 (see --help)" ;;
      *) positional+=("$1"); shift ;;
    esac
  done

  case "$cmd" in
    help|-h|--help) usage ;;
    doctor) cmd_doctor ;;
    setup) cmd_setup ;;
    test) cmd_test ;;
    build) cmd_build ;;
    install) cmd_install ;;
    update) cmd_update ;;
    first-run) cmd_first_run ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    status) cmd_status ;;
    logs) cmd_logs "${positional[@]+"${positional[@]}"}" ;;
    reset-permissions) require_macos; reset_permissions ;;
    signing-cert) cmd_signing_cert ;;
    uninstall) cmd_uninstall ;;
    *) usage >&2; die "Unknown command '$cmd'" ;;
  esac
}

main "$@"
