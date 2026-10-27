#!/bin/bash
# Build listn.app with Swift Package Manager and Command Line Tools.
# No Xcode project.
#
# Signing: ad-hoc by default (codesign --sign -). macOS keys privacy grants
# for an ad-hoc app to that exact build, so after a rebuild the old
# Microphone / System Audio Recording grant no longer matches and capture
# can come back as silence. Either reset the grants after each rebuild
# (printed at the end), or sign with a stable identity:
#   LISTN_SIGN_IDENTITY="listn local" macos/scripts/build-app.sh
# (create a self-signed "Code Signing" certificate once in Keychain Access →
# Certificate Assistant → Create a Certificate).
#
# The app is assembled in a staging folder and only replaces Listn.app once
# every step, including signature verification, has passed.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The menu bar app builds only on macOS (it links Core Audio and AppKit)." >&2
  exit 1
fi
for tool in swift codesign; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool was not found. Install the Command Line Tools: xcode-select --install" >&2
    exit 1
  fi
done

MACOS="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$MACOS/.." && pwd)"
IDENTITY="${LISTN_SIGN_IDENTITY:--}"
APP="$MACOS/Listn.app"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/listn-build.XXXXXX")"
NEW="$STAGE/Listn.app"
cleanup() { rm -rf "$STAGE"; }
fail() { echo "build-app: $*" >&2; echo "Listn.app was not changed." >&2; exit 1; }
trap cleanup EXIT

cd "$MACOS"
echo "Building ListnMenu (deployment target macOS 14.2, from Package.swift)"
swift build -c release || fail "swift build failed."

mkdir -p "$NEW/Contents/MacOS" "$NEW/Contents/Resources"
cp "$MACOS/.build/release/ListnMenu" "$NEW/Contents/MacOS/Listn"
cp "$MACOS/Info.plist" "$NEW/Contents/Info.plist"

ICONSET="$MACOS/Resources/AppIcon.iconset"
if [[ -d "$ICONSET" ]] && command -v iconutil >/dev/null 2>&1; then
  iconutil -c icns "$ICONSET" -o "$NEW/Contents/Resources/AppIcon.icns" || fail "iconutil failed."
else
  echo "No iconutil or no iconset; building without AppIcon.icns." >&2
fi

# The default app carries its own interpreter and site-packages, so opening
# it needs neither this checkout nor uv. Model files stay in ~/.cache/huggingface.
# LISTN_STANDALONE=0 points at .venv/bin/listn instead.
#
# Speaker labels (pyannote.audio 4 + torch, the "diarize" extra) are part of
# the app. LISTN_WITHOUT_SPEAKERS=1 builds without them; the app then says
# so in Settings and keeps Me / Others only.
SYNC_HINT="uv sync --extra mlx --extra llm --extra diarize"
if [[ "${LISTN_STANDALONE:-1}" == "1" ]]; then
  PY="$ROOT/.venv/bin/python"
  [[ -x "$PY" ]] || fail "No $PY. Run: $SYNC_HINT (or set LISTN_STANDALONE=0)."
  if [[ "${LISTN_WITHOUT_SPEAKERS:-0}" != "1" ]]; then
    "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyannote.audio') and importlib.util.find_spec('torch') else 1)" \
      || fail "The diarize extra is not installed in .venv. Run: $SYNC_HINT (or set LISTN_WITHOUT_SPEAKERS=1)."
  fi
  echo "Bundling Python into the app (models are not copied)"
  PYDEST="$NEW/Contents/Resources/python"
  "$PY" -m listn.bundle --venv "$ROOT/.venv" --project "$ROOT" --dest "$PYDEST" || fail "Bundling Python failed."
  "$PYDEST/bin/listn" --version >/dev/null || fail "The bundled engine does not start."
  if [[ "${LISTN_WITHOUT_SPEAKERS:-0}" != "1" ]]; then
    # Imports pyannote and torch from the bundle. No model is loaded and nothing is downloaded.
    "$PYDEST/bin/listn" speakers --check >/dev/null || fail "pyannote.audio does not import from the bundled Python."
    echo "Speaker labels: pyannote.audio and torch are bundled"
  fi
  printf '%s\n' "python/bin/listn" > "$NEW/Contents/Resources/engine.path"
  echo "Signing $(wc -l < "$PYDEST/macho.txt" | tr -d ' ') Mach-O files inside the bundled Python"
  ( cd "$PYDEST" && tr '\n' '\0' < macho.txt | xargs -0 -n 64 codesign --force --sign "$IDENTITY" --timestamp=none ) \
    || fail "Signing the bundled Python failed."
else
  ENGINE="$ROOT/.venv/bin/listn"
  "$ROOT/.venv/bin/python" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyannote.audio') else 1)" 2>/dev/null \
    || echo "Note: .venv has no diarize extra, so speaker labels are off. Run: $SYNC_HINT" >&2
  if [[ -x "$ENGINE" ]]; then
    printf '%s\n' "$ENGINE" > "$NEW/Contents/Resources/engine.path"
    echo "Engine: $ENGINE"
  else
    echo "No $ENGINE yet. The app will look for LISTN_BIN, a .venv next to the repo, or Homebrew's listn." >&2
  fi
fi

# Everything inside the bundle must be in place before the outer signature
# seals it; writing a file afterwards breaks the seal.
du -sh "$NEW" | awk '{print $1 " (app + Python environment; models are separate)"}' > "$NEW/Contents/Resources/SIZE.txt"

codesign --force --sign "$IDENTITY" --identifier local.listn.menu --timestamp=none "$NEW" || fail "codesign failed."
codesign --verify --deep --strict --verbose=2 "$NEW" || fail "The signature did not verify."

rm -rf "$APP"
mv "$NEW" "$APP"
echo
echo "Built $APP ($(cat "$APP/Contents/Resources/SIZE.txt"))"
echo "Run it with:  open \"$APP\""
if [[ "$IDENTITY" == "-" ]]; then
  echo
  echo "Ad-hoc signed. If recordings come back silent after this rebuild, reset the old grants and allow again:"
  echo "  tccutil reset Microphone local.listn.menu"
  echo "  tccutil reset AudioCapture local.listn.menu"
fi
