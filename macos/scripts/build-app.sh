#!/bin/bash
# Build listn.app with Swift Package Manager and Command Line Tools.
# No Xcode project. The app is ad-hoc signed so macOS will attach the
# audio-capture and microphone permission prompts to this bundle.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The menu bar app builds only on macOS (it links Core Audio and AppKit)." >&2
  exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "swift was not found. Install the Command Line Tools: xcode-select --install" >&2
  exit 1
fi
if ! command -v codesign >/dev/null 2>&1; then
  echo "codesign was not found. It ships with the Command Line Tools." >&2
  exit 1
fi

MACOS="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$MACOS/.." && pwd)"
cd "$MACOS"

echo "Building ListnMenu (deployment target macOS 14.2, from Package.swift)"
swift build -c release

APP="$MACOS/Listn.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$MACOS/.build/release/ListnMenu" "$APP/Contents/MacOS/Listn"
cp "$MACOS/Info.plist" "$APP/Contents/Info.plist"

ICONSET="$MACOS/Resources/AppIcon.iconset"
if [[ -d "$ICONSET" ]] && command -v iconutil >/dev/null 2>&1; then
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
  echo "Icon: $APP/Contents/Resources/AppIcon.icns"
else
  echo "No iconutil, or no iconset. The app builds without AppIcon.icns on this machine." >&2
fi

# The default app carries its own interpreter and site-packages, so opening
# it does not need this checkout or uv. Model files stay in ~/.cache/huggingface.
# LISTN_STANDALONE=0 keeps the older pointer at .venv/bin/listn.
if [[ "${LISTN_STANDALONE:-1}" == "1" ]]; then
  PY="$ROOT/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    echo "No $PY. Run: uv sync --extra mlx" >&2
    echo "Or build against an existing binary with LISTN_STANDALONE=0." >&2
    exit 1
  fi
  echo "Bundling Python into the app (this can take a minute; models are not copied)"
  "$PY" -m listn.bundle --venv "$ROOT/.venv" --project "$ROOT" --dest "$APP/Contents/Resources/python"
  printf '%s\n' "python/bin/listn" > "$APP/Contents/Resources/engine.path"
else
  ENGINE="$ROOT/.venv/bin/listn"
  if [[ -x "$ENGINE" ]]; then
    printf '%s\n' "$ENGINE" > "$APP/Contents/Resources/engine.path"
    echo "Engine: $ENGINE"
  else
    echo "No $ENGINE yet. The app will look for LISTN_BIN, a .venv next to the repo, or Homebrew's listn." >&2
  fi
fi

codesign --force --deep --sign - --identifier local.listn.menu "$APP"
codesign --verify --verbose=2 "$APP"
echo "App size (environment only; Whisper and the notes model are separate):"
du -sh "$APP" | tee "$APP/Contents/Resources/SIZE.txt"
codesign --verify --verbose=2 "$APP"
echo
echo "Built $APP"
echo "Run it with:  open \"$APP\""
