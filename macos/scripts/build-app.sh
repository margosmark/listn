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

ENGINE="$ROOT/.venv/bin/listn"
if [[ -x "$ENGINE" ]]; then
  printf '%s\n' "$ENGINE" > "$APP/Contents/Resources/engine.path"
  echo "Engine: $ENGINE"
else
  echo "No $ENGINE yet. The app will look for LISTN_BIN, a .venv next to the repo, or Homebrew's listn." >&2
  echo "Create the environment with: uv sync --extra mlx" >&2
fi

codesign --force --sign - --identifier local.listn.menu "$APP"
codesign --verify --verbose=2 "$APP"
echo
echo "Built $APP"
echo "Run it with:  open \"$APP\""
