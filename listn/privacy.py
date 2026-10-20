"""macOS privacy panes and the messages shown when capture is withheld.

When an app lacks Microphone or System Audio Recording access, macOS does
not raise an error: the stream runs and every sample is zero. listn treats
a couple of seconds of exact zeros as that case and says which pane to fix.
"""

from __future__ import annotations

import subprocess
import sys

PANES = {
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "audio-capture": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
}

SILENT_SESSION = (
    "listn received only silence. macOS sends silence when listn is not allowed to record. "
    "Allow listn in System Settings > Privacy & Security > Microphone and "
    "Screen & System Audio Recording, then record again."
)


def silence_message(track: str) -> tuple[str, str]:
    if track == "system":
        return ("listn isn't receiving system audio. Allow it in System Settings > Privacy & Security > "
                "Screen & System Audio Recording.", "audio-capture")
    return ("listn isn't receiving audio from the microphone. Allow it in System Settings > Privacy & Security > "
            "Microphone, then start again.", "microphone")


def open_pane(pane: str) -> bool:
    url = PANES.get(pane)
    if url is None:
        raise KeyError(pane)
    if sys.platform != "darwin":
        return False
    subprocess.run(["open", url], check=False)
    return True
