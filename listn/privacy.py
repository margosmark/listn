"""macOS privacy panes and the messages shown when capture is silence.

macOS delivers exact zeros both when permission is missing and when the tap
or the microphone is opened in a way that never receives samples. The message
says the capture is silence and points at capture.log; the Settings pane is
there for the case where access really was reset.
"""

from __future__ import annotations

import subprocess
import sys

PANES = {
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "audio-capture": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
}

SILENT_SESSION = (
    "This recording is silence. listn received audio frames, but every sample was zero. "
    "See ~/Library/Application Support/listn/capture.log."
)


def silence_message(track: str) -> tuple[str, str]:
    if track == "system":
        return (
            "listn is recording, but system audio returned silence (every sample is zero). "
            "See ~/Library/Application Support/listn/capture.log. "
            "If access was reset, allow listn in System Settings > Privacy & Security > "
            "Screen & System Audio Recording.",
            "audio-capture",
        )
    return (
        "listn is recording, but the microphone returned silence (every sample is zero). "
        "See ~/Library/Application Support/listn/capture.log. "
        "If access was reset, allow listn in System Settings > Privacy & Security > Microphone.",
        "microphone",
    )


def open_pane(pane: str) -> bool:
    url = PANES.get(pane)
    if url is None:
        raise KeyError(pane)
    if sys.platform != "darwin":
        return False
    subprocess.run(["open", url], check=False)
    return True
