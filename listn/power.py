"""Battery and disk policy for a long recording.

The menu bar reports whether the Mac is on battery and the charge percent.
Below 40% on battery, with the saver on, the session uses the ``small``
Whisper model and skips the notes model until the machine is plugged in.
The saved settings are not changed. Recording refuses to start when free
space on the data volume is under ``min_free_gb``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from listn.config import Settings

LOW_BATTERY_PERCENT = 40.0
BATTERY_MODEL = "small"


class DiskSpaceError(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def require_free_space(path: Path, min_gb: float) -> float:
    free = free_gb(path)
    if free < float(min_gb):
        raise DiskSpaceError(
            f"Only {free:.1f} GB is free. listn keeps at least {float(min_gb):.1f} GB before it records."
        )
    return free


def model_for_power(settings: Settings, on_battery: bool, percent: float | None) -> tuple[str, bool]:
    """Return the Whisper model for this session, and whether to run the notes model.

    ``percent`` is 0–100. Unknown charge (``None``) does not force the saver,
    so a desktop or a missing sensor keeps the model you chose.
    """
    low = False
    if on_battery and percent is not None:
        try:
            low = float(percent) < LOW_BATTERY_PERCENT
        except (TypeError, ValueError):
            low = False
    if settings.battery_saver and low:
        return BATTERY_MODEL, False
    return settings.model, True
