"""Open input streams robustly and measure input levels.

``sd`` arguments accept the ``sounddevice`` module or a test double with the
same surface (``query_devices``, ``query_hostapis``, ``default``,
``check_input_settings``, ``InputStream``, ``PortAudioError``).
"""

from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass

import numpy as np

from listn import devices as devmod


def _sd():
    import sounddevice as sd

    return sd


@dataclass
class OpenedStream:
    stream: object
    device: devmod.Device
    channels: int
    samplerate: float


class DeviceError(RuntimeError):
    pass


def open_input(selector, callback, sd=None, blocksize: int = 0) -> OpenedStream:
    """Resolve ``selector`` and open a float32 InputStream that the device accepts."""
    sd = sd or _sd()
    try:
        device = devmod.resolve(selector, devmod.list_all(sd))
    except LookupError as exc:
        raise DeviceError(str(exc)) from exc

    errors = []
    for channels, rate in devmod.candidate_stream_configs(device):
        try:
            sd.check_input_settings(device=device.index, channels=channels, samplerate=rate, dtype="float32")
            stream = sd.InputStream(device=device.index, channels=channels, samplerate=rate,
                                    dtype="float32", blocksize=blocksize, callback=callback)
            stream.start()
            return OpenedStream(stream, device, channels, rate)
        except Exception as exc:
            errors.append(f"{channels}ch @ {int(rate)} Hz: {exc}")
    detail = "; ".join(errors[-3:])
    raise DeviceError(
        f"Could not open '{device.name}'. Tried {len(errors)} configurations ({detail}). "
        "If this is an aggregate device, open Audio MIDI Setup and make sure all its sub-devices "
        "share one sample rate and one is set as clock source."
    )


def dbfs(value: float) -> float:
    return -120.0 if value <= 1e-6 else max(-120.0, 20 * math.log10(value))


def measure_levels(selector, seconds: float = 2.0, sd=None) -> dict:
    """Record briefly and report per-channel RMS/peak so users can see signal."""
    sd = sd or _sd()
    blocks: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        blocks.put(indata.copy())

    opened = open_input(selector, cb, sd=sd)
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        opened.stream.stop()
        opened.stream.close()

    chunks = []
    while not blocks.empty():
        chunks.append(blocks.get_nowait())
    data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, opened.channels), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    per_channel = []
    for ch in range(data.shape[1]):
        col = data[:, ch]
        r = float(np.sqrt(np.mean(np.square(col)))) if col.size else 0.0
        p = float(np.max(np.abs(col))) if col.size else 0.0
        per_channel.append({"channel": ch + 1, "rms_db": round(dbfs(r), 1), "peak_db": round(dbfs(p), 1)})
    loudest = max((c["peak_db"] for c in per_channel), default=-120.0)
    if loudest < -60:
        verdict = "silent"
        advice = ("No signal. Play some audio and make sure your system output (or the meeting app's "
                  "speaker) is the Multi-Output Device that includes BlackHole.")
    elif loudest < -35:
        verdict = "quiet"
        advice = "Signal is faint. Raise the playback volume; BlackHole captures at the level macOS sends it."
    else:
        verdict = "ok"
        advice = "Signal detected."
    return {
        "device": opened.device.to_dict(),
        "channels": opened.channels,
        "samplerate": opened.samplerate,
        "seconds": round(data.shape[0] / opened.samplerate, 2) if opened.samplerate else 0,
        "per_channel": per_channel,
        "verdict": verdict,
        "advice": advice,
    }
