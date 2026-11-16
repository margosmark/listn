from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from listn.transcriber import RawSegment, Transcriber


class FakeDefault:
    def __init__(self, device):
        self.device = device


class FakeStream:
    """Feeds a sine wave (or silence) to the callback from a thread."""

    def __init__(self, sd, device, channels, samplerate, dtype, blocksize, callback):
        self.sd = sd
        self.channels = channels
        self.samplerate = samplerate
        self.callback = callback
        self.block = 1024
        self._stop = threading.Event()
        self._thread = None
        self.active = True
        sd.opened.append(self)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t = 0
        while not self._stop.is_set():
            n = np.arange(t, t + self.block)
            data = np.zeros((self.block, self.channels), dtype=np.float32)
            for ch in self.sd.active_channels:
                if ch < self.channels:
                    data[:, ch] = self.sd.amplitude * np.sin(2 * np.pi * 440 * n / self.samplerate)
            t += self.block
            self.callback(data, self.block, None, None)
            time.sleep(self.block / self.samplerate / self.sd.speed)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def close(self):
        pass


class FakeSD:
    """Stand-in for the sounddevice module."""

    class PortAudioError(Exception):
        pass

    def __init__(self, devices, default=(0, 1), output_name=None, reject=None, speed=20.0):
        self.devices = devices
        self.default = FakeDefault(default)
        self.reject = reject or (lambda channels, rate: False)
        self.opened: list[FakeStream] = []
        self.active_channels = [0]
        self.amplitude = 0.3
        self.speed = speed

    def query_devices(self, idx=None):
        if idx is None:
            return [dict(d, index=i) for i, d in enumerate(self.devices)]
        return dict(self.devices[idx], index=idx)

    def query_hostapis(self):
        return [{"name": "Core Audio"}]

    def check_input_settings(self, device, channels, samplerate, dtype):
        if self.reject(channels, samplerate):
            raise self.PortAudioError(f"Invalid number of channels/rate {channels}/{samplerate}")

    def InputStream(self, **kwargs):  # noqa: N802
        return FakeStream(self, **kwargs)


def dev(name, inputs=2, outputs=0, rate=48000.0):
    return {"name": name, "max_input_channels": inputs, "max_output_channels": outputs,
            "default_samplerate": rate, "hostapi": 0}


MAC_DEVICES = [
    dev("MacBook Pro Microphone", inputs=1, rate=48000.0),
    dev("MacBook Pro Speakers", inputs=0, outputs=2),
    dev("BlackHole 2ch", inputs=2, outputs=2),
    dev("Multi-Output Device", inputs=0, outputs=2),
    dev("Aggregate Device", inputs=3, outputs=2, rate=48000.0),
]


class FakeTranscriber(Transcriber):
    backend = "fake"

    def __init__(self, model="tiny"):
        super().__init__(model)
        self.calls = 0

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        self.calls += 1
        duration = audio.size / 16000
        yield RawSegment(0.0, duration / 2, f"chunk {self.calls} first half")
        yield RawSegment(duration / 2, duration, "Thanks for watching!")


@pytest.fixture
def data_dir(monkeypatch):
    """Short directory. macOS pytest temps under /var/folders exceed sun_path."""
    root = Path(tempfile.mkdtemp(prefix="lt", dir="/tmp"))
    path = root / "d"
    monkeypatch.setenv("LISTN_DATA_DIR", str(path))
    yield path
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def fake_sd():
    return FakeSD(MAC_DEVICES, default=(0, 3))


@pytest.fixture
def fake_factory():
    instances = []

    def factory(model, backend, status=None):
        if status:
            status("loading fake model")
        t = FakeTranscriber(model)
        instances.append(t)
        return t

    factory.instances = instances
    return factory


def join_stray_finishers(timeout: float = 5.0) -> None:
    """Wait for any ``listn-finish`` thread still running.

    A test that proves ``stop(wait=True)`` *used* to skip the join leaves the
    finisher behind; joining it here keeps that RED run from racing the
    ``data_dir`` cleanup, so it fails on its own assertions only.
    """
    for thread in threading.enumerate():
        if thread.name == "listn-finish" and thread is not threading.current_thread():
            thread.join(timeout)


@pytest.fixture(autouse=True)
def no_local_llm(monkeypatch):
    """Never load a real MLX model in tests; tests pass a fake ``complete``.

    The env override also pins every ``Settings.load()`` in the suite to the
    extractive backend, so a machine with an operator's Ollama endpoint
    configured can never make listn's test run call it: a test that wants the
    Ollama path builds its own ``Settings`` (or clears the override) and talks
    to a local fake server.
    """
    import listn.ask
    import listn.summary

    monkeypatch.setattr(listn.summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(listn.ask, "mlx_lm_available", lambda: False)
    monkeypatch.setenv("LISTN_LLM_BACKEND", "extractive")
