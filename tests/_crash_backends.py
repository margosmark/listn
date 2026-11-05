"""Backend factories that crash a WORKER process on purpose.

``listn.transcriber._FACTORY_PATH`` points the isolation worker child at
these, so the SIGSEGV/SIGABRT happen inside the child process — exactly
where a broken CUDA stack would kill listn if inference were not
isolated. Nothing here touches the network or a real recording.

``LISTN_TEST_LOAD_LOG`` may name a file; every factory appends one line
per model build, which is how tests count how many child processes really
loaded a model (one for an explicit failure, two for an auto CPU retry).
"""

from __future__ import annotations

import os
import signal

from listn.transcriber import RawSegment, Transcriber


def _log_load() -> None:
    path = os.environ.get("LISTN_TEST_LOAD_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write("load\n")


def _device_of(knobs) -> str:
    """The device this build asked for; anything but cpu pretends a GPU is up."""
    return "cpu" if str(knobs.get("compute_device") or "auto").strip().lower() == "cpu" else "cuda"


def _make_backend(device: str, segments) -> type:
    """A Transcriber that reports ``device`` and yields ``segments(audio)``."""

    class Worker(Transcriber):
        backend = "faster-whisper"

        def __init__(self, model):
            super().__init__(model)
            self.device = device
            self.device_message = f"device: {device}"

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield from segments(audio)

    return Worker


def roundtrip_factory(model, download=None, **knobs):
    """Control case: never crashes, always the same two lines."""
    _log_load()

    def lines(_audio):
        yield RawSegment(0.0, 1.0, "hello from the worker")
        yield RawSegment(1.0, 2.0, "second line")

    return _make_backend(_device_of(knobs), lines)(model)


def segfault_unless_cpu_factory(model, download=None, **knobs):
    """Loads fine, then SIGSEGVs the moment inference starts (post-load crash)."""
    _log_load()
    device = _device_of(knobs)

    def lines(_audio):
        if device != "cpu":
            os.kill(os.getpid(), signal.SIGSEGV)  # dies here, inside the child
        yield RawSegment(0.0, 1.0, "cpu result one")
        yield RawSegment(1.0, 2.0, "cpu result two")

    return _make_backend(device, lines)(model)


def abort_mid_iteration_factory(model, download=None, **knobs):
    """SIGABRTs after yielding one segment: partial output must be discarded."""
    _log_load()
    device = _device_of(knobs)

    def lines(_audio):
        if device != "cpu":
            yield RawSegment(0.0, 0.5, "partial segment that must never be written")
            os.kill(os.getpid(), signal.SIGABRT)  # mid-generator, in the child
        yield RawSegment(0.0, 1.0, "complete one")
        yield RawSegment(1.0, 2.0, "complete two")

    return _make_backend(device, lines)(model)


def crash_on_load_factory(model, download=None, **knobs):
    """Dies while constructing the model on a GPU attempt."""
    _log_load()
    device = _device_of(knobs)

    class CrashOnLoad(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = device
            self.device_message = f"device: {device}"
            if device != "cpu":
                os.kill(os.getpid(), signal.SIGSEGV)

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "rebuilt on cpu")

    return CrashOnLoad(model)


def always_crash_factory(model, download=None, **knobs):
    """Crashes on every inference attempt, even on plain CPU."""
    _log_load()
    device = _device_of(knobs)

    def lines(_audio):
        os.kill(os.getpid(), signal.SIGSEGV)
        yield  # pragma: no cover — unreachable: the process is already dead

    return _make_backend(device, lines)(model)


def cuda_error_unless_cpu_factory(model, download=None, **knobs):
    """The real-world Python-level CUDA refusal during inference, not a signal."""
    _log_load()
    device = _device_of(knobs)

    def lines(_audio):
        if device != "cpu":
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        yield RawSegment(0.0, 1.0, "recovered on cpu")

    return _make_backend(device, lines)(model)
