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
import threading
import time

from listn.transcriber import (CPU_FALLBACK_MESSAGE, CudaUnavailable,
                               RawSegment, Transcriber)


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

# ---------------------------------------------------------------------------
# S1: workers that never answer. The parent's request bound must fire, the
# caller must get a clean error, and the worker must be gone afterwards.
# ---------------------------------------------------------------------------

def hang_load_factory(model, download=None, **knobs):
    """Never answers the load request (never returns from the factory)."""
    _log_load()
    time.sleep(600)  # pragma: no cover — only reached if the bound never fires


def hang_transcribe_factory(model, download=None, **knobs):
    """Loads fine, then never answers the transcription request."""
    _log_load()
    device = _device_of(knobs)

    class Hang(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = device
            self.device_message = f"device: {device}"

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            time.sleep(600)  # pragma: no cover — bound must fire first
            yield RawSegment(0.0, 1.0, "bound never fired", 0.0)  # pragma: no cover

    return Hang(model)


# ---------------------------------------------------------------------------
# S2/S3: hostile or broken child answers. The parent must adopt only what
# confirms its own device decision.
# ---------------------------------------------------------------------------

def gpu_claim_factory(model, download=None, **knobs):
    """Claims the GPU in its device, its device line AND a status event."""
    _log_load()

    class Claimer(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = "cuda"
            self.device_message = ("device: cuda — using GPU acceleration "
                                   "(CTranslate2 cuda)")
            if download:
                download("device: cuda — using GPU acceleration "
                         "(CTranslate2 cuda)")

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "ran where it was told", 0.0)

    return Claimer(model)


def reclaim_after_pin_factory(model, download=None, **knobs):
    """SIGSEGVs on the GPU attempt; the CPU retry claims cuda anyway (the
    post-pin re-claim: the parent's pin must survive it)."""
    _log_load()
    device = _device_of(knobs)
    if device != "cpu":
        os.kill(os.getpid(), signal.SIGSEGV)

    class Reclaimer(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = "cuda"
            self.device_message = ("device: cuda — using GPU acceleration "
                                   "(CTranslate2 cuda)")

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "cpu work", 0.0)

    return Reclaimer(model)


def injected_note_factory(model, download=None, **knobs):
    """A consistent cpu claim carrying a lying canonical fallback note and a
    GPU device line — neither may be adopted for a request that never fell
    back."""
    _log_load()

    class Injector(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = "cpu"
            self.device_message = "device: cuda — running on GPU"
            self.status_note = CPU_FALLBACK_MESSAGE

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "on cpu", 0.0)

    return Injector(model)


def narrowed_cpu_note_factory(model, download=None, **knobs):
    """The honest control: the probe narrowed `auto` to CPU and says so with
    the canonical note — validation must KEEP this one."""
    _log_load()

    class Narrowed(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = "cpu"
            self.device_message = "device: cpu"
            self.status_note = CPU_FALLBACK_MESSAGE

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "narrowed to cpu", 0.0)

    return Narrowed(model)


def kind_device_oom_factory(model, download=None, **knobs):
    """Raises CudaUnavailable carrying an OOM — kind='device' must not turn
    a runtime fault into the canonical CPU fallback."""
    _log_load()
    raise CudaUnavailable("CUDA out of memory. Tried to allocate 2.00 GiB "
                          "(GPU 0)")


# ---------------------------------------------------------------------------
# S1r: a child that stops reading the request pipe OUTSIDE an active request.
# ---------------------------------------------------------------------------

def freeze_between_factory(model, download=None, **knobs):
    """Answers the load, then freezes itself — a child that stops reading
    the request pipe while it is *not* inside any request (SIGSTOP /
    wedged native thread / cgroup freeze equivalent)."""
    _log_load()
    device = _device_of(knobs)

    def freeze():
        time.sleep(1.5)          # let the load response reach the parent first
        os.kill(os.getpid(), signal.SIGSTOP)

    threading.Thread(target=freeze, daemon=True).start()
    return _make_backend(device, lambda _audio: iter(
        [RawSegment(0.0, 1.0, "never runs", 0.0)]))(model)


# ---------------------------------------------------------------------------
# S2r: hostile answers on the default device (`auto`), one channel each.
# ---------------------------------------------------------------------------

def auto_note_factory(model, download=None, **knobs):
    """`auto`: fabricates the canonical CPU-fallback note with no failure
    of any kind — both channels: as the response's status_note and as a
    status event — plus a consistent cpu claim (gate C6)."""
    _log_load()
    if download:
        download(CPU_FALLBACK_MESSAGE)   # hostile: as a status event too

    class NoteInjector(Transcriber):
        backend = "faster-whisper"

        def __init__(self, name):
            super().__init__(name)
            self.device = "cpu"
            self.device_message = "device: cpu"
            self.status_note = CPU_FALLBACK_MESSAGE

        def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
            yield RawSegment(0.0, 1.0, "on cpu", 0.0)

    return NoteInjector(model)


def auto_status_factory(model, download=None, **knobs):
    """`auto`: hostile status event only — the response itself claims
    nothing (device stays `auto`), so only the status channel is dirty
    (gate C7)."""
    _log_load()
    if download:
        download("device: cuda — using GPU acceleration (CTranslate2 cuda)")
    device = str(knobs.get("compute_device") or "auto") or "auto"
    return _make_backend(device, lambda _audio: iter(
        [RawSegment(0.0, 1.0, "ok", 0.0)]))(model)

