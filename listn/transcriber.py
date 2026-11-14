"""Local Whisper backends behind one small interface.

* ``faster-whisper`` (default): CTranslate2 int8 on CPU, built-in Silero VAD,
  streams segments so long files report progress. Roughly 4x faster than
  openai-whisper on CPU with a fraction of the memory, and no PyTorch.
* ``mlx``: Apple's MLX port, runs on the Apple Silicon GPU. Optional extra.
* ``openai-whisper``: the original reference implementation. Optional extra.

Models are downloaded once from Hugging Face (or OpenAI's CDN for
openai-whisper) and loaded from the local cache afterwards. A cached load
sets ``HF_HUB_OFFLINE`` first, so it does not contact the Hub. The
unauthenticated-request warning only happens on that first download, and
only when ``HF_TOKEN`` is unset.

Where the default backend runs is the ``compute_device`` setting: ``auto``
(the default — main's shipped behaviour: GPU when this machine's CUDA *or
ROCm* runtime is really usable, CPU otherwise, said out loud), ``cpu``
(never touch the GPU), ``cuda`` (require NVIDIA and fail with a readable
message instead of falling back) or ``rocm`` (require AMD's HIP build the
same way). Which of
those this machine can actually run — and the exact remedy for the piece
that is missing — is answered by the vendor-neutral probe in
:mod:`listn.compute_backend` (:func:`probe_capabilities`); see
:func:`resolve_device` for the selection policy.

Model load and inference for the faster-whisper backend run in a child
process (:class:`IsolatedTranscriber`, worker: ``listn.inference_worker``)
so a native crash — SIGSEGV/SIGABRT from the CUDA or HIP path — can never
kill the caller or leave a partial transcript behind. ``auto`` then falls
back to CPU truthfully; explicit ``cuda``/``rocm`` fails with an actionable
message.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

from listn import compute_backend
from listn.audio import TARGET_RATE, rms
from listn.compute_backend import (  # re-exported: callers and the isolation
    # worker import these from this module.
    CUDA_RUNTIME_LIBS, Capability, CudaUnavailable, DeviceUnavailable, Plan,
    format_report, probe_capabilities,
)
from listn.hub import load_local_or_download

log = logging.getLogger(__name__)

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

SPEECH_RMS_GATE = 0.003

CPU_FALLBACK_MESSAGE = "CUDA libraries unavailable — transcribing on CPU"

# CTranslate2 names the library it could not load when a GPU is visible but the
# CUDA user-space libraries are absent (libcublas.so.12 on a default CachyOS):
# "Library libcublas.so.12 is not found or cannot be loaded". Any message here
# must also carry one of these library/device markers.
_CUDA_ERROR_MARKERS = ("cuda", "cublas", "dnn", "nccl")

# Only load/availability phrasing counts as "unavailable". Every other member
# of the CUDA error family is a fault on a working stack and must propagate:
#   cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED   (GPU OOM)
#   CUDA failed with error out of memory                   (GPU OOM)
#   CUDA failed with error an illegal memory access ...    (device fault)
#   NCCL failed with error 1                               (runtime failure)
# Strings verified against the shipped libctranslate2 4.8.3.
_AVAILABILITY_MARKERS = (
    "is not found",            # "Library libcublas.so.12 is not found ..."
    "cannot be loaded",        # ... "or cannot be loaded"
    "cannot open shared object",  # dlopen of libcuda.so.1 itself
    "driver version is insufficient",
    "stub library",            # "CUDA driver is a stub library"
    "busy or unavailable",
    "no cuda-capable device",  # no GPU at all
    "api call is not supported in the installed cuda driver",
    "unsupported display driver / cuda driver combination",
)

# Belt and braces: a message that is both availability-shaped and a runtime
# fault is a fault — never recover it as "CUDA libraries unavailable".
_RUNTIME_FAILURE_MARKERS = (
    "out of memory",
    "alloc_failed",
    "illegal memory",
    "failed with status",
)


def _is_cuda_unavailable(exc: BaseException) -> bool:
    """True only for load/availability CUDA RuntimeErrors worth CPU recovery.

    A GPU OOM or device fault on a healthy stack is *not* "CUDA libraries
    unavailable": it must propagate (and stay labelled as itself) rather than
    relabel the job, the live status and the cache key.
    """
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    if not any(marker in message for marker in _CUDA_ERROR_MARKERS):
        return False
    if any(marker in message for marker in _RUNTIME_FAILURE_MARKERS):
        return False
    return any(marker in message for marker in _AVAILABILITY_MARKERS)


def _is_runtime_failure_text(text: str) -> bool:
    """True when ``text`` carries a runtime-fault marker (OOM, NCCL, …).

    The same belt-and-braces rule the availability classifier applies: a
    message that names a runtime fault is a fault whatever ``kind`` the
    isolation child wrapped it in (S3 — a hostile child must not be able
    to relabel an OOM as "CUDA libraries unavailable").
    """
    message = str(text).lower()
    return any(marker in message for marker in _RUNTIME_FAILURE_MARKERS)


def _cuda_device_count() -> int:
    """``ctranslate2.get_cuda_device_count()``, or -1 when the probe fails.

    Only settles the obvious case: a count of 0 means ``device="auto"`` can
    resolve to nothing but CPU, so a CUDA-shaped error has no CPU rebuild to
    offer. A count above 0 proves nothing about the libraries — the broken
    laptop reports 1 with none installed — so the real gate is the error
    itself, raised at op time.
    """
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return -1

_ALWAYS_HALLUCINATED = {
    "thanks for watching", "thank you for watching", "thanks for watching and please subscribe",
    "please subscribe", "like and subscribe", "subscribe to my channel", "blank audio",
    "music", "applause", "silence", "inaudible",
}
_FILLERS = {"you", "thank you", "thanks", "bye", "bye bye", "goodbye", "okay", "oh", "uh", "um", "hmm", "mm", "ah", "so"}
_HALLUCINATED_FRAGMENTS = ("amara.org", "subtitles by", "transcribed by", "captions by")


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']|_", " ", text.lower()).split())


def is_hallucination(text: str, no_speech_prob: float = 0.0, quiet: bool = False) -> bool:
    """Whisper invents stock phrases on silence or noise. Drop those.

    Short real words ("OK", "No", "Yes") are kept; generic fillers are only
    dropped when the model itself thinks there's no speech or the audio was
    very quiet.
    """
    lower = text.lower()
    if any(frag in lower for frag in _HALLUCINATED_FRAGMENTS):
        return True
    norm = _normalize(text)
    if not norm:
        return True
    if norm in _ALWAYS_HALLUCINATED:
        return True
    if norm in _FILLERS and (no_speech_prob >= 0.4 or quiet):
        return True
    return False


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


def available_backends() -> dict[str, bool]:
    out = {}
    for name, mod in (("faster-whisper", "faster_whisper"), ("mlx", "mlx_whisper"), ("openai-whisper", "whisper")):
        try:
            __import__(mod)
            out[name] = True
        except Exception:
            out[name] = False
    return out


# The child-process probe itself lives in compute_backend; this module-level
# name is the seam tests patch to drive timeouts, signal deaths and library
# failures through a real child (tests/test_transcriber.py).
_PROBE_SCRIPT = compute_backend.PROBE_SCRIPT


def cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can see; query in a child process.

    A native driver/runtime bug becomes a quiet zero rather than killing listn.
    The actionable reason is emitted by :func:`cuda_runtime_ready` when a GPU
    device was requested explicitly.
    """
    result, _reason = compute_backend.run_probe(
        script=_PROBE_SCRIPT, sonames=(), timeout=30.0)
    return int(result.get("devices", 0)) if result else 0


def cuda_runtime_ready(sonames: tuple[str, ...] = CUDA_RUNTIME_LIBS,
                       timeout: float = 30.0) -> tuple[bool, str]:
    """Can the CUDA runtime libraries CTranslate2 needs actually load?

    Returns ``(ready, detail)``. The native loader and CTranslate2 device
    query run in a child process: either can terminate the process without a
    Python exception. A child that dies by signal, times out, or prints
    nothing means "not ready" — returned as a reason, never as a crash.
    """
    result, error = compute_backend.run_probe(
        script=_PROBE_SCRIPT, sonames=sonames, timeout=timeout)
    if error or result is None:
        return False, error or "the CUDA library probe returned no usable answer"
    missing = result.get("missing", [])
    if missing:
        return False, "; ".join(missing)
    if result.get("device_error"):
        return False, f"CTranslate2 could not query the CUDA driver ({result['device_error']})"
    return True, ""


def _is_cuda_error(exc: BaseException) -> bool:
    """True when an exception is CTranslate2 refusing (or failing) to use the GPU."""
    text = str(exc).lower()
    return any(word in text for word in
               ("cuda", "cublas", "cudnn", "nccl", "libcuda", "gpu", "rocm"))


def _resolve_plan(requested: str) -> Plan:
    """The full selection plan: device, truthful note, and GPU truth.

    Delegates to :func:`listn.compute_backend.resolve` with this module's
    probe seams, so tests that pin ``cuda_device_count``/``cuda_runtime_ready``
    (and ``compute_backend.rocm_capability``) drive the real policy.
    """
    return compute_backend.resolve(
        requested, cuda_count=cuda_device_count, cuda_ready=cuda_runtime_ready,
        rocm=compute_backend.rocm_capability)


def resolve_device(requested: str) -> tuple[str, str | None]:
    """Turn the ``compute_device`` setting into a WhisperModel device.

    Returns ``(device, note)``. ``note`` explains a CPU fallback (or a ROCm
    resolution) when there is one and is ``None`` otherwise. Raises
    :class:`CudaUnavailable` when ``cuda``/``rocm`` was requested by name and
    this machine cannot run it — checked before any model is built, so the
    answer is one readable message with the missing piece and the remedy
    rather than a failure inside CTranslate2.

    ``auto`` means GPU only when a CUDA device is visible *and* the runtime
    libraries load (or a ROCm build with a visible device); otherwise it
    stays on the historical behaviour (``auto``, which CTranslate2 resolves
    to CPU) or pins ``cpu`` when a GPU is visible but unusable. The full
    plan (including which GPU backend resolved, for compute-type mapping)
    is :func:`_resolve_plan`.
    """
    plan = _resolve_plan(requested)
    return plan.device, plan.note


def _reported_device(model: object, asked: str) -> str:
    """The device CTranslate2 says the loaded model is actually running on."""
    reported = getattr(getattr(model, "model", None), "device", None)
    return str(reported) if reported else asked


def _loud_device_failure(want: str, exc: BaseException) -> BaseException:
    """The readable, actionable failure for an explicit ``cuda``/``rocm`` ask.

    The message carries the missing piece and the exact remedy, so the user
    never sees a bare CTranslate2 refusal. Only used on the in-process path;
    the isolation parent builds the same messages in ``_on_cuda_failure``.
    """
    if want == "rocm":
        return CudaUnavailable(
            "compute_device=rocm was requested, but the model could not be "
            f"built on the ROCm GPU: {exc}. Check the ROCm setup "
            "(docs/GPU_LINUX.md), or run on CPU with --compute-device cpu.")
    return CudaUnavailable(
        "compute_device=cuda was requested, but the model could not be "
        f"built on the GPU: {exc}. Install the CUDA 12 libraries "
        "(docs/GPU_LINUX.md), or run on CPU with --compute-device cpu.")


class Transcriber:
    backend = "base"
    # The device the model is actually built for; only faster-whisper sets it
    # ("auto" as configured, "cpu" after a CUDA-library fallback). Others keep
    # the default and are keyed without a device.
    device: str | None = None
    # Set by a backend that had to abandon its configured device (the CPU
    # rebuild after CUDA libraries turned out to be unusable). Shown as the
    # job's stage so the fallback is visible instead of silent. This is the
    # canonical status surface: engine.py reads status_note.
    status_note: str | None = None
    # Set by backends that know where they are running; reported verbatim.
    # Richer detail beside status_note (probe reason, resolved device) —
    # additive, never a replacement for it.
    device_message: str | None = None

    def __init__(self, model: str):
        self.model_name = model

    def segments(self, audio: np.ndarray, language: str | None = None,
                 prompt: str | None = None, vad: bool = True, hotwords: str | None = None) -> Iterator[RawSegment]:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: str | None = None,
                   vad: bool = True, offset: float = 0.0, hotwords: str | None = None,
                   progress: Callable[[float], None] | None = None) -> list[RawSegment]:
        """Transcribe and filter. Segment times are shifted by ``offset``."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        quiet = rms(audio) < SPEECH_RMS_GATE * 2
        duration = audio.size / TARGET_RATE
        out = []
        for seg in self.segments(audio, language, prompt, vad, hotwords):
            text = seg.text.strip()
            if progress and duration > 0:
                progress(min(1.0, seg.end / duration))
            if not text or is_hallucination(text, seg.no_speech_prob, quiet):
                continue
            out.append(RawSegment(seg.start + offset, seg.end + offset, text, seg.no_speech_prob))
        if progress:
            progress(1.0)
        return out


class FasterWhisper(Transcriber):
    backend = "faster-whisper"
    # What CTranslate2 reports after the model loads, not what we asked for.
    device = "cpu"

    def __init__(self, model: str, download: Callable[[str], None] | None = None, *,
                 compute_type: str = "int8", cpu_threads: int | None = None,
                 beam_size: int = 5, compute_device: str = "auto"):
        super().__init__(model)
        from faster_whisper import WhisperModel

        # Defaults stay byte-identical to the old hard-coded call: int8, no
        # cpu_threads passed (CTranslate2 picks its own count), and "auto" on
        # any machine without a usable GPU runtime. A GPU build left at the
        # default compute type gets float16 via compute_type_for().
        self.beam_size = beam_size
        plan = _resolve_plan(compute_device)
        device, note = plan.device, plan.note
        notes = [note] if note else []
        want = (compute_device or "auto").strip().lower() or "auto"
        # The probe narrows `auto` before any build: when it pins CPU because
        # the GPU runtime is unusable, the canonical status note is main's
        # exact library-unavailable message.
        if want == "auto" and device == "cpu" and note:
            self.status_note = CPU_FALLBACK_MESSAGE
        base_compute_type = compute_type
        compute_type = compute_backend.compute_type_for(
            compute_type, device, gpu_active=plan.gpu_active)
        kwargs: dict = dict(device=device, compute_type=compute_type)
        if cpu_threads:
            kwargs["cpu_threads"] = int(cpu_threads)
        explicit_gpu = want in ("cuda", "rocm")
        gpu_label = "ROCm" if plan.kind == "rocm" else "CUDA"
        failures: list[Exception] = []
        # ``download`` is get_transcriber's status callback: it reports the
        # one-time model download and the CPU fallback alike.
        self._status = download

        def load(dev: str):
            """Build the model on ``dev``.

            A device failure on the cached attempt is recorded in ``failures``
            so the download path never runs: a broken GPU is not a missing
            model, and retrying it over the network would both waste a round
            trip and claim a download that is not happening.
            """
            kw: dict = dict(
                kwargs, device=dev,
                compute_type=compute_backend.compute_type_for(
                    base_compute_type, dev,
                    gpu_active=plan.gpu_active and dev != "cpu"))
            failures.clear()

            def cached():
                try:
                    return WhisperModel(model, local_files_only=True, **kw)
                except Exception as exc:
                    if _is_cuda_error(exc):
                        failures.append(exc)
                    raise

            def fetched():
                if failures:
                    raise failures[0]
                return WhisperModel(model, **kw)

            def notify():
                if download and not failures:
                    download(f"Downloading Whisper '{model}' (one time)...")

            return load_local_or_download(cached, fetched,
                                          on_download=notify if download else None)

        # Kept for a segments-time rebuild (the in-process path only).
        self._load = load

        try:
            self._model = load(device)
        except Exception as exc:
            if _IN_WORKER:
                # The isolation child reports every error; the parent owns
                # the classification (F1: the classifier lives in the parent).
                raise
            if not _is_cuda_unavailable(exc):
                # F1: runtime faults (OOM, illegal memory access, NCCL) and
                # unrelated errors propagate as themselves — no CPU rebuild,
                # never relabelled as "CUDA libraries unavailable".
                raise
            if explicit_gpu:
                # explicit cuda/rocm keeps the branch behaviour: one loud,
                # actionable failure with the missing piece and the remedy.
                raise _loud_device_failure(want, exc) from exc
            if want == "cpu":
                raise  # nowhere to fall back to
            # auto: a broken GPU must not stop transcription — one rebuild,
            # said out loud with main's exact message.
            log.warning("Model build on CUDA failed (%s); rebuilding on CPU", exc)
            self.status_note = CPU_FALLBACK_MESSAGE
            notes.append(f"{gpu_label} unusable ({exc}); running on CPU")
            device = "cpu"
            self._model = load("cpu")

        self.device = _reported_device(self._model, device)
        # Truthful, user-facing: what the model reports, plus any fallback
        # reason. get_transcriber passes this to the status callback on every
        # call, cache hit or not.
        self.device_message = f"device: {self.device}"
        if notes:
            self.device_message += " — " + "; ".join(notes)

    def _rebuild_on_cpu(self):
        """Rebuild once on CPU after a CUDA-library failure; never loops.

        Only the in-process path reaches this: the isolation child reports
        the fault and the parent applies the policy instead. The weights are
        already on disk, so this load is local-only.
        """
        self.device = "cpu"
        self.status_note = CPU_FALLBACK_MESSAGE
        self._model = self._load("cpu")
        return self._model

    def _announce_fallback(self) -> None:
        """Tell the status callback (UI/tray) why transcription moved to CPU."""
        if self.status_note and self._status:
            self._status(self.status_note)

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        kwargs: dict = dict(
            language=language, initial_prompt=prompt, vad_filter=vad,
            beam_size=self.beam_size, condition_on_previous_text=False,
        )
        if hotwords:
            kwargs["hotwords"] = hotwords
        yielded = 0
        while True:
            try:
                try:
                    segs, _info = self._model.transcribe(audio, **kwargs)
                except TypeError:
                    kwargs.pop("hotwords", None)
                    segs, _info = self._model.transcribe(audio, **kwargs)
                for s in segs:
                    yielded += 1
                    yield RawSegment(s.start, s.end, s.text, getattr(s, "no_speech_prob", 0.0))
                return
            except RuntimeError as exc:
                # One CPU rebuild per model, and only before the first segment
                # went out (restarting mid-stream would repeat it). A second
                # CUDA-shaped error — or anything else — propagates instead
                # of looping. In the isolation child nothing is recovered
                # here: the fault is reported and the parent classifies it.
                if (_IN_WORKER or self.device == "cpu" or yielded
                        or not _is_cuda_unavailable(exc)):
                    raise
                log.warning("Transcription on CUDA failed (%s); rebuilding on CPU", exc)
                self._rebuild_on_cpu()
                self._announce_fallback()


class _WindowedBackend(Transcriber):
    """Backends that don't stream: feed long audio in windows for progress."""

    window_seconds = 120

    def _run(self, audio, language, prompt) -> list[dict]:
        raise NotImplementedError

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        step = self.window_seconds * TARGET_RATE
        carry = prompt
        for start in range(0, audio.size, step):
            piece = audio[start:start + step]
            if rms(piece) < SPEECH_RMS_GATE / 2:
                continue
            base = start / TARGET_RATE
            for d in self._run(piece, language, carry):
                yield RawSegment(base + float(d["start"]), base + float(d["end"]), str(d["text"]),
                                 float(d.get("no_speech_prob", 0.0)))
                carry = str(d["text"])[-200:]


class MlxWhisper(_WindowedBackend):
    backend = "mlx"

    def __init__(self, model: str, download: Callable[[str], None] | None = None):
        super().__init__(model)
        import mlx_whisper
        from huggingface_hub import snapshot_download

        repo = MLX_REPOS[model]

        def cached():
            return snapshot_download(repo, local_files_only=True)

        def fetched():
            return snapshot_download(repo)

        self._path = load_local_or_download(
            cached,
            fetched,
            on_download=(lambda: download(f"Downloading Whisper '{model}' for MLX (one time)...")) if download else None,
        )
        self._mlx = mlx_whisper

    def _run(self, audio, language, prompt):
        result = self._mlx.transcribe(audio, path_or_hf_repo=self._path, language=language,
                                      initial_prompt=prompt, verbose=None,
                                      condition_on_previous_text=False)
        return result.get("segments", [])


class OpenAIWhisper(_WindowedBackend):
    backend = "openai-whisper"

    def __init__(self, model: str, download: Callable[[str], None] | None = None):
        super().__init__(model)
        import whisper

        if download:
            download(f"Loading openai-whisper '{model}' (downloads once if missing)...")
        self._model = whisper.load_model(model)

    def _run(self, audio, language, prompt):
        result = self._model.transcribe(audio, language=language, initial_prompt=prompt,
                                        fp16=False, verbose=None, condition_on_previous_text=False)
        return result.get("segments", [])


_BACKENDS = {"faster-whisper": FasterWhisper, "mlx": MlxWhisper, "openai-whisper": OpenAIWhisper}
# Keyed by (backend, model, compute_type, cpu_threads, beam_size, isolate,
# device): a knob change reloads the model instead of reusing one built with
# the old settings, and a build that fell back to CPU is only ever served
# from its cpu key — never as the configured (auto) one.
_cache: dict[tuple, Transcriber] = {}
_cache_lock = threading.Lock()


def _build_backend(name: str, model: str, knobs: dict,
                   download: Callable[[str], None] | None = None) -> Transcriber:
    """Build backend ``name`` in THIS process.

    Shared by the in-process path and by the isolation worker child, so
    both construct models through exactly the same code — device
    resolution, the CPU fallback on a broken CUDA runtime and the download
    messaging all behave identically wherever they run. Whether that code
    is isolated in a child process is the caller's decision; this function
    never spawns anything.
    """
    try:
        return _BACKENDS[name](model, download=download, **knobs)
    except ImportError as exc:
        extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
        hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
        raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc


# --- Process isolation for native inference ---------------------------------
# CTranslate2's CUDA path can end the process with SIGSEGV/SIGABRT from
# native code; no Python exception handler in this process can catch that.
# IsolatedTranscriber therefore runs the COMPLETE model load + inference
# operation in a child process (listn.inference_worker) and only hands a
# result back once it is complete. Callers — engine jobs, live sessions —
# write to the library only after that answer arrives, so a native crash
# can never kill listn or leave a partial/duplicate transcript behind.

# Test seam: "module:attr" factory the worker builds instead of the real
# backend, so tests can kill a real child process on purpose. Production
# leaves it None and the worker uses the registered backend.
_FACTORY_PATH: str | None = None
# True inside the isolation worker child (listn.inference_worker.main sets
# it). The child never recovers a device fault itself: it reports the error
# and the parent's classifier (the F1 policy) decides. In-process builds
# recover on their own, because there is no parent to ask.
_IN_WORKER = False


class _Retry:
    """Policy answer: restart the worker pinned to CPU."""


_RETRY = _Retry()


class _ChildDied(RuntimeError):
    """The worker closed its response pipe without answering."""

    def __init__(self, detail: str, phase: str | None = None):
        super().__init__(detail)
        self.detail = detail
        self.phase = phase


class _CudaFailure(RuntimeError):
    """The worker answered ``kind="cuda"``/``kind="device"``.

    ``device`` is True when the child raised :class:`CudaUnavailable`
    itself — a refusal the probe issued before any model build, so it is
    an availability answer by construction and keeps its type instead of
    being re-read (and demoted) by the message classifier.
    """

    def __init__(self, detail: str, device: bool = False):
        super().__init__(detail)
        self.detail = detail
        self.device = device


class _BackendError(RuntimeError):
    """The worker answered ``kind="error"``: a normal Python failure.

    The worker itself is healthy, so this does not restart it.
    """


# --- request bounds (S1) ----------------------------------------------------
# Every worker request is bounded: the parent never waits on the child
# forever. The bound is silence-based — any frame from the child (a status
# line, a progress tick, the final answer) resets the deadline — so a slow
# but answering child is never killed, while a child that stops answering
# entirely (a hung CUDA kernel, a wedged factory) is terminated and reaped
# within the bound and the request fails with a clean error instead of
# blocking the caller, the engine's single job-worker thread, or close().
# Silence alone is not enough (S1c): a child that keeps TALKING without
# ever finishing (a backend looping in progress()) resets that deadline
# forever, so every request also carries a hard total wall cap below.
#
# The bound also covers writing the request (S1r): a child that stopped
# reading the request pipe *outside* an active request (wedged native
# thread after load, cgroup freeze, SIGSTOP) blocks `os.write` on any
# payload larger than the 64 KB pipe buffer with no deadline armed — the
# frames are therefore written by a helper thread joined against a
# deadline armed before the header goes out, so such a child is stopped
# too instead of wedging the caller in the send path.
#
# load gets a much longer bound than inference because a first-run model
# download is silent for as long as it takes; engine file jobs stream
# progress frames, and live chunks are short, so inference never needs it.
_LOAD_IDLE_TIMEOUT = 1800.0     # seconds a load/download may stay silent
_REQUEST_IDLE_TIMEOUT = 120.0   # seconds an inference may stay silent
# S1c: a hard wall cap for ONE request, on top of the silence bound. Any
# frame from the child resets the silence clock above, so a backend that
# loops in progress() forever is never "silent" and that bound alone can
# never stop it — but it can never push past `started + cap` either, and
# is then stopped with one clean _ChildDied. Deliberately generous
# (hours): a legitimately long transcription must never hit it. Read from
# the module global at call time so tests can monkeypatch it small.
_REQUEST_MAX_TOTAL = 6 * 3600.0  # seconds one request may run in total
# close() waits at most this long for an in-flight request's lock before it
# stops the worker anyway (atexit and eviction must not wedge on a hang).
_CLOSE_LOCK_TIMEOUT = 10.0

# A child status/device line that claims GPU use. Checked against the
# parent's own device decision (S2): a worker that was told `cpu` — or
# pinned to it after a crash — can never announce GPU acceleration.
# S8: positive form — any line naming a GPU device counts as a claim, so
# a phrasing the old denylist missed ("using the GPU", "transcribing on
# GPU", "on the GPU") cannot slip past it; false positives are accepted,
# this channel also carries download text. The canonical CPU-fallback
# note deliberately does NOT match: it names CUDA as the reason the
# build is on CPU — it claims CPU, not a GPU, and the F1 marker
# semantics must not flip.
_GPU_CLAIM = re.compile(
    r"device:\s*(?:cuda|rocm|gpu)|"
    r"\b(?:rocm|gpu|hip)\b|"
    r"\bcuda\b(?! libraries unavailable)",
    re.IGNORECASE)

# The one other honest message that legitimately names a GPU device: the
# probe's unavailability reason (`compute_backend.resolve` narrows
# auto -> cpu with it, and it rides into the child's device_message). It
# reports WHY the build is on CPU, it does not claim GPU use, so it is
# stripped before the scan — but only that exact sentence: anything a
# child appends around it is still scanned, so wrapping a claim in the
# reason does not hide it. _claims_gpu() applies both exemptions.
_PROBE_REASON = re.compile(
    r"CUDA is visible but the CUDA runtime cannot be used \([^)]*\)"
    r"(?:; see docs/GPU_LINUX\.md)?",
    re.IGNORECASE)


def _claims_gpu(text: str) -> bool:
    """True when ``text`` claims GPU use (S2/S8).

    The two honest CPU-truth messages that name a GPU device — the
    canonical fallback note and the probe's unavailability reason — are
    removed first; everything else in the line is scanned, so a claim
    wrapped around them still matches.
    """
    remainder = _PROBE_REASON.sub(" ", str(text))
    if CPU_FALLBACK_MESSAGE in remainder:
        remainder = remainder.replace(CPU_FALLBACK_MESSAGE, " ")
    return bool(_GPU_CLAIM.search(remainder))

# The ``device: <name>`` token a device message opens with, if any.
_DEVICE_TOKEN = re.compile(r"device:\s*([A-Za-z0-9_-]+)")


def _drain_stderr(stream, lines: deque) -> None:
    """Keep the worker's stderr flowing (so it can never block) and tail it."""
    try:
        for raw in iter(stream.readline, b""):
            lines.append(raw.decode("utf-8", "replace").rstrip())
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


# Queue sentinel: the worker's response pipe closed (child gone). Pumped
# once when the reader thread sees EOF, and the eof flag covers waiters
# that arrive after it was consumed.
_EOF = object()


class _WorkerSession:
    """One worker child process and the two pipes that talk to it.

    The response pipe is drained by a dedicated reader thread into a queue,
    so no caller ever blocks on pipe I/O itself (S1): ``request`` only ever
    waits on the queue, with a deadline. The write side is bounded too
    (S1r): frames go out through a helper thread joined against a deadline
    armed before the header, so a child that stopped reading the request
    pipe cannot block the caller in ``os.write`` either.
    """

    def __init__(self, proc, to_child: Connection, from_child: Connection,
                 stderr_lines: deque):
        self.proc = proc
        self._to = to_child
        self._from = from_child
        self._stderr = stderr_lines
        # Bounded: a child that spams frames while nobody is asking gets
        # pipe backpressure instead of growing this queue without limit
        # (the queue is only drained by an active request).
        self._inbox: queue.Queue = queue.Queue(maxsize=1024)
        self._eof = threading.Event()
        threading.Thread(target=self._pump, args=(self._from,),
                         name="worker-response-reader", daemon=True).start()

    def _pump(self, stream: Connection) -> None:
        """Move every frame off the response pipe so nobody blocks on I/O.

        Ends only when the pipe does — the child died, was killed, or the
        parent closed the session — and then marks the session eof so later
        waiters fail fast instead of serving a stale queue.
        """
        try:
            while True:
                self._inbox.put(stream.recv_bytes())
        except Exception:  # noqa: BLE001 - EOF, closed pipe, torn-down fd
            pass
        finally:
            self._eof.set()
            self._inbox.put(_EOF)

    @classmethod
    def spawn(cls) -> "_WorkerSession":
        req_r, req_w = os.pipe()   # parent writes requests, child reads
        res_r, res_w = os.pipe()   # child writes responses, parent reads
        env = dict(os.environ)
        # Make `python -m listn.inference_worker` resolvable no matter the cwd.
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "listn.inference_worker",
                 str(req_r), str(res_w)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, pass_fds=(req_r, res_w),
                env=env, close_fds=True)
        except BaseException:
            for fd in (req_r, req_w, res_r, res_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        os.close(req_r)
        os.close(res_w)
        lines: deque[str] = deque(maxlen=80)
        threading.Thread(target=_drain_stderr, args=(proc.stderr, lines),
                         daemon=True).start()
        return cls(proc, Connection(req_w), Connection(res_r), lines)

    def death_detail(self) -> str:
        """How the worker died, for a message a user can act on."""
        proc = self.proc
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return "stopped responding"
        if code is None:
            return "stopped responding"
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
            return f"was killed by {name}"
        if code > 0:
            tail = self._stderr[-1] if self._stderr else ""
            detail = f"exited with code {code}"
            return f"{detail}: {tail}" if tail else detail
        return "exited without answering"

    def request(self, header: dict, payload=None, *, on_status=None,
                on_progress=None, timeout: float | None = None) -> dict:
        """Send one request and read events until its final answer.

        A worker that died mid-request surfaces as :class:`_ChildDied` —
        never as a partial result, because the final answer is the only
        thing this method returns.

        Every request is bounded (S1/S1r). ``timeout`` caps how long the
        child may stay silent: each frame from it resets the clock, so a
        slow but answering worker is never killed, while one that stops
        answering entirely is terminated and reaped within the bound of
        its last frame and the request fails with a clean
        :class:`_ChildDied` — never an indefinite block of the caller,
        the engine's job-worker thread, or ``close()``. The bound starts
        before the request is written, so a child that stopped reading
        the request pipe is stopped too instead of blocking the caller in
        ``os.write`` with no deadline armed. When not given, the bound
        follows the op (``load`` includes first-run downloads, so it is
        the longer one).

        Frames reset that silence clock but can never push the request
        past ``_REQUEST_MAX_TOTAL`` (S1c): a child that keeps talking
        without ever finishing — a backend looping in ``progress()``
        forever — is stopped with one clean :class:`_ChildDied` at the
        cap instead of resetting the silence deadline indefinitely.
        """
        if timeout is None:
            timeout = (_LOAD_IDLE_TIMEOUT if header.get("op") == "load"
                       else _REQUEST_IDLE_TIMEOUT)
        # S1c: the total wall cap covers the whole request, send included.
        started = time.monotonic()
        try:
            frames: list = [json.dumps(header).encode("utf-8")]
            if payload is not None:
                view = payload if isinstance(payload, (bytes, bytearray)) \
                    else memoryview(payload).cast("B")
                frames.append(view)
        except (OSError, ValueError) as exc:
            raise _ChildDied(self.death_detail()) from exc
        # S1r: the deadline that bounds this request starts BEFORE the
        # request is written, not after — see _send_bounded. The total
        # cap bounds the send too, so it can never be outrun there.
        self._send_bounded(frames, min(timeout, _REQUEST_MAX_TOTAL))
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            # S1c: any frame resets the silence clock below, but no frame
            # can push the request past `started + _REQUEST_MAX_TOTAL` —
            # a child looping in progress() forever is stopped here.
            if now - started >= _REQUEST_MAX_TOTAL:
                raise self._max_total_death()
            remaining = deadline - now
            if remaining <= 0:
                raise self._timeout_death(timeout)
            # Never sleep past the total cap either: wake up to re-check it.
            remaining = min(remaining, started + _REQUEST_MAX_TOTAL - now)
            try:
                raw = self._inbox.get(timeout=remaining)
            except queue.Empty:
                if self._eof.is_set():
                    raise _ChildDied(self.death_detail())
                continue  # silence bound or total cap expired: re-check both
            if raw is _EOF:
                raise _ChildDied(self.death_detail())
            deadline = time.monotonic() + timeout  # any frame resets the clock
            try:
                message = json.loads(raw)
            except ValueError:
                raise _ChildDied("returned an unreadable answer") from None
            if "ok" in message:
                return message
            event = message.get("event")
            if event == "status" and on_status:
                on_status(message.get("message"))
            elif event == "progress" and on_progress:
                on_progress(float(message.get("value") or 0.0))

    def _send_bounded(self, frames, timeout: float) -> None:
        """Write every frame of one request within ``timeout`` seconds (S1r).

        ``send_bytes`` blocks in ``os.write`` once the payload outgrows the
        pipe buffer (64 KB), and a child that stopped reading the request
        pipe *outside* an active request — a wedged native thread after
        load, a cgroup freeze, SIGSTOP — would hold the caller (and the
        lock it owns) forever with no deadline armed: the silence deadline
        was only ever armed after the send. The frames are therefore
        written by a helper thread joined against a deadline armed at
        entry, and on expiry the child is stopped (which fails the blocked
        write with EPIPE so the helper can unwind) and the request fails
        with one clean error. The read side keeps its silence deadline,
        unchanged.
        """
        deadline = time.monotonic() + timeout
        outcome: dict[str, BaseException] = {}

        def sender() -> None:
            try:
                for frame in frames:
                    self._to.send_bytes(frame)
            except BaseException as exc:  # noqa: BLE001 - reported to caller
                outcome["exc"] = exc

        helper = threading.Thread(target=sender, name="worker-request-send",
                                  daemon=True)
        helper.start()
        helper.join(max(0.0, deadline - time.monotonic()))
        if helper.is_alive():
            raise self._send_timeout_death(timeout, helper)
        exc = outcome.get("exc")
        if exc is not None:
            # S9: every helper failure is a child death to the caller —
            # OSError/ValueError (dead pipe) and anything else alike. A
            # close() racing the send nulls Connection._handle mid-write,
            # so the helper can raise TypeError from os.write; that must
            # surface as _ChildDied(death_detail()), never as a raw
            # internal exception escaping request().
            raise _ChildDied(self.death_detail()) from exc

    def _send_timeout_death(self, timeout: float,
                            helper: threading.Thread) -> _ChildDied:
        """S1r: the child stopped reading the request pipe — stop it now.

        The worker is killed FIRST (SIGKILL: a frozen or wedged child
        ignores SIGTERM), which closes its end of the pipe and fails the
        helper's blocked ``os.write`` with EPIPE so the helper can unwind;
        only then is the session torn down — never closing the connection
        out from under a live writer — and the request reports the send
        bound as the reason it failed.
        """
        proc = self.proc
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        helper.join(timeout=5)   # EPIPE releases the blocked write
        self.close()
        return _ChildDied(
            f"stopped responding (the request could not be delivered "
            f"within {timeout:g}s)")

    def _timeout_death(self, timeout: float) -> _ChildDied:
        """S1: the child went silent for the whole bound — stop it now.

        Closes the pipes and terminate+kill+reaps the process (no orphan
        worker survives a timeout), then reports the bound as the reason
        the request failed.
        """
        self.close()
        return _ChildDied(f"stopped responding (no answer for {timeout:g}s)")

    def _max_total_death(self) -> _ChildDied:
        """S1c: the child kept talking but the request ran past its total
        cap — stop it now.

        Frames reset the silence clock, so a backend looping in
        ``progress()`` forever never trips ``_timeout_death``; the total
        cap is the bound that catches it. Same teardown as a silence
        timeout (close stops and reaps the child — no orphan), but the
        message says the work never finished rather than the child going
        silent, because it did not.
        """
        cap = _REQUEST_MAX_TOTAL
        self.close()
        return _ChildDied(
            f"never finished (the request ran for {cap:g}s without completing)")

    def close(self) -> None:
        for conn in (self._to, self._from):
            try:
                conn.close()
            except Exception:
                pass
        proc = self.proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass


class IsolatedTranscriber(Transcriber):
    """A backend whose model load and inference run in a child process.

    The child takes every native crash. If it dies by signal mid-load or
    mid-inference, the parent sees a closed pipe — not a dead listn — and
    applies the compute-device policy:

    * ``auto`` — a crash that happened (or could have happened) on the
      GPU restarts the worker pinned to CPU, exactly once, and says so on
      the status line: a truthful fallback, never a silent one. A crash
      that was already on CPU, or a second crash after the fallback, is
      an error.
    * ``cuda``/``rocm`` — any crash or GPU refusal raises
      :class:`CudaUnavailable` with the per-backend advice (CUDA libraries
      or ROCm setup); the worker is never restarted on CPU.
    * ``cpu`` — a crash is an error: there is no other device to fall
      back to.

    Only complete answers cross the process boundary, so callers write to
    the library exactly when (and because) inference finished.
    """

    backend = "faster-whisper"
    device = "cpu"

    def __init__(self, model: str, download: Callable[[str], None] | None = None, *,
                 compute_type: str = "int8", cpu_threads: int | None = None,
                 beam_size: int = 5, compute_device: str = "auto"):
        super().__init__(model)
        self._notify = download
        requested = (compute_device or "auto").strip().lower() or "auto"
        self._requested_device = requested
        self._effective_device = requested
        self._knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads,
                           beam_size=beam_size, compute_device=requested)
        self._session: _WorkerSession | None = None
        self._loaded = False
        self._device: str | None = None
        # S2r: what this parent's own probe says `auto` resolves to on this
        # machine — computed once per wrapper on first use, see
        # _parent_auto_plan.
        self._parent_plan: Plan | None = None
        self._notes: list[str] = []
        self._fatal: tuple[type[BaseException], str] | None = None
        self.device_message = "device: unknown"
        self._lock = threading.Lock()
        # The first load happens here — where the in-process backend would
        # have built its model — so callers see identical timing. Status
        # announcements made during it are deferred to get_transcriber,
        # which speaks exactly once for the finished build (F1: one line).
        self._initializing = True
        try:
            self._run("loading the model", self._worker)
        finally:
            self._initializing = False

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Stop the worker process (its model and all) and release it.

        The lock wait is bounded (S1): a request wedged in another thread
        — or an atexit/eviction close during one — must not block forever.
        After the bound the worker is stopped anyway; the wedged request
        then fails cleanly through its own dead-pipe path instead of
        holding the process (or the whole cache) open.
        """
        acquired = self._lock.acquire(timeout=_CLOSE_LOCK_TIMEOUT)
        try:
            self._kill_worker()
        finally:
            if acquired:
                self._lock.release()

    def __del__(self):
        try:
            self._kill_worker()
        except Exception:
            pass

    def _kill_worker(self) -> None:
        session, self._session = self._session, None
        self._loaded = False
        if session is not None:
            session.close()

    def _refresh_message(self) -> None:
        self.device_message = f"device: {self._device or 'unknown'}"
        if self._notes:
            self.device_message += " — " + "; ".join(self._notes)

    # -- what the child may claim (S2) ------------------------------------
    def _parent_auto_plan(self) -> Plan:
        """What this parent's own probe says ``auto`` resolves to here (S2r).

        Under ``auto`` the child must not be the device authority: before a
        GPU claim, a GPU status line or a CPU-fallback note it reports is
        believed, the parent re-runs the same subprocess-based probe the
        in-process build uses (:func:`_resolve_plan` — ``cuda_device_count``,
        ``cuda_runtime_ready`` and the ROCm check all run in a child
        process, so no native module is ever loaded here). Computed once
        per wrapper; the seams are the module's own, so tests pin them
        exactly like the in-process path does. A probe that sees no usable
        GPU answers ``gpu_active == False`` and a probe that did not
        narrow answers ``device != "cpu"`` — both drop the child's claim
        rather than believe it.
        """
        if self._parent_plan is None:
            self._parent_plan = _resolve_plan("auto")
        return self._parent_plan

    def _gpu_line_allowed(self, target: str) -> bool:
        """May a line claiming GPU use pass for this device decision (S2r)?

        ``cpu`` never — the parent chose CPU and a GPU line contradicts it.
        An explicit ``cuda``/``rocm`` ask, or a claim the parent's own
        probe corroborated under ``auto``, yes. Under ``auto`` without a
        usable GPU in the parent's probe, no: the child would be inventing
        the device it is running on.
        """
        if target == "cpu":
            return False
        if target in ("cuda", "rocm"):
            return True
        return self._parent_auto_plan().gpu_active

    def _adopt_device_claim(self, claimed: str) -> str:
        """A worker answer may only confirm the parent's device decision.

        ``cpu`` (asked for, or pinned after a crash) must come back as
        ``cpu``; an explicit ``cuda``/``rocm`` ask must come back as that
        same GPU; ``auto`` accepts its own decision and ``cpu`` (neither
        ever widens) and a GPU only when this parent's own probe
        corroborates a usable GPU (S2r) — the child runs the same resolve
        on the same machine, but it does not get to be the sole device
        truth on the default setting. Anything else is a contradiction:
        the claim is dropped, the parent's decision stands, and the
        mismatch is logged — a hostile or broken child can never talk the
        parent onto a device it did not choose.
        """
        want = self._effective_device
        if want == "cpu":
            accepted = claimed == "cpu"
        elif want == "auto":
            if claimed in ("auto", "cpu"):
                accepted = True   # the parent's own decision, or narrower
            else:
                # S2r: a GPU claim under auto must be corroborated by this
                # parent's own probe of this machine.
                plan = self._parent_auto_plan()
                accepted = bool(plan.gpu_active) and (
                    claimed == plan.kind
                    or (plan.kind == "rocm" and claimed == "cuda"))
        else:
            # CTranslate2's device API names every GPU "cuda", so a ROCm
            # build may legitimately answer "cuda" for a "rocm" ask.
            accepted = claimed == want or (want == "rocm" and claimed == "cuda")
        if accepted:
            return claimed
        log.warning("the transcription process reported device %r for "
                    "compute_device=%r; keeping %r", claimed, want, want)
        return want

    def _note_is_adoptable(self, note: str) -> bool:
        """A status note may only describe a fallback that really happened.

        The child may announce the canonical CPU-fallback note only for an
        ``auto`` build that resolved to CPU. A ``cpu`` ask never fell back,
        an explicit GPU ask never falls back, and a note this parent wrote
        itself (the crash pin) is never overwritten by a later answer.
        S2r adds the parent's own word: under ``auto`` the note must also
        describe a narrowing THIS parent's probe made — a child whose
        machine the parent's probe did not narrow is fabricating it.
        """
        if self._notes or self._effective_device != "auto":
            return False
        if self._device not in (None, "cpu"):
            return False
        if _claims_gpu(note):
            return False   # a note may not claim a GPU
        if self._parent_auto_plan().device != "cpu":
            log.warning("dropped a worker status note the parent's own "
                        "probe did not narrow for: %r", note)
            return False
        return True

    def _message_is_adoptable(self, message: str) -> bool:
        """The worker's ``device: …`` line may not contradict our decision."""
        target = self._device or self._effective_device
        token = _DEVICE_TOKEN.match(message.strip())
        if token and token.group(1) != target:
            log.warning("dropped a worker device message claiming %r "
                        "(compute device is %r): %r",
                        token.group(1), target, message)
            return False
        if _claims_gpu(message) and not self._gpu_line_allowed(target):
            log.warning("dropped a worker device message claiming GPU use "
                        "(compute device is %r): %r", target, message)
            return False
        return True

    def _child_status(self, message) -> None:
        """Forward a worker status line unless it contradicts our decision.

        A worker asked to run on CPU — or pinned to it after a crash — can
        never announce GPU acceleration to the user (S2): such a line is
        dropped. Under ``auto`` the same line is dropped unless this
        parent's own probe saw a usable GPU (S2r), and a CPU-fallback note
        is forwarded only when the parent's own probe narrowed too.
        Everything else passes through unchanged: download notices,
        truthful ``device: …`` lines.
        """
        if message is None:
            return
        text = str(message)
        target = self._device or self._effective_device
        if _claims_gpu(text) and not self._gpu_line_allowed(target):
            log.warning("dropped a worker status line claiming GPU use while "
                        "compute device is %r: %r", target, text)
            return
        if (self._effective_device == "auto" and CPU_FALLBACK_MESSAGE in text
                and not self._note_is_adoptable(text)):
            log.warning("dropped a worker status line announcing a CPU "
                        "fallback the parent's own probe did not narrow "
                        "for: %r", text)
            return
        if self._notify:
            self._notify(text)

    def _worker(self) -> _WorkerSession:
        """The worker process, spawned and with its model loaded."""
        if self._session is None:
            self._session = _WorkerSession.spawn()
            self._loaded = False
        if not self._loaded:
            header = {"op": "load", "backend": self.backend,
                      "model": self.model_name,
                      "knobs": dict(self._knobs,
                                    compute_device=self._effective_device),
                      "factory": _FACTORY_PATH}
            try:
                response = self._session.request(header,
                                                 on_status=self._child_status)
            except _ChildDied as death:
                raise _ChildDied(death.detail,
                                 phase="loading the model") from None
            if not response.get("ok"):
                detail = str(response.get("error") or "the model could not be loaded")
                if response.get("kind") in ("cuda", "device"):
                    raise _CudaFailure(
                        detail, device=response.get("kind") == "device")
                raise _BackendError(detail)
            self._loaded = True
            # S2: the child's answer may only CONFIRM the parent's device
            # decision — it can never widen it.
            reported = response.get("device")
            if reported:
                adopted = self._adopt_device_claim(str(reported))
                self.device = adopted
                self._device = adopted
            note = response.get("status_note")
            if note and self._note_is_adoptable(str(note)):
                # The child's own pre-build narrowing (the probe pinning CPU
                # with the canonical message), adopted only when it describes
                # the fallback this parent's decision allows. Set once; a
                # later reload that reports no note of its own never
                # overwrites it.
                self.status_note = str(note)
            if self._notes:
                # Keep the fallback note visible beside what the fresh
                # worker reports; the note is why it is running on CPU.
                self._refresh_message()
            elif (response.get("device_message")
                  and self._message_is_adoptable(str(response["device_message"]))):
                self.device_message = str(response["device_message"])
            else:
                self._refresh_message()
        return self._session

    # -- fallback policy --------------------------------------------------

    def _run(self, phase: str, fn):
        """Run ``fn``; turn a worker death into an error or a CPU fallback."""
        if self._fatal:
            cls, message = self._fatal
            raise cls(message)
        retried = False
        while True:
            try:
                return fn()
            except _ChildDied as death:
                answer = self._on_death(death.detail, death.phase or phase, retried)
                if isinstance(answer, _Retry):
                    retried = True
                    continue
                raise answer
            except _CudaFailure as failure:
                answer = self._on_cuda_failure(failure.detail, phase, retried,
                                               device=failure.device)
                if isinstance(answer, _Retry):
                    retried = True
                    continue
                raise answer
            except _BackendError:
                raise  # a normal Python error: the worker itself is fine
            except BaseException:
                # Cancelled / KeyboardInterrupt: stop native work at once.
                self._kill_worker()
                raise

    def _set_fatal(self, cls: type, message: str) -> None:
        # Store type+text, not the exception: it gets re-raised on every
        # later call and must not accumulate tracebacks.
        self._fatal = (cls, message)

    def _pin_cpu(self, detail: str, phase: str,
                 status_note: str | None = None) -> None:
        """Truthful ``auto`` fallback: restart on CPU and say exactly why.

        ``status_note`` overrides what is announced: the canonical
        ``CPU_FALLBACK_MESSAGE`` for a load/availability fault (F1),
        otherwise the descriptive note itself (a signal death).
        """
        self._effective_device = "cpu"
        self._knobs["compute_device"] = "cpu"
        self._device = "cpu"
        self.device = "cpu"
        note = (
            f"the transcription process {detail} while {phase}; running on CPU")
        self._notes.append(note)
        self.status_note = status_note or note
        self._refresh_message()
        self._kill_worker()
        if self._notify and not self._initializing:
            self._notify(str(self.status_note))

    def _on_death(self, detail: str, phase: str, retried: bool) -> BaseException | _Retry:
        want = self._requested_device
        gpu_suspect = self._device != "cpu"  # None = died before reporting
        if want in ("cuda", "rocm"):
            advice = (
                "Check the ROCm setup (docs/GPU_LINUX.md), or run on CPU with "
                "--compute-device cpu." if want == "rocm" else
                "Install the CUDA 12 libraries (docs/GPU_LINUX.md), or run on "
                "CPU with --compute-device cpu.")
            message = (f"compute_device={want} was requested, but the transcription "
                       f"process {detail} while {phase}. {advice}")
            self._set_fatal(CudaUnavailable, message)
            self._kill_worker()
            return CudaUnavailable(message)
        if want == "auto" and gpu_suspect and not retried:
            self._pin_cpu(detail, phase)
            return _RETRY
        if retried:
            message = (f"The transcription process {detail} while {phase} even "
                       "after falling back to CPU; the model cannot run on this "
                       "machine. See docs/GPU_LINUX.md.")
        else:
            message = (f"The transcription process {detail} while {phase} "
                       f"(listn was running on "
                       f"{self._device or self._effective_device}, so there is "
                       "no other device to fall back to). See docs/GPU_LINUX.md.")
        self._set_fatal(RuntimeError, message)
        self._kill_worker()
        return RuntimeError(message)

    def _on_cuda_failure(self, detail: str, phase: str, retried: bool,
                         *, device: bool = False) -> BaseException | _Retry:
        want = self._requested_device
        # S3: the text is classified whatever ``kind`` the child wrapped it
        # in. A runtime fault (OOM, illegal memory access, NCCL …) stays
        # itself even under kind="device" — no CPU rebuild, no relabelling
        # as "CUDA libraries unavailable" — the same belt-and-braces rule
        # the availability classifier applies to kind="cuda".
        if _is_runtime_failure_text(detail):
            return RuntimeError(detail)
        # F1: only a load/availability failure ("CUDA libraries unavailable")
        # may be recovered as CPU. A runtime fault — OOM, illegal memory
        # access, NCCL — propagates as itself: no CPU rebuild, no relabel.
        # ``device`` skips the re-read: the child raised CudaUnavailable
        # itself (the probe's pre-build refusal), which is availability by
        # construction — and whose phrasing ("sees no CUDA device") the
        # marker list would otherwise demote to a bare RuntimeError.
        if not device and not _is_cuda_unavailable(RuntimeError(detail)):
            return RuntimeError(detail)
        if want == "auto" and not retried:
            self._pin_cpu(f"failed ({detail})", phase,
                          status_note=CPU_FALLBACK_MESSAGE)
            return _RETRY
        if want in ("cuda", "rocm"):
            if "--compute-device cpu" in detail:
                message = detail
            elif want == "rocm":
                message = (f"{detail} (compute_device=rocm was requested). Check "
                           "the ROCm setup (docs/GPU_LINUX.md), or run on CPU "
                           "with --compute-device cpu.")
            else:
                message = (f"{detail} (compute_device=cuda was requested). Install "
                           "the CUDA 12 libraries (docs/GPU_LINUX.md), or run on "
                           "CPU with --compute-device cpu.")
        else:
            message = f"{detail} (compute device: {self._device or self._effective_device})"
        self._set_fatal(CudaUnavailable, message)
        self._kill_worker()
        return CudaUnavailable(message)

    # -- inference --------------------------------------------------------

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None, vad: bool = True,
                   offset: float = 0.0, hotwords: str | None = None,
                   progress: Callable[[float], None] | None = None) -> list[RawSegment]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        audio = np.ascontiguousarray(audio)
        with self._lock:
            self._run("loading the model", self._worker)
            return self._run(
                "transcribing",
                lambda: self._send_transcribe(audio, language, prompt, vad,
                                              offset, hotwords, progress))

    def _send_transcribe(self, audio, language, prompt, vad, offset, hotwords,
                         progress) -> list[RawSegment]:
        session = self._worker()
        header = {"op": "transcribe", "language": language, "prompt": prompt,
                  "vad": bool(vad), "offset": float(offset or 0.0),
                  "hotwords": hotwords, "progress": progress is not None,
                  "nbytes": int(audio.nbytes)}
        response = session.request(header, audio, on_status=self._child_status,
                                   on_progress=progress)
        if not response.get("ok"):
            detail = str(response.get("error") or "transcription failed")
            if response.get("kind") in ("cuda", "device"):
                raise _CudaFailure(detail,
                                   device=response.get("kind") == "device")
            raise _BackendError(detail)
        return [RawSegment(float(row[0]), float(row[1]), str(row[2]),
                           float(row[3]))
                for row in response.get("segments") or []]

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        # The worker runs the full transcribe (which applies the offset and
        # the hallucination filter); expose that result as an iterator.
        return iter(self.transcribe(audio, language=language, prompt=prompt,
                                    vad=vad, hotwords=hotwords))


def _close_cached(keep: tuple | None = None) -> None:
    """Stop the workers of cached models that are about to be evicted."""
    for key, instance in list(_cache.items()):
        if key == keep:
            continue
        close = getattr(instance, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            log.debug("closing a cached transcriber worker failed", exc_info=True)


# Children must not outlive the parent process; pipes would end their lives
# anyway at exit, but terminate them promptly instead of at process teardown.
atexit.register(_close_cached)


def get_transcriber(model: str, backend: str = "auto",
                    status: Callable[[str], None] | None = None,
                    *, compute_type: str | None = None, cpu_threads: int | None = None,
                    beam_size: int | None = None,
                    compute_device: str | None = None,
                    isolate: bool | None = None) -> Transcriber:
    """Build (and cache) a backend.

    The performance knobs and the compute device default to the effective
    ``Settings`` (settings.json plus ``LISTN_*`` env overrides), so every
    caller — live capture, file jobs, the ingest socket — picks them up
    without being changed. Pass them explicitly to override. Only the
    faster-whisper backend uses them.

    ``isolate`` says where the work runs. ``None`` (the default) isolates
    the faster-whisper backend — the one whose CUDA/HIP path can die on a
    native signal — in a child process (:class:`IsolatedTranscriber`), and
    runs every other backend in-process. ``False`` forces in-process (the
    unit tests inspect the constructed model); ``True`` forces a child.

    ``compute_device`` accepts ``cpu``/``auto``/``cuda``/``rocm`` — the
    ``rocm`` name is accepted here, in settings.json, in
    ``LISTN_COMPUTE_DEVICE`` and by ``--compute-device`` alike.

    ``status`` receives one truthful line per call: the canonical fallback
    reason (``status_note``) when the build had to abandon its device,
    otherwise the ``device: …`` report of where the model really runs.
    A faster-whisper build whose CUDA libraries turn out to be unusable is
    rebuilt on CPU, announced through ``status``, and cached under its own
    device key.
    """
    name = resolve_backend(backend)
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")
    knobs: dict = {}
    if name == "faster-whisper":
        if (compute_type is None or cpu_threads is None or beam_size is None
                or compute_device is None):
            from listn.config import Settings

            s = Settings.load()
            compute_type = s.compute_type if compute_type is None else compute_type
            cpu_threads = s.cpu_threads if cpu_threads is None else cpu_threads
            beam_size = s.beam_size if beam_size is None else beam_size
            # The device is a setting too: engine jobs call this without one,
            # so settings.json / LISTN_COMPUTE_DEVICE (what --compute-device
            # publishes) have to reach the backend here or the flag would be
            # unreachable outside the tests that pass it explicitly.
            compute_device = (s.compute_device
                              if compute_device is None else compute_device)
        knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads,
                     beam_size=beam_size, compute_device=compute_device)
    if isolate is None:
        isolate = name == "faster-whisper"
    # The device sits LAST in the key: a build that fell back to CPU is only
    # ever served from its cpu key — never as the configured (auto) one.
    prefix = (name, model, knobs.get("compute_type"), knobs.get("cpu_threads"),
              knobs.get("beam_size"), bool(isolate))
    requested_device = (knobs.get("compute_device")
                        if name == "faster-whisper" else None)
    candidates = [prefix + (requested_device,)]
    if requested_device == "auto":
        candidates.append(prefix + ("cpu",))
    instance: Transcriber | None = None
    with _cache_lock:
        for key in candidates:
            found = _cache.get(key)
            if found is None:
                continue
            instance = found
            if requested_device == "auto":
                effective = key[:-1] + (getattr(found, "device", requested_device),)
                if effective != key:
                    # It fell back to CPU after being cached: move it over so
                    # the auto key never describes it again.
                    del _cache[key]
                    _cache[effective] = found
            break
        if instance is None:
            try:
                if isolate:
                    instance = IsolatedTranscriber(model, download=status, **knobs)
                else:
                    instance = _build_backend(name, model, knobs, download=status)
            except ImportError as exc:
                extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
                hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
                raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc
            _close_cached(keep=prefix + (requested_device,))
            _cache.clear()  # keep one model resident; they're large
            _cache[prefix + (requested_device,)] = instance
    # One truthful line per call, cache hit or not: the canonical fallback
    # reason when the build had to abandon its device, otherwise where the
    # model really runs (nothing to say for a plain "auto" build).
    message = getattr(instance, "status_note", None)
    if message is None and getattr(instance, "device", None) not in (None, "auto"):
        message = getattr(instance, "device_message", None)
    if status and message:
        status(message)
    return instance


def register_backend(name: str, factory) -> None:
    """Hook for tests: register a fake backend."""
    _BACKENDS[name] = factory
    with _cache_lock:
        _close_cached()
        _cache.clear()
