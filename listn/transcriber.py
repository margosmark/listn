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
"""

from __future__ import annotations

import logging
import platform
import re
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from listn.audio import TARGET_RATE, rms
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


class Transcriber:
    backend = "base"
    # The device the model is actually built for; only faster-whisper sets it
    # ("auto" as configured, "cpu" after a CUDA-library fallback). Others keep
    # the default and are keyed without a device.
    device: str | None = None
    # Set by a backend that had to abandon its configured device (the CPU
    # rebuild after CUDA libraries turned out to be unusable). Shown as the
    # job's stage so the fallback is visible instead of silent.
    status_note: str | None = None

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

    def __init__(self, model: str, download: Callable[[str], None] | None = None, *,
                 compute_type: str = "int8", cpu_threads: int | None = None,
                 beam_size: int = 5):
        super().__init__(model)
        from faster_whisper import WhisperModel

        # Defaults are byte-identical to the old hard-coded call: device auto,
        # int8, and no cpu_threads passed (CTranslate2 picks its own count).
        self.beam_size = beam_size
        # The device this model is actually built for: "auto" unless
        # CTranslate2 can only be CPU (no GPU visible), or the CUDA libraries
        # failed and we rebuilt on CPU. The cache keys on it so a fallback
        # build is never served as if it were the configured one.
        self.device = "auto" if _cuda_device_count() != 0 else "cpu"
        # ``download`` is get_transcriber's status callback: it reports the
        # one-time model download and the CPU fallback alike.
        self._status = download
        kwargs: dict = dict(device="auto", compute_type=compute_type)
        if cpu_threads:
            kwargs["cpu_threads"] = int(cpu_threads)
        self._model_kwargs = kwargs

        def cached():
            try:
                return WhisperModel(model, local_files_only=True, **self._model_kwargs)
            except RuntimeError as exc:
                # The weights are local and the rebuild below is local-only,
                # so recovering here can never turn into a download attempt
                # by load_local_or_download.
                if self.device == "cpu" or not _is_cuda_unavailable(exc):
                    raise
                log.warning("Model build on CUDA failed (%s); rebuilding on CPU", exc)
                return self._rebuild_on_cpu()

        def fetched():
            return WhisperModel(model, **self._model_kwargs)

        self._model = load_local_or_download(
            cached,
            fetched,
            on_download=(lambda: download(f"Downloading Whisper '{model}' (one time)...")) if download else None,
        )
        if self.status_note:
            self._announce_fallback()

    def _rebuild_on_cpu(self):
        """Rebuild once on CPU after a CUDA-library failure; never loops.

        compute_type stays as configured (int8 is valid on CPU) and the
        weights are already on disk, so this load is local-only.
        """
        from faster_whisper import WhisperModel
        self.device = "cpu"
        self.status_note = CPU_FALLBACK_MESSAGE
        self._model_kwargs["device"] = "cpu"
        self._model = WhisperModel(self.model_name, local_files_only=True, **self._model_kwargs)
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
                # of looping.
                if self.device == "cpu" or yielded or not _is_cuda_unavailable(exc):
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
# Keyed by (backend, model, compute_type, cpu_threads, beam_size, device) so a
# knob change reloads the model, and a build that fell back to CPU is only ever
# served from its cpu key — never as the configured (auto) one.
_cache: dict[tuple, Transcriber] = {}
_cache_lock = threading.Lock()


def get_transcriber(model: str, backend: str = "auto",
                    status: Callable[[str], None] | None = None,
                    *, compute_type: str | None = None, cpu_threads: int | None = None,
                    beam_size: int | None = None) -> Transcriber:
    """Build (and cache) a backend.

    The performance knobs default to the effective ``Settings`` (settings.json
    plus ``LISTN_*`` env overrides), so every caller — live capture, file
    jobs, the ingest socket — picks them up without being changed. Pass them
    explicitly to override. Only the faster-whisper backend uses them.

    A faster-whisper build whose CUDA libraries turn out to be unusable is
    rebuilt on CPU, announced through ``status``, and cached under its own
    device key.
    """
    name = resolve_backend(backend)
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")
    knobs: dict = {}
    if name == "faster-whisper":
        if compute_type is None or cpu_threads is None or beam_size is None:
            from listn.config import Settings

            s = Settings.load()
            compute_type = s.compute_type if compute_type is None else compute_type
            cpu_threads = s.cpu_threads if cpu_threads is None else cpu_threads
            beam_size = s.beam_size if beam_size is None else beam_size
        knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads, beam_size=beam_size)
    prefix = (name, model, knobs.get("compute_type"), knobs.get("cpu_threads"), knobs.get("beam_size"))
    # faster-whisper builds start on device "auto"; one that had to fall back
    # to CPU is keyed as "cpu" and found only under that key.
    requested_device = "auto" if name == "faster-whisper" else None
    candidates = [prefix + (requested_device,)]
    if requested_device == "auto":
        candidates.append(prefix + ("cpu",))
    instance: Transcriber | None = None
    fresh = False
    with _cache_lock:
        for key in candidates:
            found = _cache.get(key)
            if found is None:
                continue
            instance = found
            effective = key[:-1] + (getattr(found, "device", requested_device),)
            if effective != key:
                # It fell back to CPU after being cached: move it over so the
                # auto key never describes it again.
                del _cache[key]
                _cache[effective] = found
            break
        if instance is None:
            fresh = True
            try:
                instance = _BACKENDS[name](model, download=status, **knobs)
            except ImportError as exc:
                extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
                hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
                raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc
            _cache.clear()  # keep one model resident; they're large
            _cache[prefix + (getattr(instance, "device", requested_device),)] = instance
    if not fresh and getattr(instance, "status_note", None) and status:
        # A cache hit on a CPU-fallback build says so instead of silently
        # transcribing on a different device than the one configured.
        status(instance.status_note)
    return instance


def register_backend(name: str, factory) -> None:
    """Hook for tests: register a fake backend."""
    _BACKENDS[name] = factory
    with _cache_lock:
        _cache.clear()
