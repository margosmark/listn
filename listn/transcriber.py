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

Where the default backend runs is the ``compute_device`` setting: ``cpu``
(the safe default, never touch the GPU), ``auto`` (GPU when this machine's
CUDA runtime is really usable, CPU otherwise), or ``cuda`` (require the GPU
and fail with a readable message instead of falling back). See
:func:`resolve_device`.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from listn.audio import TARGET_RATE, rms
from listn.hub import load_local_or_download

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

SPEECH_RMS_GATE = 0.003

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


class CudaUnavailable(RuntimeError):
    """``compute_device=cuda`` was asked for and this machine cannot run it.

    Raised before any model is built, so the user gets one readable line
    instead of a CTranslate2 failure (or a native crash) half-way through
    loading.
    """


# Shared libraries CTranslate2 dlopens by soname when a model is built for
# CUDA. On a machine with an NVIDIA driver but without the CUDA runtime
# libraries the load fails with "Library libcublas.so.12 is not found or
# cannot be loaded" — the failure seen on the CachyOS laptop.
CUDA_RUNTIME_LIBS = ("libcublas.so.12",)

# Runs in a child process. Loading a CUDA library, or anything CTranslate2
# would do with it, can terminate a process without ever raising a Python
# exception; the child takes that risk and the parent only reads one line of
# JSON back, so a broken GPU can never take listn down with it. Even the
# CTranslate2 device-count API is isolated because it calls the CUDA driver.
_PROBE_SCRIPT = """
import ctypes, json, sys
result = {"missing": [], "devices": 0}
try:
    import ctranslate2
    result["devices"] = int(ctranslate2.get_cuda_device_count())
except Exception as exc:
    result["device_error"] = str(exc)
for soname in sys.argv[1:]:
    try:
        ctypes.CDLL(soname)
    except OSError as exc:
        result["missing"].append("%s: %s" % (soname, exc))
print(json.dumps(result))
"""


def _run_cuda_probe(sonames: tuple[str, ...], timeout: float) -> tuple[dict | None, str | None]:
    """Run the native CUDA checks out-of-process; ``None`` means unsafe/unavailable."""
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE_SCRIPT, *sonames],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"the CUDA library probe timed out after {timeout:g}s"
    if done.returncode != 0:
        how = (f"signal {-done.returncode}" if done.returncode < 0
               else f"exit code {done.returncode}")
        return None, f"the CUDA library probe terminated with {how}"
    try:
        return json.loads(done.stdout.strip().splitlines()[-1]), None
    except (IndexError, ValueError):
        return None, "the CUDA library probe returned no usable answer"


def cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can see; query in a child process.

    A native driver/runtime bug becomes a quiet zero rather than killing listn.
    The actionable reason is emitted by :func:`cuda_runtime_ready` when a GPU
    device was requested explicitly.
    """
    result, _reason = _run_cuda_probe((), timeout=30.0)
    return int(result.get("devices", 0)) if result else 0


def cuda_runtime_ready(sonames: tuple[str, ...] = CUDA_RUNTIME_LIBS,
                       timeout: float = 30.0) -> tuple[bool, str]:
    """Can the CUDA runtime libraries CTranslate2 needs actually load?

    Returns ``(ready, detail)``. The native loader and CTranslate2 device
    query run in a child process: either can terminate the process without a
    Python exception. A child that dies by signal, times out, or prints
    nothing means "not ready" — returned as a reason, never as a crash.
    """
    result, error = _run_cuda_probe(sonames, timeout)
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
    return any(word in text for word in ("cuda", "cublas", "cudnn", "nccl", "libcuda", "gpu"))


def resolve_device(requested: str) -> tuple[str, str | None]:
    """Turn the ``compute_device`` setting into a WhisperModel device.

    Returns ``(device, note)``. ``note`` explains a CPU fallback when there is
    one and is ``None`` otherwise. Raises :class:`CudaUnavailable` when CUDA
    was requested by name and this machine cannot run it — checked before any
    model is built, so the answer is a message rather than a failure inside
    CTranslate2.

    ``auto`` means GPU only when a CUDA device is visible *and* the runtime
    libraries load; otherwise it stays on the historical behaviour (``auto``,
    which CTranslate2 resolves to CPU) or pins ``cpu`` when a GPU is visible
    but unusable.
    """
    want = (requested or "auto").strip().lower() or "auto"
    if want == "cpu":
        return "cpu", None
    if sys.platform == "darwin":
        # macOS has no CUDA; "auto" has always meant CPU for faster-whisper here.
        if want == "cuda":
            raise CudaUnavailable(
                "compute_device=cuda was requested, but macOS has no CUDA. "
                "Use compute_device=cpu, or leave it on auto.")
        return "auto", None
    if cuda_device_count() <= 0:
        if want == "cuda":
            raise CudaUnavailable(
                "compute_device=cuda was requested, but CTranslate2 sees no CUDA "
                "device (no NVIDIA driver, or none exposed to this process). "
                "Run on CPU with --compute-device cpu, or leave it on auto.")
        return "auto", None
    ready, detail = cuda_runtime_ready()
    if not ready:
        reason = f"CUDA is visible but the CUDA runtime cannot be used ({detail})"
        if want == "cuda":
            raise CudaUnavailable(
                f"compute_device=cuda was requested, but {reason}. Install the "
                "CUDA 12 libraries (docs/GPU_LINUX.md), or run on CPU with "
                "--compute-device cpu.")
        return "cpu", f"{reason}; see docs/GPU_LINUX.md"
    return ("cuda" if want == "cuda" else "auto"), None


def _reported_device(model: object, asked: str) -> str:
    """The device CTranslate2 says the loaded model is actually running on."""
    reported = getattr(getattr(model, "model", None), "device", None)
    return str(reported) if reported else asked


class Transcriber:
    backend = "base"
    # Set by backends that know where they are running; reported verbatim.
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
        # any machine without a usable CUDA runtime.
        self.beam_size = beam_size
        device, note = resolve_device(compute_device)
        notes = [note] if note else []
        kwargs: dict = dict(device=device, compute_type=compute_type)
        if cpu_threads:
            kwargs["cpu_threads"] = int(cpu_threads)
        want_cuda = (compute_device or "auto").strip().lower() == "cuda"
        failures: list[Exception] = []

        def load(dev: str):
            """Build the model on ``dev``.

            A device failure on the cached attempt is recorded in ``failures``
            so the download path never runs: a broken GPU is not a missing
            model, and retrying it over the network would both waste a round
            trip and claim a download that is not happening.
            """
            kw: dict = dict(kwargs, device=dev)
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

        try:
            self._model = load(device)
        except Exception as exc:
            if not failures and not _is_cuda_error(exc):
                raise  # genuinely not a device problem: bad files, no memory, ...
            if want_cuda:
                raise CudaUnavailable(
                    "compute_device=cuda was requested, but the model could not "
                    f"be built on the GPU: {exc}. Install the CUDA 12 libraries "
                    "(docs/GPU_LINUX.md), or run on CPU with "
                    "--compute-device cpu.") from exc
            # auto: a broken GPU must not stop transcription.
            notes.append(f"CUDA unusable ({exc}); running on CPU")
            self._model = load("cpu")

        self.device = _reported_device(self._model, device)
        # Truthful, user-facing: what the model reports, plus any fallback
        # reason. get_transcriber passes this to the status callback on every
        # call, cache hit or not.
        self.device_message = f"device: {self.device}"
        if notes:
            self.device_message += " — " + "; ".join(notes)

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        kwargs = dict(
            language=language, initial_prompt=prompt, vad_filter=vad,
            beam_size=self.beam_size, condition_on_previous_text=False,
        )
        if hotwords:
            kwargs["hotwords"] = hotwords
        try:
            segs, _info = self._model.transcribe(audio, **kwargs)
        except TypeError:
            kwargs.pop("hotwords", None)
            segs, _info = self._model.transcribe(audio, **kwargs)
        for s in segs:
            yield RawSegment(s.start, s.end, s.text, getattr(s, "no_speech_prob", 0.0))


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
# Keyed by (backend, model, compute_type, cpu_threads, beam_size) so a knob
# change reloads the model instead of reusing one built with the old settings.
_cache: dict[tuple, Transcriber] = {}
_cache_lock = threading.Lock()


def get_transcriber(model: str, backend: str = "auto",
                    status: Callable[[str], None] | None = None,
                    *, compute_type: str | None = None, cpu_threads: int | None = None,
                    beam_size: int | None = None,
                    compute_device: str | None = None) -> Transcriber:
    """Build (and cache) a backend.

    The performance knobs and the compute device default to the effective
    ``Settings`` (settings.json plus ``LISTN_*`` env overrides), so every
    caller — live capture, file jobs, the ingest socket — picks them up
    without being changed. Pass them explicitly to override. Only the
    faster-whisper backend uses them; ``status`` receives a truthful
    ``device: …`` line once the model reports where it is actually running.
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
            compute_device = s.compute_device if compute_device is None else compute_device
        knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads,
                     beam_size=beam_size, compute_device=compute_device)
    key = (name, model, knobs.get("compute_type"), knobs.get("cpu_threads"),
           knobs.get("beam_size"), knobs.get("compute_device"))
    with _cache_lock:
        instance = _cache.get(key)
        if instance is None:
            try:
                instance = _BACKENDS[name](model, download=status, **knobs)
            except ImportError as exc:
                extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
                hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
                raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc
            _cache.clear()  # keep one model resident; they're large
            _cache[key] = instance
    # Reported on every call, cache hit or not: where the model really runs.
    message = getattr(instance, "device_message", None)
    if status and message:
        status(message)
    return instance


def register_backend(name: str, factory) -> None:
    """Hook for tests: register a fake backend."""
    _BACKENDS[name] = factory
    with _cache_lock:
        _cache.clear()
