"""Local Whisper backends behind one small interface.

* ``faster-whisper`` (default): CTranslate2 int8 on CPU, built-in Silero VAD,
  streams segments so long files report progress. Roughly 4x faster than
  openai-whisper on CPU with a fraction of the memory, and no PyTorch.
* ``mlx``: Apple's MLX port, runs on the Apple Silicon GPU. Optional extra.
* ``openai-whisper``: the original reference implementation. Optional extra.

Models are downloaded once from Hugging Face (or OpenAI's CDN for
openai-whisper) and loaded from the local cache afterwards; listn tries the
cache first so it works fully offline once a model is present.
"""

from __future__ import annotations

import platform
import re
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from listn.audio import TARGET_RATE, rms

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


class Transcriber:
    backend = "base"

    def __init__(self, model: str):
        self.model_name = model

    def segments(self, audio: np.ndarray, language: str | None = None,
                 prompt: str | None = None, vad: bool = True) -> Iterator[RawSegment]:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: str | None = None,
                   vad: bool = True, offset: float = 0.0,
                   progress: Callable[[float], None] | None = None) -> list[RawSegment]:
        """Transcribe and filter. Segment times are shifted by ``offset``."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        quiet = rms(audio) < SPEECH_RMS_GATE * 2
        duration = audio.size / TARGET_RATE
        out = []
        for seg in self.segments(audio, language, prompt, vad):
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

    def __init__(self, model: str, download: Callable[[str], None] | None = None):
        super().__init__(model)
        from faster_whisper import WhisperModel

        kwargs = dict(device="auto", compute_type="int8")
        try:
            self._model = WhisperModel(model, local_files_only=True, **kwargs)
        except Exception:
            if download:
                download(f"Downloading Whisper '{model}' (one time)...")
            self._model = WhisperModel(model, **kwargs)

    def segments(self, audio, language=None, prompt=None, vad=True):
        segs, _info = self._model.transcribe(
            audio, language=language, initial_prompt=prompt, vad_filter=vad,
            beam_size=5, condition_on_previous_text=False,
        )
        for s in segs:
            yield RawSegment(s.start, s.end, s.text, getattr(s, "no_speech_prob", 0.0))


class _WindowedBackend(Transcriber):
    """Backends that don't stream: feed long audio in windows for progress."""

    window_seconds = 120

    def _run(self, audio, language, prompt) -> list[dict]:
        raise NotImplementedError

    def segments(self, audio, language=None, prompt=None, vad=True):
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
        try:
            self._path = snapshot_download(repo, local_files_only=True)
        except Exception:
            if download:
                download(f"Downloading Whisper '{model}' for MLX (one time)...")
            self._path = snapshot_download(repo)
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
_cache: dict[tuple[str, str], Transcriber] = {}
_cache_lock = threading.Lock()


def get_transcriber(model: str, backend: str = "auto",
                    status: Callable[[str], None] | None = None) -> Transcriber:
    name = resolve_backend(backend)
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")
    key = (name, model)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        try:
            instance = _BACKENDS[name](model, download=status)
        except ImportError as exc:
            extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
            hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
            raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc
        _cache.clear()  # keep one model resident; they're large
        _cache[key] = instance
        return instance


def register_backend(name: str, factory) -> None:
    """Hook for tests: register a fake backend."""
    _BACKENDS[name] = factory
    with _cache_lock:
        _cache.clear()
