"""Settings and the on-disk data directory.

Everything listn writes (recordings, transcripts, settings) lives under one
private data directory outside the repository:

* macOS: ``~/Library/Application Support/listn``
* elsewhere: ``$XDG_DATA_HOME/listn`` (default ``~/.local/share/listn``)

``LISTN_DATA_DIR`` overrides both.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from listn.llm import DEFAULT_LLM

MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
BACKENDS = ["auto", "faster-whisper", "mlx", "openai-whisper"]


def disable_network_telemetry() -> None:
    """Turn off usage reporting in third-party libraries we load."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def default_model() -> str:
    """Live model for a fresh install.

    On Apple Silicon, ``large-v3-turbo`` is the default when mlx-whisper is
    installed. It is a distilled large-v3 (about 1.6 GB): on an M-series GPU
    a 6-second chunk finishes well inside the chunk interval, including when
    system audio and the microphone are transcribed one after the other.
    Without MLX, turbo would run on CPU and can fall behind two live tracks,
    so the default stays ``small``. Other machines stay on ``small`` too.
    """
    if apple_silicon():
        try:
            import mlx_whisper  # noqa: F401

            return "large-v3-turbo"
        except ImportError:
            return "small"
    return "small"


def data_dir() -> Path:
    override = os.environ.get("LISTN_DATA_DIR")
    if override:
        base = Path(override).expanduser()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "listn"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".local" / "share") / "listn"
    base.mkdir(parents=True, exist_ok=True)
    try:
        base.chmod(0o700)
    except OSError:
        pass
    return base


@dataclass
class Settings:
    device: str | None = None
    model: str = field(default_factory=default_model)
    language: str | None = None
    backend: str = "auto"
    chunk_seconds: float = 6.0
    save_audio: bool = True
    diarize_after: bool = False
    hf_token: str | None = None
    llm_model: str = DEFAULT_LLM
    summarize_after: bool = True
    offer_meetings: bool = True
    auto_record_meetings: bool = False
    launch_at_login: bool = False
    vocabulary: list = field(default_factory=list)
    vad_chunks: bool = True
    match_speakers: bool = True
    speaker_threshold: float = 0.72

    @classmethod
    def path(cls) -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in raw.items() if k in known})
        return settings.validated()

    def validated(self) -> "Settings":
        if self.model not in MODELS:
            self.model = "small"
        if self.backend not in BACKENDS:
            self.backend = "auto"
        try:
            self.chunk_seconds = min(30.0, max(2.0, float(self.chunk_seconds)))
        except (TypeError, ValueError):
            self.chunk_seconds = 6.0
        self.language = (self.language or "").strip().lower() or None
        self.device = self.device or None
        self.hf_token = (self.hf_token or "").strip() or None
        self.llm_model = (self.llm_model or DEFAULT_LLM).strip()[:200] or DEFAULT_LLM
        self.summarize_after = bool(self.summarize_after)
        self.offer_meetings = bool(self.offer_meetings)
        self.auto_record_meetings = bool(self.auto_record_meetings)
        self.launch_at_login = bool(self.launch_at_login)
        from listn.vocab import clean_vocabulary

        self.vocabulary = clean_vocabulary(self.vocabulary)
        self.vad_chunks = bool(self.vad_chunks)
        self.match_speakers = bool(self.match_speakers)
        try:
            self.speaker_threshold = min(0.95, max(0.4, float(self.speaker_threshold)))
        except (TypeError, ValueError):
            self.speaker_threshold = 0.72
        return self

    def save(self) -> None:
        path = self.path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.chmod(0o600)
        tmp.replace(path)

    def resolved_hf_token(self) -> str | None:
        return self.hf_token or os.environ.get("HF_TOKEN") or None

    def public(self) -> dict:
        data = asdict(self)
        data["hf_token"] = bool(self.resolved_hf_token())
        return data
