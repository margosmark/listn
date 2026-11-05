"""Settings and the on-disk data directory.

Everything listn writes (recordings, transcripts, settings) lives under one
private data directory outside the repository:

* macOS: ``~/Library/Application Support/listn``
* elsewhere: ``$XDG_DATA_HOME/listn`` (default ``~/.local/share/listn``)

``LISTN_DATA_DIR`` overrides both.

The transcription performance knobs (``compute_type``, ``cpu_threads``,
``beam_size``) can be set in ``settings.json`` or overridden per-run with
``LISTN_COMPUTE_TYPE``, ``LISTN_CPU_THREADS`` and ``LISTN_BEAM_SIZE``. The
environment wins over the file. Invalid values raise a ``SettingsError`` with
a readable message instead of silently falling back; ``SettingsError`` is a
``ValueError`` subclass so existing handlers keep working.
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
# CTranslate2 compute types faster-whisper accepts. "int8" is the default:
# quantised, smallest memory, fastest on CPU.
COMPUTE_TYPES = ["default", "auto", "int8", "int8_float32", "int8_float16",
                 "int8_bfloat16", "int16", "float16", "bfloat16", "float32"]
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 5
SETTINGS_VERSION = 2
SPEAKER_THRESHOLD = 0.55
SPEAKER_THRESHOLD_MIN = 0.3
SPEAKER_THRESHOLD_MAX = 0.9


class SettingsError(ValueError):
    """A settings value the user can fix: settings.json or a ``LISTN_*`` variable.

    Kept distinct from a bare ``ValueError`` so the CLI can turn *this* into a
    one-line ``listn: ...`` message with exit 2 while an unrelated ``ValueError``
    raised by a bug still escapes with its traceback. It subclasses
    ``ValueError`` on purpose: callers that already treat a bad setting as a
    user-facing error (the web API, existing tests) keep working unchanged.
    """


def disable_network_telemetry() -> None:
    """Turn off usage reporting in third-party libraries we load."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    # pyannote 4 reports usage to otel.pyannote.ai unless this is "0" before it is imported.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["OTEL_SDK_DISABLED"] = "true"


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
    # Transcription performance knobs. Defaults match the values that used to
    # be hard-coded in the faster-whisper backend, so nothing changes when
    # they are left alone. cpu_threads=None means "don't pass it" and lets
    # CTranslate2 pick (today's behaviour).
    compute_type: str = DEFAULT_COMPUTE_TYPE
    cpu_threads: int | None = None
    beam_size: int = DEFAULT_BEAM_SIZE
    chunk_seconds: float = 6.0
    save_audio: bool = True
    # None follows the token: on when one is set (or the model is cached).
    diarize_after: bool | None = None
    hf_token: str | None = None
    llm_model: str = DEFAULT_LLM
    summarize_after: bool = True
    offer_meetings: bool = True
    auto_record_meetings: bool = False
    launch_at_login: bool = False
    vocabulary: list = field(default_factory=list)
    vad_chunks: bool = True
    match_speakers: bool = True
    # Cosine similarity of WeSpeaker ResNet34 embeddings (see listn/speakers.py).
    speaker_threshold: float = SPEAKER_THRESHOLD
    # Only used by the spectrum fallback when pyannote is not installed.
    spectrum_threshold: float = 0.72
    # Linux desktop extras (listn tray), opt-in and inert elsewhere:
    # captions render only while a recording runs and are never logged or
    # copied; the shortcut pair is registered with the desktop's own
    # global-shortcut portal, which reports what it actually granted.
    captions_enabled: bool = False
    hotkey_start_stop: str = "Ctrl+Alt+R"
    hotkey_bookmark: str = "Ctrl+Alt+B"
    audio_retention_days: int = 0
    min_free_gb: float = 2.0
    battery_saver: bool = True
    settings_version: int = SETTINGS_VERSION

    @classmethod
    def path(cls) -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in migrate(raw).items() if k in known})
        settings.apply_env_overrides()
        return settings.validated()

    def apply_env_overrides(self) -> None:
        """``LISTN_*`` environment variables win over settings.json."""
        for var, attr in (("LISTN_COMPUTE_TYPE", "compute_type"),
                          ("LISTN_CPU_THREADS", "cpu_threads"),
                          ("LISTN_BEAM_SIZE", "beam_size")):
            raw = os.environ.get(var)
            if raw is not None and raw.strip():
                setattr(self, attr, raw.strip())

    def validated(self) -> "Settings":
        if self.model not in MODELS:
            self.model = "small"
        if self.backend not in BACKENDS:
            self.backend = "auto"
        # Performance knobs fail loudly instead of silently reverting: an
        # invalid value must be impossible to miss when tuning for speed.
        if self.compute_type is not None:
            self.compute_type = str(self.compute_type).strip().lower()
        else:
            self.compute_type = DEFAULT_COMPUTE_TYPE
        if self.compute_type not in COMPUTE_TYPES:
            raise SettingsError(
                f"compute_type must be one of {', '.join(COMPUTE_TYPES)} — "
                f"got {self.compute_type!r}.")
        raw_threads = self.cpu_threads
        if raw_threads is None or (isinstance(raw_threads, str) and not raw_threads.strip()):
            self.cpu_threads = None
        else:
            try:
                threads = int(str(raw_threads).strip())
            except (TypeError, ValueError):
                threads = 0
            if threads < 1:
                raise SettingsError(
                    "cpu_threads must be a whole number of 1 or more "
                    f"(unset it to let the backend choose) — got {raw_threads!r}.")
            self.cpu_threads = threads
        raw_beam = self.beam_size
        try:
            beam = int(str(raw_beam).strip())
        except (TypeError, ValueError):
            beam = 0
        if not 1 <= beam <= 100:
            raise SettingsError(
                f"beam_size must be a whole number between 1 and 100 — got {raw_beam!r}.")
        self.beam_size = beam
        try:
            self.chunk_seconds = min(30.0, max(2.0, float(self.chunk_seconds)))
        except (TypeError, ValueError):
            self.chunk_seconds = 6.0
        from listn.shortcuts import ShortcutError, normalize_chord

        self.captions_enabled = bool(self.captions_enabled)
        for attr in ("hotkey_start_stop", "hotkey_bookmark"):
            try:
                setattr(self, attr, normalize_chord(getattr(self, attr)))
            except ShortcutError as exc:
                raise SettingsError(f"{attr}: {exc}") from None
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
        self.diarize_after = None if self.diarize_after is None else bool(self.diarize_after)
        try:
            self.speaker_threshold = min(SPEAKER_THRESHOLD_MAX, max(SPEAKER_THRESHOLD_MIN, float(self.speaker_threshold)))
        except (TypeError, ValueError):
            self.speaker_threshold = SPEAKER_THRESHOLD
        try:
            self.spectrum_threshold = min(0.95, max(0.4, float(self.spectrum_threshold)))
        except (TypeError, ValueError):
            self.spectrum_threshold = 0.72
        self.settings_version = SETTINGS_VERSION
        try:
            self.audio_retention_days = min(3650, max(0, int(self.audio_retention_days)))
        except (TypeError, ValueError):
            self.audio_retention_days = 0
        try:
            self.min_free_gb = min(100.0, max(0.0, float(self.min_free_gb)))
        except (TypeError, ValueError):
            self.min_free_gb = 2.0
        self.battery_saver = bool(self.battery_saver)
        return self

    def save(self) -> None:
        path = self.path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.chmod(0o600)
        tmp.replace(path)

    def resolved_hf_token(self) -> str | None:
        return self.hf_token or os.environ.get("HF_TOKEN") or None

    def speakers_after(self) -> bool:
        """Label speakers when a recording ends. Auto means: a token is set or the model is cached."""
        if self.diarize_after is not None:
            return bool(self.diarize_after)
        from listn import diarize

        if not diarize.installed():
            return False
        return bool(self.resolved_hf_token()) or diarize.cached_pipeline() is not None

    def public(self) -> dict:
        data = asdict(self)
        data["hf_token"] = bool(self.resolved_hf_token())
        data["diarize_after_auto"] = self.diarize_after is None
        data["diarize_after"] = self.speakers_after()
        return data


def migrate(raw: dict) -> dict:
    """Bring a settings file from an older listn up to date."""
    raw = dict(raw)
    try:
        version = int(raw.get("settings_version") or 1)
    except (TypeError, ValueError):
        version = 1
    if version < 2:
        # 0.4 saved diarize_after=false for everyone; it now follows the token.
        if raw.get("diarize_after") is not True:
            raw["diarize_after"] = None
        # The old threshold was on the spectrum scale, which means nothing for WeSpeaker.
        if "speaker_threshold" in raw:
            raw["spectrum_threshold"] = raw.pop("speaker_threshold")
        raw["speaker_threshold"] = SPEAKER_THRESHOLD
    raw["settings_version"] = SETTINGS_VERSION
    return raw
