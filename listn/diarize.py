"""Speaker diarization with pyannote.audio 4.

Diarization runs over a whole recording, never per live chunk: pyannote
labels are only consistent within one call, so per-chunk labels would assign
SPEAKER_00 to different people every few seconds. Live sessions get speakers
after you stop recording, on the Others track only (Me is the microphone).

``pyannote/speaker-diarization-community-1`` is tried first and
``pyannote/speaker-diarization-3.1`` is the fallback. Both are gated on
Hugging Face: the token is passed to the one download and nowhere else.
Every later load reads the local cache with the Hub offline.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from listn.audio import TARGET_RATE
from listn.hub import enable_hub_offline, is_cached, load_local_or_download, redact
from listn.library import Segment

log = logging.getLogger("listn")

COMMUNITY_ID = "pyannote/speaker-diarization-community-1"
LEGACY_ID = "pyannote/speaker-diarization-3.1"
PIPELINE_IDS = (COMMUNITY_ID, LEGACY_ID)
# Model pages whose terms have to be accepted before the first download.
TERMS_PAGES = {
    COMMUNITY_ID: (COMMUNITY_ID,),
    LEGACY_ID: (LEGACY_ID, "pyannote/segmentation-3.0"),
}
INSTALL_HINT = "Speaker labels need the diarize extra. The app includes it; from source run: uv sync --extra diarize"
Turn = tuple[float, float, str]

# Progress through one pyannote run, by hook step: (start, end) of the bar.
STEP_SPAN = {
    "segmentation": (0.0, 0.35),
    "speaker_counting": (0.35, 0.4),
    "embeddings": (0.4, 0.9),
    "discrete_diarization": (0.9, 0.97),
}

_pipelines: dict[str, object] = {}
_lock = threading.Lock()


class DiarizationUnavailable(RuntimeError):
    """The model can't be loaded: extra missing, no token for the first download, or terms not accepted."""


class Cancelled(Exception):
    pass


def installed() -> bool:
    """pyannote.audio is importable. Checked without importing torch."""
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):
        return False


available = installed


def telemetry_off() -> None:
    """pyannote 4 sends usage metrics to otel.pyannote.ai unless this is off before import."""
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
    except Exception:
        return
    try:
        set_telemetry_metrics(False)
    except Exception:
        pass


def cached_pipeline() -> str | None:
    for model_id in PIPELINE_IDS:
        if is_cached(model_id, "config.yaml"):
            return model_id
    return None


def terms_urls(model_ids=PIPELINE_IDS) -> list[str]:
    pages: list[str] = []
    for model_id in model_ids:
        for page in TERMS_PAGES.get(model_id, (model_id,)):
            url = f"https://huggingface.co/{page}"
            if url not in pages:
                pages.append(url)
    return pages


def readiness(token: str | None) -> dict:
    """What the UI shows about speaker labels. Never touches the network."""
    is_installed = installed()
    cached = cached_pipeline() if is_installed else None
    has_token = bool(token)
    loaded = next(iter(_pipelines), None)
    if not is_installed:
        message = INSTALL_HINT
    elif cached or loaded:
        message = f"Speaker model is on this machine ({(loaded or cached).split('/')[-1]}). It loads offline."
    elif has_token:
        message = ("The speaker model downloads once, the first time speakers are labelled. "
                   f"Accept the terms at {terms_urls((COMMUNITY_ID,))[0]} first.")
    else:
        message = ("Add a Hugging Face token in Settings to download the speaker model once. "
                   f"Accept the terms at {terms_urls((COMMUNITY_ID,))[0]} first.")
    return {
        "installed": is_installed,
        "token": has_token,
        "cached": cached,
        "loaded": loaded,
        "ready": bool(is_installed and (cached or loaded or has_token)),
        # Ready only because a token is set: nothing is on disk yet, so the first
        # labelling is what downloads the model. Readiness and "will download"
        # are different facts and the UI shows them separately.
        "needs_download": bool(is_installed and has_token and not (cached or loaded)),
        "message": message,
        "terms": terms_urls(),
    }


def pick_device():
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _import_stack() -> None:
    telemetry_off()
    try:
        import torch  # noqa: F401
        import pyannote.audio  # noqa: F401
    except Exception as exc:
        raise DiarizationUnavailable(f"pyannote.audio could not be imported ({exc}). {INSTALL_HINT}") from exc
    telemetry_off()


def _from_pretrained(model_id: str, token: str | None):
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_id, token=token) if token else Pipeline.from_pretrained(model_id)
    if pipeline is None:
        raise DiarizationUnavailable(f"{model_id} did not load.")
    return pipeline


def _load_one(model_id: str, token: str | None, on_download: Callable[[], None] | None):
    def local():
        return _from_pretrained(model_id, None)

    def remote():
        if not token:
            raise DiarizationUnavailable("no token for the first download")
        return _from_pretrained(model_id, token)

    return load_local_or_download(local, remote, on_download=on_download)


def _to_device(pipeline):
    try:
        device = pick_device()
    except Exception:
        return pipeline, "cpu"
    if device.type == "cpu":
        return pipeline, "cpu"
    try:
        pipeline.to(device)
        return pipeline, device.type
    except Exception as exc:
        log.info("Speaker model stays on the CPU: %s", exc)
        return pipeline, "cpu"


def load_pipeline(token: str | None, status: Callable[[str], None] | None = None):
    """Return ``(pipeline, model_id)``, community-1 first, 3.1 if that fails."""
    with _lock:
        for model_id in PIPELINE_IDS:
            if model_id in _pipelines:
                return _pipelines[model_id], model_id
        if not installed():
            raise DiarizationUnavailable(INSTALL_HINT)
        _import_stack()
        errors: list[str] = []
        for model_id in PIPELINE_IDS:
            def downloading(model_id=model_id):
                if status:
                    status(f"Downloading {model_id.split('/')[-1]} (one time)...")

            try:
                pipeline = _load_one(model_id, token, downloading)
            except Exception as exc:
                message = redact(exc, token).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
                log.info("%s did not load: %s", model_id, message)
                errors.append(f"{model_id.split('/')[-1]}: {message}")
                continue
            pipeline, device = _to_device(pipeline)
            log.info("Speaker model %s on %s", model_id, device)
            _pipelines[model_id] = pipeline
            return pipeline, model_id
        enable_hub_offline()
        if not token:
            where = (
                f"The saved speaker model didn't load ({'; '.join(errors)})."
                if cached_pipeline()
                else "The speaker model isn't on this machine yet."
            )
            raise DiarizationUnavailable(
                f"{where} Add a Hugging Face token in Settings for the one-time download, after accepting the "
                f"terms at {terms_urls((COMMUNITY_ID,))[0]}."
            )
        raise DiarizationUnavailable(
            "Could not load a speaker model. Check that the token can read gated models and that the terms are "
            f"accepted at {', '.join(terms_urls())}. Details: {'; '.join(errors)}"
        )


def reset() -> None:
    with _lock:
        _pipelines.clear()


def _annotation_turns(annotation) -> list[Turn]:
    if not hasattr(annotation, "itertracks"):
        return []
    return [(float(t.start), float(t.end), str(label)) for t, _, label in annotation.itertracks(yield_label=True)]


def turns_from_output(output) -> list[Turn]:
    """Accept pyannote 3.x (Annotation, or (Annotation, embeddings)) and 4.x (DiarizeOutput).

    4.x also has an exclusive diarization with one speaker at a time, which
    lines up with a transcript better, so it is preferred.
    """
    if isinstance(output, tuple) and output:
        output = output[0]
    for name in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, name, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return _annotation_turns(annotation)
    return _annotation_turns(output)


def progress_hook(progress: Callable[[float], None] | None, cancelled: Callable[[], bool] | None = None):
    """A pyannote ``hook`` that maps its steps onto 0..1 and stops when ``cancelled``."""

    def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
        if cancelled is not None and cancelled():
            raise Cancelled()
        if progress is None:
            return
        start, end = STEP_SPAN.get(step_name, (None, None))
        if start is None:
            return
        if total and completed is not None:
            progress(start + (end - start) * min(1.0, max(0.0, completed / total)))
        else:
            progress(end if step_artifact is not None else start)

    return hook


def _call(pipeline, audio: np.ndarray, rate: int, hook):
    import torch

    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1))
    try:
        return pipeline({"waveform": waveform, "sample_rate": rate}, hook=hook)
    except TypeError:
        return pipeline({"waveform": waveform, "sample_rate": rate})


def run(pipeline, audio: np.ndarray, rate: int = TARGET_RATE,
        progress: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None) -> list[Turn]:
    hook = progress_hook(progress, cancelled)
    try:
        output = _call(pipeline, audio, rate, hook)
    except Cancelled:
        raise
    except Exception as exc:
        device = getattr(pipeline, "device", None)
        if getattr(device, "type", "cpu") == "cpu":
            raise
        import torch

        log.info("Speaker model failed on %s (%s); retrying on the CPU", device, exc)
        pipeline.to(torch.device("cpu"))
        output = _call(pipeline, audio, rate, hook)
    if progress:
        progress(1.0)
    return turns_from_output(output)


def assign_speakers(segments: list[Segment], turns: list[Turn], max_gap: float = 1.0,
                    default: str | None = None) -> list[Segment]:
    """Label each segment with the speaker who overlaps it most.

    Falls back to the nearest turn within ``max_gap`` seconds, so segments
    whose midpoint lands in a short pause still get a label. Segments with
    no speech nearby get ``default``.
    """
    out = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for start, end, label in turns:
            ov = min(seg.end, end) - max(seg.start, start)
            if ov > 0:
                overlap[label] = overlap.get(label, 0.0) + ov
        speaker = default
        if overlap:
            speaker = max(overlap.items(), key=lambda kv: kv[1])[0]
        elif turns:
            def gap(t: Turn) -> float:
                return max(t[0] - seg.end, seg.start - t[1], 0.0)

            nearest = min(turns, key=gap)
            if gap(nearest) <= max_gap:
                speaker = nearest[2]
        out.append(Segment(seg.start, seg.end, seg.text, speaker))
    return out


def friendly_names(segments: list[Segment], skip: tuple[str, ...] = ("Me", "Others")) -> dict[str, str]:
    """SPEAKER_00 -> 'Speaker 1', ordered by first appearance."""
    names: dict[str, str] = {}
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.speaker and seg.speaker not in skip and seg.speaker not in names:
            names[seg.speaker] = f"Speaker {len(names) + 1}"
    return names


def speaker_audio(audio: np.ndarray, turns: list[Turn], label: str, rate: int = TARGET_RATE,
                  max_seconds: float = 30.0, min_turn: float = 0.5) -> np.ndarray:
    """Up to ``max_seconds`` of one speaker, longest turns first."""
    picked = sorted((t for t in turns if t[2] == label and t[1] - t[0] >= min_turn),
                    key=lambda t: t[1] - t[0], reverse=True)
    pieces, total = [], 0
    limit = int(max_seconds * rate)
    for start, end, _ in picked:
        a, b = max(0, int(start * rate)), min(audio.size, int(end * rate))
        if b <= a:
            continue
        piece = audio[a:min(b, a + limit - total)]
        pieces.append(piece)
        total += piece.size
        if total >= limit:
            break
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, dtype=np.float32)


@dataclass
class DiarizeResult:
    segments: list[Segment]
    names: dict[str, str]
    model_id: str
    turns: list[Turn] = field(default_factory=list)
    matches: dict[str, dict] = field(default_factory=dict)


def diarize_segments(audio: np.ndarray, segments: list[Segment], token: str | None,
                     status: Callable[[str], None] | None = None,
                     progress: Callable[[float], None] | None = None,
                     cancelled: Callable[[], bool] | None = None,
                     default: str | None = None,
                     identify: Callable[[np.ndarray, list[Turn], list[str]], dict[str, dict]] | None = None,
                     ) -> DiarizeResult:
    """Label ``segments`` from ``audio`` and, with ``identify``, name known voices."""
    if status:
        status("Loading speaker model...")
    pipeline, model_id = load_pipeline(token, status=status)
    if status:
        status("Identifying speakers...")
    turns = run(pipeline, audio, progress=progress, cancelled=cancelled)
    labelled = assign_speakers(segments, turns, default=default)
    names = friendly_names(labelled)
    matches: dict[str, dict] = {}
    if identify is not None and names:
        if status:
            status("Matching known voices...")
        try:
            matches = identify(audio, turns, list(names))
        except Exception as exc:
            log.info("Known-voice matching skipped: %s", exc)
            matches = {}
        for label, found in matches.items():
            if found.get("name"):
                names[label] = found["name"]
    return DiarizeResult(labelled, names, model_id, turns, matches)
