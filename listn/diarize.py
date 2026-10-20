"""Optional speaker diarization with pyannote.audio.

Diarization runs over a whole recording, never per live chunk: pyannote
labels are only consistent within one call, so per-chunk labels would assign
SPEAKER_00 to different people every few seconds. Live sessions get speakers
after you stop recording.

The pyannote model is gated on Hugging Face: a token is needed for the first
download only. After that it loads from the local cache.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from listn.audio import TARGET_RATE
from listn.library import Segment

PIPELINE_ID = "pyannote/speaker-diarization-3.1"
Turn = tuple[float, float, str]

_pipeline = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import pyannote.audio  # noqa: F401

        return True
    except Exception:
        return False


def load_pipeline(token: str | None):
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("Speaker labels need the diarization extra: pip install 'listn[diarize]'") from exc
        try:
            pipeline = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=token)
        if pipeline is None:
            raise RuntimeError(
                "Could not load the pyannote model. On first use, accept the terms at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                "https://huggingface.co/pyannote/segmentation-3.0, then add a Hugging Face token in Settings."
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                pipeline.to(torch.device("mps"))
            except Exception:
                pass
        _pipeline = pipeline
        return pipeline


def turns_from_output(output) -> list[Turn]:
    """Accept both pyannote 3.x (Annotation) and 4.x (DiarizeOutput)."""
    annotation = getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        return []
    return [(float(t.start), float(t.end), str(label)) for t, _, label in annotation.itertracks(yield_label=True)]


def run(pipeline, audio: np.ndarray, rate: int = TARGET_RATE) -> list[Turn]:
    import torch

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
    return turns_from_output(pipeline({"waveform": waveform, "sample_rate": rate}))


def assign_speakers(segments: list[Segment], turns: list[Turn], max_gap: float = 1.0) -> list[Segment]:
    """Label each segment with the speaker who overlaps it most.

    Falls back to the nearest turn within ``max_gap`` seconds, so segments
    whose midpoint lands in a short pause still get a label.
    """
    out = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for start, end, label in turns:
            ov = min(seg.end, end) - max(seg.start, start)
            if ov > 0:
                overlap[label] = overlap.get(label, 0.0) + ov
        speaker = None
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


def friendly_names(segments: list[Segment]) -> dict[str, str]:
    """SPEAKER_00 -> 'Speaker 1', ordered by first appearance."""
    names: dict[str, str] = {}
    for seg in segments:
        if seg.speaker and seg.speaker not in names:
            names[seg.speaker] = f"Speaker {len(names) + 1}"
    return names


def diarize_segments(audio: np.ndarray, segments: list[Segment], token: str | None,
                     status: Callable[[str], None] | None = None) -> tuple[list[Segment], dict[str, str]]:
    if status:
        status("Loading speaker model...")
    pipeline = load_pipeline(token)
    if status:
        status("Identifying speakers...")
    labelled = assign_speakers(segments, run(pipeline, audio))
    return labelled, friendly_names(labelled)
