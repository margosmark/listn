"""Read-only status snapshot for the local dashboard.

Assembles what the engine already tracks — the active live session, the
running transcription/speaker jobs and the stored library — into one JSON
document for ``GET /api/dashboard``. There is no separate dashboard state,
no mock data and nothing is written or sent anywhere: the values come from
the same ``Engine``/``Library`` objects every other route reads, behind the
server's loopback and origin middleware.

Honest degradation is part of the contract: with no recordings the result
is ``recordings: []`` (plus a real total), with nothing running ``live`` is
``None`` — never an empty shell that looks like data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from listn import __version__
from listn.config import data_dir

# The display vocabulary the dashboard promises (recording / processing /
# failed). The raw engine status rides along untouched so nothing is lost
# in translation.
STATUS_LABELS = {
    "recording": "Recording",
    "transcribing": "Processing",
    "done": "Done",
    "error": "Failed",
}

TRACK_FILES = (("system", "audio-system.wav"), ("mic", "audio-mic.wav"), ("mixed", "audio.wav"))


def status_label(status: str | None) -> str:
    """Human label for an engine session status; unknown values are shown as-is."""
    if not status:
        return "Unknown"
    return STATUS_LABELS.get(status, str(status).replace("-", " ").capitalize())


def track_files(library, meta) -> dict:
    """Which separated tracks are on disk for one session.

    Reports only files the engine itself wrote into the session directory —
    the same names ``LiveSession`` uses (``audio-system.wav`` /
    ``audio-mic.wav`` / the ``audio.wav`` device recording or ingest
    mixdown) plus any imported file under another name.
    """
    try:
        root: Path | None = library.session_path(meta.id)
    except KeyError:
        root = None
    present = {key: bool(root is not None and (root / name).exists())
               for key, name in TRACK_FILES}
    other = None
    audio_file = meta.audio_file or None
    if audio_file and audio_file not in {name for _, name in TRACK_FILES}:
        other = audio_file if (root is not None and (root / audio_file).exists()) else None
    return {"system": present["system"], "mic": present["mic"],
            "mixed": present["mixed"], "other": other}


def track_summary(tracks: dict) -> str:
    """One-line description of a session's track separation, or its absence."""
    if tracks["system"] and tracks["mic"]:
        base = "System + microphone (separate tracks)"
    elif tracks["system"] or tracks["mic"]:
        base = "One separated track"
    elif tracks["mixed"]:
        base = "One track"
    elif tracks["other"]:
        base = f"Imported file: {tracks['other']}"
    else:
        return "No audio kept"
    return base + (" + mixdown" if (tracks["system"] or tracks["mic"]) and tracks["mixed"] else "")


def _transcription_state(meta, has_job: bool) -> str:
    if has_job:
        return "processing"
    if meta.status == "recording":
        return "recording"
    if meta.status == "transcribing":
        return "processing"
    if meta.status == "error":
        return "failed"
    if meta.status == "done":
        return "done"
    return "unknown"


def recording_row(engine, meta, live_id: str | None = None) -> dict:
    """One recent recording with tracks, transcription and speaker state."""
    library = engine.library
    job = engine.jobs.get(meta.id)
    tracks = track_files(library, meta)
    try:
        segments = len(library.segments(meta.id))
    except OSError:
        segments = 0
    return {
        "id": meta.id,
        "title": meta.title,
        "created_at": meta.created_at,
        "source": meta.source,
        "source_name": meta.source_name,
        "status": meta.status,
        "status_label": status_label(meta.status),
        "duration": meta.duration,
        "error": meta.error,
        "warning": meta.warning,
        "silent": meta.silent,
        "audio_removed": meta.audio_removed,
        "live": bool(live_id and meta.id == live_id),
        "tracks": tracks,
        "track_summary": track_summary(tracks),
        "transcription": {
            "state": _transcription_state(meta, job is not None),
            "segments": segments,
            "model": meta.model,
            "backend": meta.backend,
            "job": None if job is None else job.to_dict(),
        },
        "speakers": {"state": meta.speaker_state, "note": meta.speaker_note},
    }


def build_snapshot(engine, pipewire: dict | None = None, limit: int = 20) -> dict:
    """The whole dashboard document: live status, jobs, recent recordings.

    ``pipewire`` is the live ``PipeWireLive.status()`` mapping when a
    two-track capture is running (``None`` otherwise); it is passed in so
    this module never reaches into server state.
    """
    live_id = engine.live.meta.id if engine.live is not None and engine.live_active() else None
    live = None
    if engine.live is not None and engine.live_active():
        live = engine.live.status()
        live["active"] = True
        live["tracks"] = (pipewire or {}).get("tracks")
    items = engine.library.list()
    recordings = [recording_row(engine, meta, live_id) for meta in items[:limit]]
    return {
        "version": __version__,
        "platform": sys.platform,
        "data_dir": str(data_dir()),
        "live": live,
        "pipewire": pipewire,
        "jobs": [job.to_dict() for job in engine.jobs.values()],
        "recordings": recordings,
        "recordings_total": len(items),
    }
