"""Delete old audio while keeping the transcript and notes.

``audio_retention_days`` of 0 keeps everything. A session that is still
recording or transcribing is never touched, and neither is a file that is
not audio.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from listn.library import AUDIO_EXTS, Library


def apply_retention(library: Library, days: int, now: datetime | None = None) -> list[str]:
    try:
        days = int(days)
    except (TypeError, ValueError):
        return []
    if days <= 0:
        return []
    moment = now or datetime.now()
    cutoff = moment - timedelta(days=days)
    removed: list[str] = []
    for meta in library.list():
        if meta.status in ("recording", "transcribing"):
            continue
        try:
            created = datetime.fromisoformat(meta.created_at)
        except (TypeError, ValueError):
            continue
        if created > cutoff:
            continue
        root = library.session_path(meta.id)
        deleted = False
        if root.exists():
            for path in list(root.iterdir()):
                if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                    path.unlink()
                    deleted = True
        if deleted or meta.audio_file or not meta.audio_removed:
            if deleted or meta.audio_file:
                library.update(meta.id, audio_file=None, audio_removed=True)
                removed.append(meta.id)
    return removed
