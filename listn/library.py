"""On-disk library of recordings and transcripts.

Layout under the data directory::

    sessions/<id>/meta.json        title, source, model, status, duration...
    sessions/<id>/transcript.json  list of segments
    sessions/<id>/audio.<ext>      recording or imported file (optional)
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from listn.config import data_dir

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mov", ".webm", ".aiff", ".aif"}
EXPORT_FORMATS = ("txt", "md", "srt", "vtt", "json")
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(float(d["start"]), float(d["end"]), str(d["text"]), d.get("speaker"))


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: str
    source: str  # "live" or "file"
    source_name: str | None = None
    model: str | None = None
    backend: str | None = None
    language: str | None = None
    duration: float = 0.0
    status: str = "done"  # recording | transcribing | done | error
    error: str | None = None
    audio_file: str | None = None
    diarized: bool = False
    speakers: dict = field(default_factory=dict)
    owner_pid: int | None = None
    attendees: list = field(default_factory=list)
    audio_removed: bool = False
    silent: bool = False
    warning: str | None = None
    # Speaker labels after the transcript: queued | running | done | skipped | error | cancelled
    speaker_state: str | None = None
    speaker_note: str | None = None
    speaker_model: str | None = None
    speaker_matches: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def new_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _label(seg: Segment, speakers: dict) -> str | None:
    if not seg.speaker:
        return None
    return speakers.get(seg.speaker) or seg.speaker


def export(meta: SessionMeta, segments: list[Segment], fmt: str) -> str:
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose one of: {', '.join(EXPORT_FORMATS)}")
    sp = meta.speakers or {}
    if fmt == "json":
        return json.dumps({"meta": meta.to_dict(), "segments": [s.to_dict() for s in segments]}, indent=2, ensure_ascii=False)
    if fmt == "txt":
        lines = []
        for s in segments:
            who = _label(s, sp)
            lines.append(f"[{fmt_clock(s.start)}] " + (f"{who}: " if who else "") + s.text)
        return "\n".join(lines) + ("\n" if lines else "")
    if fmt == "md":
        out = [f"# {meta.title}", "", f"*{meta.created_at[:16].replace('T', ' ')} · {fmt_clock(meta.duration)}*", ""]
        for s in segments:
            who = _label(s, sp)
            out.append(f"**[{fmt_clock(s.start)}]** " + (f"**{who}:** " if who else "") + s.text + "  ")
        return "\n".join(out) + "\n"
    if fmt == "srt":
        blocks = []
        for i, s in enumerate(segments, 1):
            who = _label(s, sp)
            text = f"{who}: {s.text}" if who else s.text
            blocks.append(f"{i}\n{_fmt_ts(s.start, ',')} --> {_fmt_ts(s.end, ',')}\n{text}\n")
        return "\n".join(blocks)
    blocks = ["WEBVTT\n"]
    for s in segments:
        who = _label(s, sp)
        text = f"<v {who}>{s.text}" if who else s.text
        blocks.append(f"{_fmt_ts(s.start, '.')} --> {_fmt_ts(s.end, '.')}\n{text}\n")
    return "\n".join(blocks)


class Library:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else data_dir() / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _dir(self, session_id: str) -> Path:
        if not _ID_RE.match(session_id or ""):
            raise KeyError(session_id)
        return self.root / session_id

    @staticmethod
    def _write_json(path: Path, data) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)

    def create(self, title: str, source: str, **kwargs) -> SessionMeta:
        with self._lock:
            sid = new_id()
            while (self.root / sid).exists():
                sid = new_id()
            (self.root / sid).mkdir(parents=True)
            meta = SessionMeta(id=sid, title=title, created_at=datetime.now().isoformat(timespec="seconds"),
                               source=source, owner_pid=os.getpid(), **kwargs)
            self.save_meta(meta)
            self.save_segments(sid, [])
            return meta

    def session_path(self, session_id: str) -> Path:
        return self._dir(session_id)

    def audio_path(self, meta: SessionMeta) -> Path | None:
        if not meta.audio_file:
            return None
        p = self._dir(meta.id) / meta.audio_file
        return p if p.exists() else None

    def new_audio_path(self, session_id: str, ext: str = ".wav") -> Path:
        return self._dir(session_id) / f"audio{ext.lower()}"

    def import_audio(self, meta: SessionMeta, src: Path, move: bool = False) -> Path:
        ext = Path(src).suffix.lower() or ".wav"
        dest = self.new_audio_path(meta.id, ext)
        if move:
            shutil.move(str(src), dest)
        else:
            shutil.copy2(src, dest)
        meta.audio_file = dest.name
        self.save_meta(meta)
        return dest

    def save_meta(self, meta: SessionMeta) -> None:
        with self._lock:
            self._write_json(self._dir(meta.id) / "meta.json", meta.to_dict())

    def save_segments(self, session_id: str, segments: list[Segment]) -> None:
        with self._lock:
            self._write_json(self._dir(session_id) / "transcript.json", [s.to_dict() for s in segments])

    def save_notes(self, session_id: str, notes: dict) -> None:
        with self._lock:
            self._write_json(self._dir(session_id) / "summary.json", notes)

    def notes(self, session_id: str) -> dict | None:
        try:
            raw = json.loads((self._dir(session_id) / "summary.json").read_text())
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def save_bookmarks(self, session_id: str, bookmarks: list[dict]) -> None:
        with self._lock:
            self._write_json(self._dir(session_id) / "bookmarks.json", bookmarks)

    def bookmarks(self, session_id: str) -> list[dict]:
        try:
            raw = json.loads((self._dir(session_id) / "bookmarks.json").read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                t = float(item.get("t", 0))
            except (TypeError, ValueError):
                continue
            out.append({"t": round(t, 2), "note": str(item.get("note") or "Bookmark")[:200]})
        return out

    def get(self, session_id: str) -> SessionMeta:
        path = self._dir(session_id) / "meta.json"
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise KeyError(session_id) from exc
        known = SessionMeta.__dataclass_fields__
        return SessionMeta(**{k: v for k, v in raw.items() if k in known})

    def segments(self, session_id: str) -> list[Segment]:
        try:
            raw = json.loads((self._dir(session_id) / "transcript.json").read_text())
        except (OSError, ValueError):
            return []
        return [Segment.from_dict(d) for d in raw]

    def list(self) -> list[SessionMeta]:
        items = []
        for child in self.root.iterdir():
            if child.is_dir() and _ID_RE.match(child.name):
                try:
                    items.append(self.get(child.name))
                except KeyError:
                    continue
        items.sort(key=lambda m: m.created_at, reverse=True)
        return items

    def delete(self, session_id: str) -> None:
        with self._lock:
            path = self._dir(session_id)
            if not path.exists():
                raise KeyError(session_id)
            shutil.rmtree(path)

    def update(self, session_id: str, **changes) -> SessionMeta:
        with self._lock:
            meta = self.get(session_id)
            for k, v in changes.items():
                if k in SessionMeta.__dataclass_fields__ and k not in ("id", "owner_pid"):
                    setattr(meta, k, v)
            if changes.get("status") in ("recording", "transcribing"):
                meta.owner_pid = os.getpid()
            self.save_meta(meta)
            return meta

    def recover_interrupted(self) -> int:
        """Mark sessions left 'recording'/'transcribing' by a dead process as errors.

        Sessions owned by another live listn process (e.g. the UI recording
        while a CLI command starts) are left alone.
        """
        count = 0
        for meta in self.list():
            if meta.status in ("recording", "transcribing") and not _pid_alive(meta.owner_pid):
                meta.status = "error"
                meta.error = "listn stopped before this session finished. The audio recorded so far is kept; use Retranscribe."
                self.save_meta(meta)
                count += 1
            elif meta.speaker_state in ("queued", "running") and not _pid_alive(meta.owner_pid):
                meta.speaker_state = "error"
                meta.speaker_note = "listn stopped before speakers were labelled. Use Label speakers to try again."
                self.save_meta(meta)
                count += 1
        return count

    def search(self, query: str, limit: int = 50) -> list[dict]:
        q = query.strip().lower()
        results = []
        for meta in self.list():
            segs = self.segments(meta.id)
            hits = [s for s in segs if q in s.text.lower()] if q else []
            title_hit = bool(q) and q in meta.title.lower()
            if not q or hits or title_hit:
                results.append({
                    "meta": meta.to_dict(),
                    "hits": [s.to_dict() for s in hits[:5]],
                    "hit_count": len(hits),
                    "preview": " ".join(s.text for s in segs[:3])[:200],
                })
            if len(results) >= limit:
                break
        return results
