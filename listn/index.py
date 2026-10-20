"""Incremental semantic index over every saved transcript."""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np

from listn.config import data_dir
from listn.embed import cosine, embed, tokens
from listn.library import Library

# A shared word or two clears this; unrelated strings stay under it.
SEMANTIC_MIN = 0.34


def _fingerprint(title: str, segments) -> str:
    digest = hashlib.sha256()
    digest.update(title.encode("utf-8"))
    for segment in segments:
        digest.update(f"\n{segment.start:.2f}|{segment.speaker or ''}|{segment.text}".encode("utf-8"))
    return digest.hexdigest()


class SearchIndex:
    def __init__(self, library: Library, path=None):
        self.library = library
        self.path = path or (data_dir() / "search-index.json")
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"sessions": {}}
        if not isinstance(raw, dict) or not isinstance(raw.get("sessions"), dict):
            return {"sessions": {}}
        return raw

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)

    def sync(self) -> dict:
        """Re-embed sessions whose transcript changed. Returns counts."""
        with self._lock:
            seen = set()
            updated = 0
            for meta in self.library.list():
                seen.add(meta.id)
                segments = self.library.segments(meta.id)
                fingerprint = _fingerprint(meta.title, segments)
                current = self._data["sessions"].get(meta.id)
                if current and current.get("fingerprint") == fingerprint:
                    continue
                rows = []
                for segment in segments:
                    text = (segment.text or "").strip()
                    if not text or text == "[paused]":
                        continue
                    rows.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": text,
                        "speaker": segment.speaker,
                        "vec": [round(float(v), 5) for v in embed(text)],
                    })
                self._data["sessions"][meta.id] = {
                    "fingerprint": fingerprint,
                    "title": meta.title,
                    "rows": rows,
                }
                updated += 1
            removed = [sid for sid in list(self._data["sessions"]) if sid not in seen]
            for sid in removed:
                del self._data["sessions"][sid]
            if updated or removed:
                self._save()
            return {"updated": updated, "removed": len(removed), "sessions": len(seen)}

    def search(self, query: str, limit: int = 12) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            self.sync()
            query_vec = embed(query)
            query_tokens = set(tokens(query))
            hits = []
            for session_id, session in self._data["sessions"].items():
                title = session.get("title") or ""
                title_tokens = set(tokens(title))
                for row in session.get("rows") or []:
                    text = row["text"]
                    keyword = query.lower() in text.lower() or query.lower() in title.lower()
                    row_tokens = set(tokens(text)) | title_tokens
                    overlap = len(query_tokens & row_tokens)
                    score = cosine(query_vec, np.asarray(row["vec"], dtype=np.float32))
                    if keyword:
                        score += 0.5
                    elif overlap == 0 or score < SEMANTIC_MIN:
                        continue
                    hits.append({
                        "session_id": session_id,
                        "title": title,
                        "start": row["start"],
                        "end": row["end"],
                        "text": text,
                        "speaker": row.get("speaker"),
                        "score": round(float(score), 4),
                        "keyword": keyword,
                    })
            hits.sort(key=lambda hit: (-hit["score"], hit["start"]))
            return hits[:limit]

    def sessions_for(self, query: str, limit: int = 50) -> list[dict]:
        """Keyword results, plus sessions that only match semantically."""
        keyword = {item["meta"]["id"]: item for item in self.library.search(query, limit=limit)}
        if not query.strip():
            return list(keyword.values())
        grouped: dict[str, dict] = {}
        for hit in self.search(query, limit=limit * 3):
            bucket = grouped.setdefault(hit["session_id"], {"hits": [], "best": 0.0})
            bucket["best"] = max(bucket["best"], hit["score"])
            if len(bucket["hits"]) < 5:
                bucket["hits"].append(hit)
        ordered = sorted(grouped, key=lambda sid: grouped[sid]["best"], reverse=True)
        results = []
        seen = set()
        for session_id in ordered:
            base = keyword.get(session_id)
            if base is None:
                try:
                    meta = self.library.get(session_id)
                except KeyError:
                    continue
                segs = self.library.segments(session_id)
                base = {
                    "meta": meta.to_dict(),
                    "hits": [],
                    "hit_count": 0,
                    "preview": " ".join(s.text for s in segs[:3])[:200],
                }
            semantic = grouped[session_id]["hits"]
            if not base["hits"]:
                base = dict(base)
                base["hits"] = [{k: hit[k] for k in ("start", "end", "text", "speaker")} for hit in semantic[:5]]
                base["hit_count"] = len(semantic)
            base = dict(base)
            base["semantic"] = True
            results.append(base)
            seen.add(session_id)
            if len(results) >= limit:
                break
        for session_id, item in keyword.items():
            if session_id not in seen:
                results.append(item)
            if len(results) >= limit:
                break
        return results
