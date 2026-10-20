"""Local speaker profiles.

An embedding is the shape of a short clip's spectrum: 64 log-magnitude
bands, mean-centred and normalised. No model is downloaded. Profiles live
in ``speakers/<name>.json`` under the data directory. A match has to clear
``speaker_threshold`` (default 0.72). Correcting a name in the library adds
that clip to the profile, so the next meeting can recognise them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from listn.audio import TARGET_RATE, read_wav
from listn.config import data_dir
from listn.library import Library

BANDS = 64


def _slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return text or "speaker"


def embed_speaker(audio: np.ndarray, rate: int = TARGET_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < int(0.2 * rate):
        audio = np.pad(audio, (0, int(0.2 * rate) - audio.size))
    win = int(0.025 * rate)
    hop = int(0.010 * rate)
    window = np.hanning(win).astype(np.float32)
    frames = []
    for start in range(0, audio.size - win, hop):
        mag = np.abs(np.fft.rfft(audio[start:start + win] * window))
        frames.append(np.log1p(mag))
    if not frames:
        return np.zeros(BANDS, dtype=np.float32)
    mean = np.mean(frames, axis=0)
    edges = np.linspace(0, mean.size, BANDS + 1, dtype=int)
    bands = np.array([
        float(mean[edges[i]:edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0
        for i in range(BANDS)
    ], dtype=np.float32)
    bands -= float(bands.mean())
    norm = float(np.linalg.norm(bands))
    if norm > 0:
        bands /= norm
    return bands


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or float(np.linalg.norm(left)) == 0 or float(np.linalg.norm(right)) == 0:
        return 0.0
    return float(np.dot(left, right))


class SpeakerBook:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else data_dir() / "speakers"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{_slug(name)}.json"

    def list(self) -> list[dict]:
        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(raw, dict) and raw.get("name"):
                found.append({"name": raw["name"], "clips": int(raw.get("count") or len(raw.get("vectors") or []))})
        return found

    def _load(self, name: str) -> dict | None:
        path = self._path(name)
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def enroll(self, name: str, audio: np.ndarray, rate: int = TARGET_RATE) -> dict:
        name = " ".join(str(name).split())[:60]
        if not name or name == "Me":
            raise ValueError("Choose a name other than Me.")
        vector = embed_speaker(audio, rate)
        if float(np.linalg.norm(vector)) == 0:
            raise ValueError("That clip is silent.")
        current = self._load(name) or {"name": name, "vectors": [], "count": 0}
        vectors = current.get("vectors") or []
        vectors.append([round(float(v), 5) for v in vector])
        current["vectors"] = vectors[-24:]
        current["count"] = len(current["vectors"])
        current["name"] = name
        path = self._path(name)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current))
        tmp.replace(path)
        return {"name": name, "clips": current["count"]}

    def match(self, audio: np.ndarray, threshold: float = 0.72, rate: int = TARGET_RATE) -> tuple[str | None, float]:
        probe = embed_speaker(audio, rate)
        if float(np.linalg.norm(probe)) == 0:
            return None, 0.0
        best_name, best_score = None, 0.0
        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            scores = [cosine(probe, np.asarray(vec, dtype=np.float32)) for vec in raw.get("vectors") or []]
            if not scores:
                continue
            score = float(np.mean(sorted(scores, reverse=True)[:3]))
            if score > best_score:
                best_name, best_score = raw.get("name"), score
        if best_name and best_score >= threshold:
            return str(best_name), round(best_score, 3)
        return None, round(best_score, 3)


def slice_session_audio(library: Library, session_id: str, start: float, end: float) -> np.ndarray:
    root = library.session_path(session_id)
    system = root / "audio-system.wav"
    # The Others track is the voice we want. The mix also contains Me.
    path = system if system.exists() else library.audio_path(library.get(session_id))
    if path is None:
        raise FileNotFoundError("This session has no saved audio to learn a voice from.")
    audio = read_wav(path)
    n0 = max(0, int(start * TARGET_RATE))
    n1 = min(audio.size, int(max(end, start + 0.3) * TARGET_RATE))
    if n1 <= n0:
        raise ValueError("That line has no audio.")
    return audio[n0:n1]


def correct_segment(library: Library, book: SpeakerBook, session_id: str, start: float, name: str) -> dict:
    segments = library.segments(session_id)
    if not segments:
        raise KeyError(session_id)
    segment = min(segments, key=lambda item: abs(item.start - start))
    audio = slice_session_audio(library, session_id, segment.start, segment.end)
    profile = book.enroll(name, audio)
    segment.speaker = name
    library.save_segments(session_id, segments)
    meta = library.get(session_id)
    speakers = dict(meta.speakers or {})
    speakers[name] = name
    library.update(session_id, speakers=speakers)
    return {"segment": segment.to_dict(), "profile": profile}
