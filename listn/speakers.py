"""Local speaker profiles: who is talking on the Others track.

Voices are compared as speaker embeddings from the WeSpeaker ResNet34 model
that pyannote's community-1 pipeline uses (256 numbers per voice, cosine
similarity). The weights come from the same local Hugging Face cache as the
diarization model. Without pyannote, a 64-band log-spectrum "embedding" is
the fallback; it needs no model but tells voices apart poorly.

Profiles live in ``speakers/<name>.json`` under the data directory, with the
clips they were learned from in ``speakers/clips/<name>/``. Vectors are kept
per embedding method, so a profile learned with one model is never compared
with vectors from another; when the model changes, profiles are re-learned
from their clips (or flagged for re-enrolment when there are none).

A known voice is named only when its score clears ``speaker_threshold`` and
beats the next-best profile by ``margin``: on a call everyone shares the
same codec and noise, which lifts every score, so the gap matters as much
as the level.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import threading
import time
import wave
from pathlib import Path

import numpy as np

from listn.audio import TARGET_RATE, read_wav, rms, to_int16
from listn.config import SPEAKER_THRESHOLD, data_dir
from listn.library import Library

log = logging.getLogger("listn")

BANDS = 64
SPECTRUM_KEY = "spectrum-64"
SPECTRUM_THRESHOLD = 0.72
# Calibrated on LibriSpeech dev-clean (40 speakers), clean and call-degraded;
# see the README section "Known voices".
NEURAL_THRESHOLD = SPEAKER_THRESHOLD
NEURAL_MARGIN = 0.08
NEURAL_MODEL = "wespeaker-resnet34"
# community-1 ships the ResNet34 weights in its own repo; the standalone
# WeSpeaker repo is not gated and is only used if community-1 can't load.
EMBEDDING_SOURCES = (
    ("pyannote/speaker-diarization-community-1", "embedding", True),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", None, False),
)
MAX_VECTORS = 24
MAX_CLIPS = 8
CLIP_SECONDS = 15.0
MIN_CLIP_SECONDS = 1.0
ENROL_CLIPS = 3
SILENCE_RMS = 1e-4


def _slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return text or "speaker"


def _unit(vector) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return np.zeros(vec.size, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


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
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    if float(np.linalg.norm(left)) == 0 or float(np.linalg.norm(right)) == 0:
        return 0.0
    return float(np.dot(left, right))


class SpectrumEmbedder:
    key = SPECTRUM_KEY
    label = "spectrum fallback (no model)"
    neural = False
    default_threshold = SPECTRUM_THRESHOLD
    margin = 0.0
    min_seconds = 0.2

    def embed(self, audio: np.ndarray, rate: int = TARGET_RATE) -> np.ndarray:
        return embed_speaker(audio, rate)


SPECTRUM = SpectrumEmbedder()


class NeuralEmbedder:
    """WeSpeaker ResNet34 through ``pyannote.audio.Inference(window="whole")``."""

    neural = True
    default_threshold = NEURAL_THRESHOLD
    margin = NEURAL_MARGIN
    min_seconds = MIN_CLIP_SECONDS

    def __init__(self, inference, source: str, device: str = "cpu"):
        self.inference = inference
        self.source = source
        self.device = device
        self.key = f"{NEURAL_MODEL}@{source}"
        self.label = f"WeSpeaker ResNet34 ({source.split('/')[1]}, {device})"
        self._lock = threading.Lock()

    def _run(self, audio: np.ndarray, rate: int):
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1))
        return self.inference({"waveform": waveform, "sample_rate": rate})

    def embed(self, audio: np.ndarray, rate: int = TARGET_RATE) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < int(self.min_seconds * rate):
            audio = np.pad(audio, (0, int(self.min_seconds * rate) - audio.size))
        with self._lock:
            try:
                out = self._run(audio, rate)
            except Exception:
                if self.device == "cpu":
                    raise
                import torch

                self.inference.to(torch.device("cpu"))
                self.device = "cpu"
                out = self._run(audio, rate)
        return _unit(np.asarray(out, dtype=np.float32))


_neural: NeuralEmbedder | None = None
_neural_lock = threading.Lock()
_offline_miss_at = 0.0
OFFLINE_RETRY_SECONDS = 300.0


def _load_inference(repo: str, subfolder: str | None, token: str | None):
    from pyannote.audio import Inference, Model

    kwargs = {"subfolder": subfolder} if subfolder else {}
    if token:
        kwargs["token"] = token
    model = Model.from_pretrained(repo, **kwargs)
    if model is None:
        raise RuntimeError(f"{repo} did not load")
    return Inference(model, window="whole")


def _place(inference) -> tuple[object, str]:
    from listn.diarize import pick_device

    try:
        device = pick_device()
    except Exception:
        return inference, "cpu"
    if device.type == "cpu":
        return inference, "cpu"
    try:
        import torch

        inference.to(device)
        probe = np.random.default_rng(0).standard_normal(TARGET_RATE).astype(np.float32) * 0.05
        inference({"waveform": torch.from_numpy(probe.reshape(1, -1)), "sample_rate": TARGET_RATE})
        return inference, device.type
    except Exception as exc:
        log.info("Voice model stays on the CPU: %s", exc)
        import torch

        inference.to(torch.device("cpu"))
        return inference, "cpu"


def load_neural_embedder(token: str | None = None, allow_download: bool = False) -> NeuralEmbedder:
    """The shared WeSpeaker embedder. Cache first; the network only when ``allow_download``,
    and then only for the ungated WeSpeaker repo. ``token`` is accepted for symmetry and
    only ever redacted from errors."""
    global _neural, _offline_miss_at
    with _neural_lock:
        if _neural is not None:
            return _neural
        from listn import diarize
        from listn.hub import load_local_or_download, redact

        if not diarize.installed():
            raise RuntimeError(diarize.INSTALL_HINT)
        if not allow_download and time.monotonic() - _offline_miss_at < OFFLINE_RETRY_SECONDS:
            raise RuntimeError("The voice model isn't cached yet.")
        diarize.telemetry_off()
        errors = []
        for repo, subfolder, gated in EMBEDDING_SOURCES:
            source = f"{repo}/{subfolder}" if subfolder else repo

            def local(repo=repo, subfolder=subfolder):
                return _load_inference(repo, subfolder, None)

            def remote(repo=repo, subfolder=subfolder, gated=gated):
                # The gated weights arrive with the community-1 pipeline download, so
                # they are never fetched (or the token sent) from here.
                if gated:
                    raise RuntimeError("not cached; it comes with the community-1 download")
                if not allow_download:
                    raise RuntimeError("not cached")
                return _load_inference(repo, subfolder, None)

            try:
                inference = load_local_or_download(local, remote)
            except Exception as exc:
                errors.append(f"{source}: {redact(exc, token).splitlines()[0] if str(exc) else type(exc).__name__}")
                continue
            inference, device = _place(inference)
            _neural = NeuralEmbedder(inference, source, device)
            log.info("Voice model %s on %s", source, device)
            return _neural
        if not allow_download:
            _offline_miss_at = time.monotonic()
        raise RuntimeError("No voice model could be loaded: " + "; ".join(errors))


def voice_embedder(token: str | None = None, allow_download: bool = False):
    """WeSpeaker when pyannote is available, otherwise the spectrum fallback."""
    try:
        return load_neural_embedder(token, allow_download)
    except Exception as exc:
        log.debug("Voice matching uses the spectrum fallback: %s", exc)
        return SPECTRUM


def loaded_embedder():
    return _neural


def reset_embedder() -> None:
    global _neural, _offline_miss_at
    with _neural_lock:
        _neural = None
        _offline_miss_at = 0.0


def threshold_for(embedder, settings) -> float:
    """``speaker_threshold`` is on the WeSpeaker cosine scale; the fallback keeps its own."""
    if getattr(embedder, "neural", False):
        return float(settings.speaker_threshold)
    return float(getattr(settings, "spectrum_threshold", SPECTRUM_THRESHOLD))


def write_clip(path: Path, audio: np.ndarray, rate: int = TARGET_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(to_int16(audio).tobytes())
    tmp.replace(path)


class SpeakerBook:
    def __init__(self, root: Path | None = None, embedder=None):
        self.root = Path(root) if root else data_dir() / "speakers"
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or SPECTRUM

    def _path(self, name: str) -> Path:
        return self.root / f"{_slug(name)}.json"

    def _clip_dir(self, name: str) -> Path:
        return self.root / "clips" / _slug(name)

    @staticmethod
    def _normalise(raw: dict) -> dict:
        vectors = raw.get("vectors")
        if isinstance(vectors, list):
            vectors = {SPECTRUM_KEY: vectors} if vectors else {}
        elif not isinstance(vectors, dict):
            vectors = {}
        clips = [c for c in raw.get("clips") or [] if isinstance(c, str)]
        return {
            "version": 2,
            "name": str(raw.get("name") or ""),
            "vectors": {str(k): v for k, v in vectors.items() if isinstance(v, list)},
            "clips": clips,
            "needs_reenroll": bool(raw.get("needs_reenroll", False)),
        }

    def _read(self, path: Path) -> dict | None:
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or not raw.get("name"):
            return None
        return self._normalise(raw)

    def _load(self, name: str) -> dict | None:
        return self._read(self._path(name))

    def _save(self, profile: dict) -> None:
        profile = dict(profile)
        active = profile["vectors"].get(self.embedder.key) or []
        profile["count"] = len(active)
        path = self._path(profile["name"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(profile))
        tmp.replace(path)

    def profiles(self) -> list[dict]:
        found = []
        for path in sorted(self.root.glob("*.json")):
            profile = self._read(path)
            if profile:
                found.append(profile)
        return found

    def list(self) -> list[dict]:
        return [{"name": p["name"], "clips": len(p["vectors"].get(self.embedder.key) or [])} for p in self.profiles()]

    def describe(self) -> list[dict]:
        out = []
        for p in self.profiles():
            active = len(p["vectors"].get(self.embedder.key) or [])
            out.append({
                "name": p["name"],
                "clips": active,
                "saved_clips": len(p["clips"]),
                "methods": sorted(p["vectors"]),
                "needs_reenroll": p["needs_reenroll"] or (self.embedder.neural and active == 0),
            })
        return out

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.exists():
            return False
        path.unlink()
        shutil.rmtree(self._clip_dir(name), ignore_errors=True)
        return True

    def _vectors_for(self, audio: np.ndarray, rate: int) -> dict[str, list[float]]:
        out = {}
        for embedder in {self.embedder.key: self.embedder, SPECTRUM.key: SPECTRUM}.values():
            vec = embedder.embed(audio, rate)
            if float(np.linalg.norm(vec)) > 0:
                out[embedder.key] = [round(float(v), 5) for v in vec]
        return out

    def enroll(self, name: str, audio: np.ndarray, rate: int = TARGET_RATE, save_clip: bool = True) -> dict:
        name = " ".join(str(name).split())[:60]
        if not name or name in ("Me", "Others"):
            raise ValueError("Choose a name other than Me.")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or rms(audio) < SILENCE_RMS:
            raise ValueError("That clip is silent.")
        vectors = self._vectors_for(audio, rate)
        if self.embedder.key not in vectors:
            raise ValueError("That clip is silent.")
        current = self._load(name) or self._normalise({"name": name})
        current["name"] = name
        for key, vec in vectors.items():
            current["vectors"][key] = ((current["vectors"].get(key) or []) + [vec])[-MAX_VECTORS:]
        if save_clip and rate == TARGET_RATE:
            clip = audio[: int(CLIP_SECONDS * rate)]
            rel = f"clips/{_slug(name)}/{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.wav"
            write_clip(self.root / rel, clip, rate)
            current["clips"].append(rel)
            for old in current["clips"][:-MAX_CLIPS]:
                (self.root / old).unlink(missing_ok=True)
            current["clips"] = current["clips"][-MAX_CLIPS:]
        current["needs_reenroll"] = False
        self._save(current)
        return {"name": name, "clips": len(current["vectors"][self.embedder.key])}

    def _score(self, vectors: list, probe: np.ndarray) -> float:
        if not vectors:
            return 0.0
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != probe.size:
            return 0.0
        if self.embedder.neural:
            return cosine(_unit(matrix.mean(axis=0)), probe)
        scores = sorted((cosine(probe, _unit(row)) for row in matrix), reverse=True)
        return float(np.mean(scores[:3]))

    def scores(self, probe: np.ndarray) -> list[tuple[float, str]]:
        probe = _unit(probe)
        if float(np.linalg.norm(probe)) == 0:
            return []
        out = []
        for profile in self.profiles():
            vectors = profile["vectors"].get(self.embedder.key)
            if vectors:
                out.append((self._score(vectors, probe), profile["name"]))
        return sorted(out, reverse=True)

    def decide(self, ranked: list[tuple[float, str]], threshold: float, margin: float | None = None,
               exclude: set[str] | frozenset = frozenset()) -> tuple[str | None, float]:
        margin = self.embedder.margin if margin is None else margin
        ranked = [r for r in ranked if r[1] not in exclude]
        if not ranked:
            return None, 0.0
        best, name = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        if best >= threshold and best - runner_up >= margin:
            return name, round(best, 3)
        return None, round(best, 3)

    def match_vector(self, probe: np.ndarray, threshold: float | None = None,
                     margin: float | None = None) -> tuple[str | None, float]:
        threshold = self.embedder.default_threshold if threshold is None else threshold
        return self.decide(self.scores(probe), threshold, margin)

    def match(self, audio: np.ndarray, threshold: float | None = None, rate: int = TARGET_RATE,
              margin: float | None = None) -> tuple[str | None, float]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or rms(audio) < SILENCE_RMS:
            return None, 0.0
        return self.match_vector(self.embedder.embed(audio, rate), threshold, margin)

    def identify(self, probes: dict[str, np.ndarray], threshold: float | None = None,
                 margin: float | None = None) -> dict[str, dict]:
        """Name diarized speakers. Each known person is given to at most one label."""
        threshold = self.embedder.default_threshold if threshold is None else threshold
        margin = self.embedder.margin if margin is None else margin
        ranked = {label: self.scores(vec) for label, vec in probes.items()}
        out = {label: {"name": None, "score": round(r[0][0], 3) if r else None} for label, r in ranked.items()}
        pairs = sorted(((score, label, name) for label, r in ranked.items() for score, name in r), reverse=True)
        taken: set[str] = set()
        done: set[str] = set()
        for _score, label, name in pairs:
            if label in done or name in taken:
                continue
            done.add(label)
            choice, best = self.decide(ranked[label], threshold, margin, exclude=taken)
            if choice is not None:
                out[label] = {"name": choice, "score": best}
                taken.add(choice)
        return out

    def reenroll(self, library: Library | None = None) -> dict:
        """Give every profile vectors for the current embedder, from its clips or the library."""
        done, flagged = [], []
        for profile in self.profiles():
            if profile["vectors"].get(self.embedder.key):
                continue
            vectors = []
            for rel in profile["clips"]:
                try:
                    audio = read_wav(self.root / rel)
                except Exception:
                    continue
                if audio.size >= int(MIN_CLIP_SECONDS * TARGET_RATE) and rms(audio) >= SILENCE_RMS:
                    vec = self.embedder.embed(audio)
                    if float(np.linalg.norm(vec)) > 0:
                        vectors.append([round(float(v), 5) for v in vec])
            if not vectors and library is not None:
                for audio in library_clips(library, profile["name"]):
                    rel = f"clips/{_slug(profile['name'])}/library-{secrets.token_hex(3)}.wav"
                    write_clip(self.root / rel, audio)
                    profile["clips"].append(rel)
                    vec = self.embedder.embed(audio)
                    if float(np.linalg.norm(vec)) > 0:
                        vectors.append([round(float(v), 5) for v in vec])
            if vectors:
                profile["vectors"][self.embedder.key] = vectors[-MAX_VECTORS:]
                profile["needs_reenroll"] = False
                done.append(profile["name"])
            else:
                profile["needs_reenroll"] = True
                flagged.append(profile["name"])
            self._save(profile)
        return {"method": self.embedder.key, "reenrolled": done, "needs_reenroll": flagged}


def others_track(library: Library, session_id: str):
    """Path of the audio that holds only other people's voices, if there is one."""
    root = library.session_path(session_id)
    system = root / "audio-system.wav"
    if system.exists():
        return system
    if (root / "audio-mic.wav").exists():
        return None
    return library.audio_path(library.get(session_id))


def library_clips(library: Library, name: str, limit: int = ENROL_CLIPS) -> list[np.ndarray]:
    """The longest lines labelled ``name`` in saved sessions, newest sessions first."""
    clips: list[np.ndarray] = []
    for meta in library.list():
        if len(clips) >= limit:
            break
        speakers = meta.speakers or {}
        labels = {name} | {label for label, shown in speakers.items() if shown == name}
        lines = [s for s in library.segments(meta.id)
                 if s.speaker in labels and s.speaker != "Me" and s.end - s.start >= MIN_CLIP_SECONDS]
        if not lines:
            continue
        try:
            path = others_track(library, meta.id)
            audio = read_wav(path) if path is not None and path.suffix.lower() == ".wav" else None
        except Exception:
            audio = None
        if audio is None:
            continue
        for seg in sorted(lines, key=lambda s: s.end - s.start, reverse=True):
            a, b = int(seg.start * TARGET_RATE), min(audio.size, int(min(seg.end, seg.start + CLIP_SECONDS) * TARGET_RATE))
            if b - a >= int(MIN_CLIP_SECONDS * TARGET_RATE) and rms(audio[a:b]) >= SILENCE_RMS:
                clips.append(audio[a:b])
            if len(clips) >= limit:
                break
    return clips


def slice_session_audio(library: Library, session_id: str, start: float, end: float) -> np.ndarray:
    """One line of the Others track, for learning a voice from.

    Prefers the speaker-isolated system track. The microphone track is the
    user's own voice, so a session that only has that is refused rather than
    enrolled from. A session recorded as a single track (an import) has no
    isolated track at all, so its own recording is used — there is nothing
    else to slice.
    """
    path = others_track(library, session_id)
    if path is None:
        if (library.session_path(session_id) / "audio-mic.wav").exists():
            raise ValueError(
                "This session only has your microphone track, so there is no audio of other people "
                "alone to learn a voice from. Label a session that has the Others track instead."
            )
        raise FileNotFoundError("This session has no saved audio to learn a voice from.")
    if not path.exists():
        raise FileNotFoundError("This session has no saved audio to learn a voice from.")
    audio = read_wav(path) if path.suffix.lower() == ".wav" else None
    if audio is None:
        from listn.audio import decode_file

        audio = decode_file(path)
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


def name_speaker(library: Library, book: SpeakerBook, session_id: str, label: str, name: str) -> dict:
    """Rename a diarized speaker in one session and learn their voice from it."""
    name = " ".join(str(name).split())[:60]
    meta = library.get(session_id)
    segments = library.segments(session_id)
    if label == "Me":
        raise ValueError("Me is the microphone track and is already you.")
    if not name or name in ("Me", "Others"):
        raise ValueError("Type the person's name.")
    lines = [s for s in segments if s.speaker == label]
    if not lines:
        raise ValueError("No lines have that speaker.")
    speakers = dict(meta.speakers or {})
    speakers[label] = name
    library.update(session_id, speakers=speakers)
    profile = None
    note = None
    if label == "Others":
        note = "Renamed. Others may be several people, so no voice was learned; label speakers first."
    else:
        path = others_track(library, session_id)
        if path is None:
            note = "Renamed. This session has no audio of other people alone, so no voice was learned."
        else:
            audio = read_wav(path) if path.suffix.lower() == ".wav" else None
            if audio is None:
                from listn.audio import decode_file

                audio = decode_file(path)
            usable = [s for s in sorted(lines, key=lambda s: s.end - s.start, reverse=True)
                      if s.end - s.start >= MIN_CLIP_SECONDS][:ENROL_CLIPS]
            for seg in usable:
                a = int(seg.start * TARGET_RATE)
                b = min(audio.size, int(min(seg.end, seg.start + CLIP_SECONDS) * TARGET_RATE))
                try:
                    profile = book.enroll(name, audio[a:b])
                except ValueError:
                    continue
            if profile is None:
                note = "Renamed. Their lines are too short or quiet to learn the voice from."
    return {"speakers": speakers, "profile": profile, "note": note}
