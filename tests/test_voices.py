import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from listn.audio import TARGET_RATE, to_int16
from listn.config import SPEAKER_THRESHOLD, Settings
from listn.library import Library, Segment
from listn.speakers import (
    MAX_CLIPS,
    NEURAL_MARGIN,
    SPECTRUM,
    SPECTRUM_KEY,
    SpeakerBook,
    name_speaker,
    threshold_for,
)


def _voice(fundamental, seconds=2.0, amp=0.2):
    t = np.arange(int(seconds * TARGET_RATE)) / TARGET_RATE
    return (amp * np.sin(2 * np.pi * fundamental * t)).astype(np.float32)


def _write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(to_int16(audio).tobytes())


class FakeNeural:
    """Deterministic stand-in for WeSpeaker: the voice print is set by the dominant pitch."""

    key = "fake-neural@test"
    label = "fake"
    neural = True
    default_threshold = SPEAKER_THRESHOLD
    margin = NEURAL_MARGIN
    min_seconds = 1.0

    def __init__(self, prints=None):
        self.prints = prints or {}
        self.calls = 0

    def embed(self, audio, rate=TARGET_RATE):
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float32)))
        pitch = int(round(np.argmax(spectrum) * rate / audio.size / 10.0) * 10)
        vec = np.asarray(self.prints.get(pitch, np.zeros(4)), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


PRINTS = {
    140: [1.0, 0.0, 0.0, 0.0],   # Priya
    150: [0.9, 0.42, 0.0, 0.0],  # Priya on a worse line: cosine ~0.91 with 140
    200: [0.0, 1.0, 0.0, 0.0],   # Sam
    300: [0.0, 0.0, 1.0, 0.0],   # someone new
    400: [0.6, 0.6, 0.53, 0.0],  # halfway between Priya and Sam
}


@pytest.fixture
def neural():
    return FakeNeural(PRINTS)


def test_default_threshold_is_on_the_wespeaker_scale():
    assert SPEAKER_THRESHOLD == 0.55
    assert Settings().speaker_threshold == 0.55
    assert threshold_for(FakeNeural(), Settings(speaker_threshold=0.62)) == 0.62
    assert threshold_for(SPECTRUM, Settings(speaker_threshold=0.62)) == 0.72


def test_known_voice_is_named_and_a_new_one_is_not(data_dir, neural):
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    book.enroll("Priya", _voice(140))
    book.enroll("Sam", _voice(200))
    assert book.match(_voice(150)) == ("Priya", pytest.approx(0.906, abs=0.01))
    name, score = book.match(_voice(300))
    assert name is None and score < SPEAKER_THRESHOLD


def test_threshold_is_respected(data_dir, neural):
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    book.enroll("Priya", _voice(140))
    name, score = book.match(_voice(150), threshold=0.95)
    assert name is None and 0.85 < score < 0.95


def test_an_ambiguous_voice_needs_a_clear_lead(data_dir, neural):
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    book.enroll("Priya", _voice(140))
    book.enroll("Sam", _voice(200))
    name, score = book.match(_voice(400), threshold=0.3)
    assert score >= 0.55
    assert name is None
    assert book.match(_voice(400), threshold=0.3, margin=0.0)[0] in ("Priya", "Sam")


def test_identify_gives_each_known_person_to_one_speaker(data_dir, neural):
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    book.enroll("Priya", _voice(140))
    book.enroll("Sam", _voice(200))
    probes = {
        "SPEAKER_00": neural.embed(_voice(150)),
        "SPEAKER_01": neural.embed(_voice(140)),
        "SPEAKER_02": neural.embed(_voice(200)),
        "SPEAKER_03": neural.embed(_voice(300)),
    }
    out = book.identify(probes)
    assert out["SPEAKER_01"]["name"] == "Priya"
    assert out["SPEAKER_00"]["name"] is None
    assert out["SPEAKER_02"]["name"] == "Sam"
    assert out["SPEAKER_03"]["name"] is None


def test_vectors_from_different_methods_are_never_compared(data_dir, neural):
    root = data_dir / "speakers"
    SpeakerBook(root).enroll("Priya", _voice(140))
    book = SpeakerBook(root, embedder=neural)
    assert book.match(_voice(140)) == (None, 0.0)
    [row] = book.describe()
    assert row["needs_reenroll"] and row["methods"] == [SPECTRUM_KEY] and row["saved_clips"] == 1


def test_enrolment_keeps_both_methods_and_a_capped_set_of_clips(data_dir, neural):
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    for _ in range(MAX_CLIPS + 2):
        book.enroll("Priya", _voice(140))
    raw = json.loads((data_dir / "speakers" / "priya.json").read_text())
    assert raw["version"] == 2
    assert set(raw["vectors"]) == {neural.key, SPECTRUM_KEY}
    assert len(raw["clips"]) == MAX_CLIPS
    assert len(list((data_dir / "speakers" / "clips" / "priya").glob("*.wav"))) == MAX_CLIPS
    assert book.delete("Priya")
    assert not (data_dir / "speakers" / "clips" / "priya").exists()
    assert book.list() == []


def test_legacy_profiles_are_relearned_from_saved_clips(data_dir, neural):
    root = data_dir / "speakers"
    SpeakerBook(root).enroll("Priya", _voice(140))
    book = SpeakerBook(root, embedder=neural)
    result = book.reenroll()
    assert result["reenrolled"] == ["Priya"] and result["needs_reenroll"] == []
    assert book.match(_voice(150))[0] == "Priya"


def test_0_4_profiles_without_clips_are_relearned_from_the_library_or_flagged(data_dir, neural):
    root = data_dir / "speakers"
    root.mkdir(parents=True)
    for name in ("Sam", "Old"):
        (root / f"{name.lower()}.json").write_text(json.dumps({"name": name, "vectors": [[0.1] * 64], "count": 1}))
    library = Library(data_dir / "sessions")
    meta = library.create("Standup", "live", speakers={"Me": "Me", "Others": "Others"})
    session = library.session_path(meta.id)
    audio = np.concatenate([_voice(200, 3.0), _voice(500, 2.0)])
    _write(session / "audio-system.wav", audio)
    _write(session / "audio-mic.wav", np.zeros_like(audio))
    library.save_segments(meta.id, [Segment(0.0, 3.0, "Sam's line, corrected in 0.4", "Sam"),
                                    Segment(3.0, 5.0, "Me talking", "Me")])
    book = SpeakerBook(root, embedder=neural)
    assert {p["name"]: p["clips"] for p in book.describe()} == {"Old": 0, "Sam": 0}
    result = book.reenroll(library)
    assert result["reenrolled"] == ["Sam"] and result["needs_reenroll"] == ["Old"]
    rows = {p["name"]: p for p in book.describe()}
    assert rows["Sam"]["clips"] == 1 and not rows["Sam"]["needs_reenroll"]
    assert rows["Old"]["needs_reenroll"]
    assert book.match(_voice(200))[0] == "Sam"


def _two_track_session(library, lines, system_audio):
    meta = library.create("Call", "live", speakers={"Me": "Me", "Others": "Others", "SPEAKER_00": "Speaker 1"})
    root = library.session_path(meta.id)
    _write(root / "audio-system.wav", system_audio)
    _write(root / "audio-mic.wav", np.zeros_like(system_audio))
    library.update(meta.id, audio_file="audio.wav", diarized=True)
    library.save_segments(meta.id, lines)
    return meta.id


def test_naming_a_diarized_speaker_renames_and_enrols(data_dir, neural):
    library = Library(data_dir / "sessions")
    audio = np.concatenate([_voice(140, 4.0), _voice(300, 2.0)])
    sid = _two_track_session(library, [
        Segment(0.0, 2.0, "First point", "SPEAKER_00"),
        Segment(2.0, 4.0, "Second point", "SPEAKER_00"),
        Segment(4.0, 4.4, "uh", "SPEAKER_00"),
        Segment(4.5, 6.0, "Mine", "Me"),
    ], audio)
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    result = name_speaker(library, book, sid, "SPEAKER_00", "  Priya ")
    assert result["speakers"]["SPEAKER_00"] == "Priya"
    assert result["profile"] == {"name": "Priya", "clips": 2}
    assert library.get(sid).speakers["SPEAKER_00"] == "Priya"
    assert [s.speaker for s in library.segments(sid)][:2] == ["SPEAKER_00", "SPEAKER_00"]
    assert book.match(_voice(150))[0] == "Priya"


def test_naming_refuses_me_and_does_not_learn_from_the_whole_others_track(data_dir, neural):
    library = Library(data_dir / "sessions")
    sid = _two_track_session(library, [Segment(0.0, 2.0, "x", "Others"), Segment(2.0, 3.0, "y", "Me")], _voice(140, 3.0))
    book = SpeakerBook(data_dir / "speakers", embedder=neural)
    with pytest.raises(ValueError):
        name_speaker(library, book, sid, "Me", "Not me")
    result = name_speaker(library, book, sid, "Others", "Team")
    assert result["profile"] is None and "several people" in result["note"]
    assert book.list() == []


def test_live_chunks_use_the_voice_model_without_downloading(data_dir, neural, monkeypatch):
    from listn import speakers
    from listn.engine import LiveSession
    from listn.engine import EventBus

    SpeakerBook(embedder=neural).enroll("Priya", _voice(140))
    asked = []

    def fake_embedder(token=None, allow_download=False):
        asked.append(allow_download)
        return neural

    monkeypatch.setattr(speakers, "voice_embedder", fake_embedder)
    session = SimpleNamespace(_book=None, settings=Settings(speaker_threshold=0.55))
    assert LiveSession._match_known(session, _voice(150)) == "Priya"
    assert LiveSession._match_known(session, _voice(300)) is None
    assert LiveSession._match_known(session, _voice(140, seconds=0.5)) is None
    assert asked == [False]
    assert EventBus
