import wave

import numpy as np
from fastapi.testclient import TestClient

from listn.audio import TARGET_RATE, to_int16
from listn.engine import Engine
from listn.library import Library, Segment
from listn.server import create_app
from listn.speakers import SPECTRUM_KEY, SpeakerBook, correct_segment, embed_speaker

from .conftest import MAC_DEVICES, FakeSD


def _voice(fundamental, seconds=1.2, rate=TARGET_RATE, amp=0.2):
    t = np.arange(int(seconds * rate)) / rate
    audio = np.zeros_like(t)
    for k, scale in ((1, 1.0), (2, 0.55), (3, 0.3), (4, 0.15)):
        audio += scale * np.sin(2 * np.pi * fundamental * k * t)
    return (amp * audio).astype(np.float32)


def _write(path, audio, rate=TARGET_RATE):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(to_int16(audio).tobytes())


def test_same_voice_matches_and_a_different_one_does_not(data_dir):
    book = SpeakerBook(data_dir / "speakers")
    priya = _voice(140)
    book.enroll("Priya", priya)
    again = _voice(140, amp=0.12) + (0.01 * np.random.default_rng(0).standard_normal(priya.size)).astype(np.float32)
    name, score = book.match(again, 0.72)
    assert name == "Priya"
    assert score >= 0.72
    other, other_score = book.match(_voice(310), 0.72)
    assert other is None
    assert other_score < 0.72
    assert book.list() == [{"name": "Priya", "clips": 1}]


def test_threshold_can_reject_a_weak_match(data_dir):
    book = SpeakerBook(data_dir / "speakers")
    book.enroll("Priya", _voice(140))
    _name, weak = book.match(_voice(310), 0.0)
    assert weak < 0.72
    name, score = book.match(_voice(310), weak + 0.01)
    assert name is None
    assert score == weak


def test_me_and_silence_are_refused(data_dir):
    book = SpeakerBook(data_dir / "speakers")
    try:
        book.enroll("Me", _voice(140))
        raise AssertionError("Me should not be enrollable")
    except ValueError:
        pass
    try:
        book.enroll("Quiet", np.zeros(TARGET_RATE, dtype=np.float32))
        raise AssertionError("silence should not enroll")
    except ValueError:
        pass


def test_correction_renames_the_line_and_grows_the_profile(data_dir):
    library = Library(data_dir / "sessions")
    meta = library.create("Standup", "live", speakers={"Others": "Others"})
    audio = _voice(180, seconds=2.0)
    path = library.session_path(meta.id) / "audio-system.wav"
    _write(path, audio)
    library.save_segments(meta.id, [Segment(0.2, 1.6, "Let's ship Friday", "Others")])
    book = SpeakerBook(data_dir / "speakers")
    result = correct_segment(library, book, meta.id, 0.2, "Sam")
    assert result["segment"]["speaker"] == "Sam"
    assert result["profile"]["clips"] == 1
    assert library.segments(meta.id)[0].speaker == "Sam"
    assert library.get(meta.id).speakers["Sam"] == "Sam"
    name, _score = book.match(_voice(180), 0.72)
    assert name == "Sam"


def test_voice_export_is_a_download_of_the_users_own_profiles(data_dir, fake_factory):
    import json

    sd = FakeSD(MAC_DEVICES, speed=40.0)
    app = create_app(engine=Engine(sd=sd, transcriber_factory=fake_factory), sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        clip = data_dir / "clip.wav"
        _write(clip, _voice(160))
        with clip.open("rb") as handle:
            saved = client.post("/api/speakers/enroll", data={"name": "Ada"}, files={"file": ("clip.wav", handle, "audio/wav")})
        assert saved.status_code == 200, saved.text

        export = client.get("/api/speakers/export")
        assert export.status_code == 200, export.text
        assert export.headers["content-disposition"] == 'attachment; filename="listn-voices.json"'
        assert export.headers["content-type"].startswith("application/json")
        payload = json.loads(export.content)
        [profile] = payload["profiles"]
        assert profile["name"] == "Ada"
        assert len(profile["vectors"][SPECTRUM_KEY]) == 1
        assert len(profile["vectors"][SPECTRUM_KEY][0]) == 64
        # The export is the numbers only: no local paths and no audio bytes.
        assert str(data_dir) not in export.text
        assert "RIFF" not in export.text and "base64" not in export.text

        assert client.delete("/api/speakers/Ada").json() == {"ok": True}
        assert json.loads(client.get("/api/speakers/export").content)["profiles"] == []


def test_embedding_is_stable_and_local(data_dir):
    left = embed_speaker(_voice(200))
    right = embed_speaker(_voice(200))
    assert left.shape == (64,)
    assert abs(float(np.dot(left, right)) - 1.0) < 1e-4
    assert not (data_dir / "speakers").exists() or list((data_dir / "speakers").glob("*")) == []


def test_speaker_api_enroll_and_correct(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, speed=40.0)
    app = create_app(engine=Engine(sd=sd, transcriber_factory=fake_factory), sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        clip = data_dir / "clip.wav"
        _write(clip, _voice(160))
        with clip.open("rb") as handle:
            enrolled = client.post("/api/speakers/enroll", data={"name": "Ada"}, files={"file": ("clip.wav", handle, "audio/wav")})
        assert enrolled.status_code == 200, enrolled.text
        assert enrolled.json()["clips"] == 1
        listed = client.get("/api/speakers")
        assert listed.json()["speakers"][0]["name"] == "Ada"

        library = Library()
        meta = library.create("Review", "file", speakers={"Others": "Others"})
        _write(library.session_path(meta.id) / "audio.wav", _voice(220, seconds=1.5))
        library.update(meta.id, audio_file="audio.wav")
        library.save_segments(meta.id, [Segment(0.0, 1.2, "The budget is fine", "Others")])
        corrected = client.post("/api/speakers/correct", json={"session_id": meta.id, "start": 0.0, "name": "Noah"})
        assert corrected.status_code == 200, corrected.text
        assert corrected.json()["segment"]["speaker"] == "Noah"
        refused = client.post("/api/speakers/enroll", data={"name": "Me"}, files={"file": ("clip.wav", clip.read_bytes(), "audio/wav")})
        assert refused.status_code == 400
