import json
import queue
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from listn import diarize, speakers
from listn.audio import TARGET_RATE, to_int16
from listn.config import Settings, migrate
from listn.engine import Engine
from listn.library import Library, Segment
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD
from .test_voices import PRINTS, FakeNeural, _voice

TOKEN = "hf_SECRETtoken42"


def _write(path, audio):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(to_int16(audio).tobytes())


class Annotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self.turns:
            yield SimpleNamespace(start=start, end=end), None, label


class FakePipeline:
    def __init__(self, turns, fail_with=None, gate=None):
        self.turns = turns
        self.fail_with = fail_with
        self.gate = gate
        self.device = SimpleNamespace(type="cpu")

    def to(self, device):
        return self

    def __call__(self, file, hook=None):
        hook("segmentation", None)
        if self.gate is not None:
            self.gate.get(timeout=10)
        for i in range(4):
            hook("embeddings", None, total=4, completed=i + 1)
        hook("discrete_diarization", object())
        out = Annotation(self.turns)
        return SimpleNamespace(speaker_diarization=out, exclusive_speaker_diarization=out)


@pytest.fixture
def stack(monkeypatch):
    """pyannote stand-ins: an installed, cached community-1 and a fake voice model."""
    diarize.reset()
    speakers.reset_embedder()
    state = SimpleNamespace(pipeline=FakePipeline([]), loads=[], neural=FakeNeural(PRINTS), error=None)

    def from_pretrained(model_id, token):
        state.loads.append((model_id, token))
        if state.error:
            raise state.error
        return state.pipeline

    monkeypatch.setattr(diarize, "installed", lambda: True)
    monkeypatch.setattr(diarize, "_import_stack", lambda: None)
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: diarize.COMMUNITY_ID)
    monkeypatch.setattr(diarize, "pick_device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda a: a))
    monkeypatch.setattr(speakers, "load_neural_embedder", lambda token=None, allow_download=False: state.neural)
    monkeypatch.setattr(speakers, "voice_embedder", lambda token=None, allow_download=False: state.neural)
    yield state
    diarize.reset()


def _call(library):
    """A two-track recording: Others has Priya (0-4 s) then someone new (4-8 s); Me talks at 8-10 s."""
    meta = library.create("Weekly sync", "live", speakers={"Me": "Me", "Others": "Others"})
    root = library.session_path(meta.id)
    others = np.concatenate([_voice(140, 4.0), _voice(300, 4.0), np.zeros(2 * TARGET_RATE, dtype=np.float32)])
    _write(root / "audio-system.wav", others)
    _write(root / "audio-mic.wav", np.zeros_like(others))
    library.update(meta.id, audio_file="audio.wav", status="done")
    lines = [Segment(0.0, 2.0, "Budget first", "Others"), Segment(2.0, 4.0, "then hiring", "Others"),
             Segment(4.2, 7.5, "Sounds good", "Others"), Segment(8.0, 9.5, "I agree", "Me"),
             Segment(9.6, 9.9, "[paused]", None)]
    library.save_segments(meta.id, lines)
    return meta.id


TURNS = [(0.0, 4.0, "SPEAKER_01"), (4.1, 8.0, "SPEAKER_00")]


def _engine(fake_factory):
    return Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)


def _wait_done(q, sid, timeout=20):
    events = []
    while True:
        event = q.get(timeout=timeout)
        if event.get("job", {}).get("id") != sid:
            continue
        events.append(event)
        if event["type"] == "job.done":
            return events


def test_others_are_split_into_speakers_and_me_stays_me(data_dir, fake_factory, stack):
    stack.pipeline = FakePipeline(TURNS)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    job = engine.enqueue_speakers(sid, Settings(hf_token=TOKEN, summarize_after=False))
    assert job.kind == "speakers"
    events = _wait_done(q, sid)
    assert events[-1]["error"] is None
    meta = engine.library.get(sid)
    segments = engine.library.segments(sid)
    assert [(s.text, s.speaker) for s in segments] == [
        ("Budget first", "SPEAKER_01"), ("then hiring", "SPEAKER_01"), ("Sounds good", "SPEAKER_00"),
        ("I agree", "Me"), ("[paused]", None)]
    assert meta.speakers == {"Me": "Me", "SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}
    assert meta.diarized and meta.speaker_state == "done" and meta.status == "done"
    assert meta.speaker_model == diarize.COMMUNITY_ID
    assert "2 speakers on the Others track" in meta.speaker_note
    progress = [e["job"]["progress"] for e in events if e["type"] == "job.update"]
    assert progress == sorted(progress) and progress[-1] >= 0.95
    assert {e["job"]["detail"] for e in events if e["type"] == "job.update"} >= {"Identifying speakers"}


def test_a_known_voice_is_named_on_the_next_call(data_dir, fake_factory, stack):
    speakers.SpeakerBook(embedder=stack.neural).enroll("Priya", _voice(150, 3.0))
    stack.pipeline = FakePipeline(TURNS)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    engine.enqueue_speakers(sid, Settings(hf_token=TOKEN, summarize_after=False))
    _wait_done(q, sid)
    meta = engine.library.get(sid)
    assert meta.speakers["SPEAKER_01"] == "Priya"
    assert meta.speakers["SPEAKER_00"] == "Speaker 2"
    assert meta.speaker_matches["SPEAKER_01"]["name"] == "Priya"
    assert "recognised Priya" in meta.speaker_note


def test_rerunning_keeps_me_and_relabels_named_lines(data_dir, fake_factory, stack):
    stack.pipeline = FakePipeline(TURNS)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    for _ in range(2):
        engine.enqueue_speakers(sid, Settings(hf_token=TOKEN, summarize_after=False))
        _wait_done(q, sid)
    assert [s.speaker for s in engine.library.segments(sid)][:4] == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "Me"]


def test_a_missing_model_is_skipped_with_a_clear_note(data_dir, fake_factory, stack):
    stack.error = FileNotFoundError("not cached")
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    before = engine.library.segments(sid)
    q = engine.bus.subscribe()
    engine.enqueue_speakers(sid, Settings(hf_token=None, summarize_after=False))
    events = _wait_done(q, sid)
    meta = engine.library.get(sid)
    assert events[-1]["error"]
    assert meta.speaker_state == "skipped" and "Hugging Face token" in meta.speaker_note
    assert meta.status == "done" and engine.library.segments(sid) == before


def test_the_token_never_reaches_notes_or_events(data_dir, fake_factory, stack):
    stack.error = RuntimeError(f"401 Unauthorized for {TOKEN}")
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    engine.enqueue_speakers(sid, Settings(hf_token=TOKEN, summarize_after=False))
    events = _wait_done(q, sid)
    assert TOKEN not in json.dumps(events)
    assert TOKEN not in (engine.library.session_path(sid) / "meta.json").read_text()
    assert engine.library.get(sid).speaker_state == "skipped"


def test_labelling_can_be_cancelled(data_dir, fake_factory, stack):
    gate = queue.Queue()
    stack.pipeline = FakePipeline(TURNS, gate=gate)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    engine.enqueue_speakers(sid, Settings(hf_token=TOKEN, summarize_after=False))
    while q.get(timeout=10).get("job", {}).get("detail") != "Identifying speakers":
        pass
    assert engine.cancel_job(sid)
    gate.put(True)
    events = _wait_done(q, sid)
    meta = engine.library.get(sid)
    assert events[-1]["error"] == "Cancelled."
    assert meta.speaker_state == "cancelled" and meta.status == "done"
    assert engine.library.segments(sid)[0].speaker == "Others"


def test_a_finished_recording_queues_labels_when_a_token_exists(data_dir, fake_factory, stack):
    stack.pipeline = FakePipeline(TURNS)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    session = SimpleNamespace(meta=engine.library.get(sid), segments=engine.library.segments(sid),
                              settings=Settings(hf_token=TOKEN, summarize_after=False))
    q = engine.bus.subscribe()
    assert engine._live_finished(session) is True
    _wait_done(q, sid)
    assert engine.library.get(sid).diarized

    session.settings = Settings(hf_token=TOKEN, diarize_after=False)
    assert engine._live_finished(session) is False


def test_turned_on_without_a_model_the_recording_says_why(data_dir, fake_factory, stack, monkeypatch):
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: None)
    engine = _engine(fake_factory)
    sid = _call(engine.library)
    q = engine.bus.subscribe()
    session = SimpleNamespace(meta=engine.library.get(sid), segments=engine.library.segments(sid),
                              settings=Settings(hf_token=None, diarize_after=True))
    assert engine._live_finished(session) is False
    event = q.get(timeout=5)
    assert event["type"] == "speakers.skipped" and "Hugging Face token" in event["message"]
    assert engine.library.get(sid).speaker_state == "skipped"
    session.settings = Settings(hf_token=None)
    assert engine._live_finished(session) is False


def test_diarize_after_follows_the_token_unless_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(diarize, "installed", lambda: True)
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: None)
    assert Settings().speakers_after() is False
    assert Settings(hf_token="hf_x").speakers_after() is True
    assert Settings(hf_token="hf_x", diarize_after=False).speakers_after() is False
    assert Settings(diarize_after=True).speakers_after() is True
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: diarize.COMMUNITY_ID)
    assert Settings().speakers_after() is True
    public = Settings(hf_token="hf_x").public()
    assert public["diarize_after"] is True and public["diarize_after_auto"] is True and public["hf_token"] is True
    monkeypatch.setattr(diarize, "installed", lambda: False)
    assert Settings(hf_token="hf_x").speakers_after() is False


def test_0_4_settings_are_migrated(data_dir):
    old = {"diarize_after": False, "speaker_threshold": 0.8, "hf_token": "hf_x", "model": "small"}
    new = migrate(old)
    assert new["diarize_after"] is None and new["speaker_threshold"] == 0.55 and new["spectrum_threshold"] == 0.8
    assert migrate({"diarize_after": True})["diarize_after"] is True
    Settings.path().write_text(json.dumps(old))
    loaded = Settings.load()
    assert loaded.diarize_after is None and loaded.speaker_threshold == 0.55 and loaded.spectrum_threshold == 0.8
    loaded.diarize_after = False
    loaded.save()
    assert Settings.load().diarize_after is False
    assert Settings(speaker_threshold=0.1).validated().speaker_threshold == 0.3
    assert Settings(speaker_threshold=2).validated().speaker_threshold == 0.9


def test_an_interrupted_labelling_is_marked_on_restart(data_dir):
    library = Library()
    meta = library.create("Call", "live", status="done", speaker_state="running")
    meta.owner_pid = 999_999_999
    library.save_meta(meta)
    library.recover_interrupted()
    again = library.get(meta.id)
    assert again.status == "done" and again.speaker_state == "error" and "Label speakers" in again.speaker_note


def test_api_relabel_name_and_forget(data_dir, fake_factory, stack):
    stack.pipeline = FakePipeline(TURNS)
    engine = _engine(fake_factory)
    app = create_app(engine=engine, sd=engine.sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        sid = _call(engine.library)
        status = client.get("/api/status").json()
        assert status["diarization"]["installed"] and status["diarization"]["ready"]
        q = engine.bus.subscribe()
        job = client.post(f"/api/sessions/{sid}/diarize")
        assert job.status_code == 200, job.text
        assert job.json()["kind"] == "speakers"
        _wait_done(q, sid)
        busy = client.post("/api/sessions/nope/diarize")
        assert busy.status_code == 404

        named = client.post(f"/api/sessions/{sid}/speakers/name", json={"label": "SPEAKER_01", "name": "Priya"})
        assert named.status_code == 200, named.text
        assert named.json()["profile"]["name"] == "Priya"
        assert client.post(f"/api/sessions/{sid}/speakers/name", json={"label": "Me", "name": "X"}).status_code == 400

        voices = client.get("/api/speakers").json()
        assert voices["neural"] and voices["threshold"] == 0.55 and voices["margin"] == 0.08
        assert voices["speakers"][0]["name"] == "Priya" and not voices["speakers"][0]["needs_reenroll"]
        assert client.post("/api/speakers/reenroll").json()["needs_reenroll"] == []
        assert client.delete("/api/speakers/Priya").json() == {"ok": True}
        assert client.delete("/api/speakers/Priya").status_code == 404


def test_api_says_when_the_extra_is_missing(data_dir, fake_factory, monkeypatch):
    monkeypatch.setattr(diarize, "installed", lambda: False)
    engine = _engine(fake_factory)
    app = create_app(engine=engine, sd=engine.sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        sid = _call(engine.library)
        r = client.post(f"/api/sessions/{sid}/diarize")
        assert r.status_code == 400 and "diarize extra" in r.json()["detail"]
        assert client.get("/api/status").json()["diarization"]["ready"] is False


def test_cli_relabels_a_saved_session(data_dir, stack, capsys, monkeypatch):
    from listn import cli

    stack.pipeline = FakePipeline(TURNS)
    sid = _call(Library())
    assert cli.main(["diarize", sid]) == 0
    out = capsys.readouterr()
    assert "2 speakers on the Others track" in out.out and "Speaker 1" in out.out
    assert "Speakers:" in out.err
    assert cli.main(["diarize", "20990101-000000-abcdef"]) == 1


def test_cli_speaker_check_without_the_extra(data_dir, monkeypatch, capsys):
    from listn import cli

    monkeypatch.setattr(diarize, "installed", lambda: False)
    assert cli.main(["speakers", "--check"]) == 3
    assert '"installed": false' in capsys.readouterr().out
    assert cli.main(["diarize", "20990101-000000-abcdef"]) == 3
