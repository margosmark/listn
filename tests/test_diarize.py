from types import SimpleNamespace

from listn.diarize import assign_speakers, friendly_names, turns_from_output
from listn.library import Segment


class FakeAnnotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self.turns:
            yield SimpleNamespace(start=start, end=end), None, label


TURNS = [(0.0, 4.0, "SPEAKER_01"), (4.2, 9.0, "SPEAKER_00")]


def test_turns_from_both_pyannote_apis():
    assert turns_from_output(FakeAnnotation(TURNS)) == TURNS
    assert turns_from_output(SimpleNamespace(speaker_diarization=FakeAnnotation(TURNS))) == TURNS
    assert turns_from_output(object()) == []


def test_assign_by_largest_overlap():
    segs = [Segment(0.5, 3.0, "a"), Segment(3.0, 8.0, "b")]
    out = assign_speakers(segs, TURNS)
    assert [s.speaker for s in out] == ["SPEAKER_01", "SPEAKER_00"]
    assert [s.text for s in out] == ["a", "b"]


def test_segment_in_pause_gets_nearest_speaker():
    out = assign_speakers([Segment(4.02, 4.15, "uh")], TURNS)
    assert out[0].speaker in {"SPEAKER_01", "SPEAKER_00"}


def test_far_segment_left_unlabelled():
    assert assign_speakers([Segment(20, 21, "x")], TURNS)[0].speaker is None
    assert assign_speakers([Segment(0, 1, "x")], [])[0].speaker is None


def test_friendly_names_by_first_appearance():
    segs = assign_speakers([Segment(0.5, 3.0, "a"), Segment(5, 8, "b"), Segment(1, 2, "c")], TURNS)
    assert friendly_names(segs) == {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}


# ---- pyannote 4 / community-1 ----------------------------------------------

import os

import numpy as np
import pytest

from listn import diarize, hub


class Out4:
    """Shape of pyannote 4's DiarizeOutput."""

    def __init__(self, turns, exclusive):
        self.speaker_diarization = FakeAnnotation(turns)
        self.exclusive_speaker_diarization = FakeAnnotation(exclusive)
        self.speaker_embeddings = None


def test_exclusive_diarization_is_preferred_and_3x_tuples_are_accepted():
    overlapping = [(0.0, 5.0, "SPEAKER_00"), (4.0, 9.0, "SPEAKER_01")]
    exclusive = [(0.0, 4.5, "SPEAKER_00"), (4.5, 9.0, "SPEAKER_01")]
    assert turns_from_output(Out4(overlapping, exclusive)) == exclusive
    assert turns_from_output((FakeAnnotation(TURNS), np.zeros((2, 256)))) == TURNS


def test_others_lines_without_speech_nearby_keep_the_default_label():
    out = assign_speakers([Segment(0.5, 3.0, "a"), Segment(30, 31, "echo")], TURNS, default="Others")
    assert [s.speaker for s in out] == ["SPEAKER_01", "Others"]


def test_alignment_follows_turns_across_a_long_call():
    turns = [(0.0, 12.0, "SPEAKER_02"), (12.0, 30.0, "SPEAKER_00"), (30.5, 41.0, "SPEAKER_02")]
    chunks = [Segment(t, t + 6.0, f"line {t}") for t in (0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0)]
    out = assign_speakers(chunks, turns)
    assert [s.speaker for s in out] == ["SPEAKER_02", "SPEAKER_02", "SPEAKER_00", "SPEAKER_00", "SPEAKER_00",
                                        "SPEAKER_02", "SPEAKER_02"]
    # A chunk that straddles a change goes to whoever talks for longer in it.
    straddle = assign_speakers([Segment(10.0, 16.0, "x")], turns)
    assert straddle[0].speaker == "SPEAKER_00"
    assert friendly_names(out) == {"SPEAKER_02": "Speaker 1", "SPEAKER_00": "Speaker 2"}


def test_friendly_names_skip_me_and_others_and_follow_time_order():
    segs = [Segment(5, 6, "b", "SPEAKER_00"), Segment(0, 1, "me", "Me"), Segment(2, 3, "a", "SPEAKER_03"),
            Segment(4, 4.5, "?", "Others")]
    assert friendly_names(segs) == {"SPEAKER_03": "Speaker 1", "SPEAKER_00": "Speaker 2"}


def test_speaker_audio_takes_the_longest_turns_up_to_the_limit():
    rate = 100
    audio = np.arange(1000, dtype=np.float32)
    turns = [(0.0, 1.0, "A"), (2.0, 6.0, "A"), (6.0, 7.0, "B"), (8.0, 8.2, "A")]
    clip = diarize.speaker_audio(audio, turns, "A", rate=rate, max_seconds=4.5)
    assert clip.size == 450
    assert clip[0] == 200 and clip[399] == 599 and clip[400] == 0
    assert diarize.speaker_audio(audio, turns, "nobody", rate=rate).size == 0


def test_progress_hook_maps_pyannote_steps_and_can_cancel():
    seen = []
    hook = diarize.progress_hook(seen.append)
    hook("segmentation", None)
    hook("segmentation", object())
    hook("speaker_counting", object())
    for i in range(4):
        hook("embeddings", None, total=4, completed=i + 1)
    hook("discrete_diarization", object())
    hook("something-new", object())
    assert seen == sorted(seen)
    assert seen[0] == 0.0 and seen[-1] == pytest.approx(0.97)
    assert pytest.approx(0.9) in seen
    stop = diarize.progress_hook(None, cancelled=lambda: True)
    with pytest.raises(diarize.Cancelled):
        stop("embeddings", None, total=10, completed=1)


class FakePipeline:
    def __init__(self, turns):
        self.turns = turns
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, file, hook=None):
        self.calls.append(file)
        hook("segmentation", None)
        hook("embeddings", None, total=2, completed=2)
        return Out4(self.turns, self.turns)


@pytest.fixture
def fake_stack(monkeypatch):
    diarize.reset()
    monkeypatch.setattr(diarize, "installed", lambda: True)
    monkeypatch.setattr(diarize, "_import_stack", lambda: None)
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: None)
    monkeypatch.setattr(diarize, "pick_device", lambda: (_ for _ in ()).throw(RuntimeError("no torch")))
    attempts = []
    yield attempts
    diarize.reset()


def test_community_1_is_loaded_first_from_the_cache_without_the_token(fake_stack, monkeypatch):
    pipeline = FakePipeline(TURNS)

    def from_pretrained(model_id, token):
        fake_stack.append((model_id, token, os.environ.get("HF_HUB_OFFLINE")))
        return pipeline

    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    got, model_id = diarize.load_pipeline("hf_secret_token")
    assert got is pipeline and model_id == diarize.COMMUNITY_ID
    assert fake_stack == [(diarize.COMMUNITY_ID, None, "1")]
    assert hub.hub_offline()
    assert diarize.load_pipeline("hf_secret_token")[1] == diarize.COMMUNITY_ID
    assert len(fake_stack) == 1


def test_community_1_downloads_once_with_the_token_then_goes_offline(fake_stack, monkeypatch):
    pipeline = FakePipeline(TURNS)

    def from_pretrained(model_id, token):
        fake_stack.append((model_id, token, os.environ.get("HF_HUB_OFFLINE")))
        if token is None:
            raise FileNotFoundError("not in the cache")
        return pipeline

    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    statuses = []
    _p, model_id = diarize.load_pipeline("hf_secret_token", status=statuses.append)
    assert model_id == diarize.COMMUNITY_ID
    assert fake_stack == [(diarize.COMMUNITY_ID, None, "1"), (diarize.COMMUNITY_ID, "hf_secret_token", None)]
    assert any("Downloading" in s for s in statuses)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_3_1_is_the_fallback_when_community_1_fails(fake_stack, monkeypatch):
    pipeline = FakePipeline(TURNS)

    def from_pretrained(model_id, token):
        fake_stack.append(model_id)
        if model_id == diarize.COMMUNITY_ID:
            raise RuntimeError("401 for hf_secret_token: accept the terms")
        return pipeline

    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    _p, model_id = diarize.load_pipeline("hf_secret_token")
    assert model_id == diarize.LEGACY_ID
    assert fake_stack[0] == diarize.COMMUNITY_ID and fake_stack[-1] == diarize.LEGACY_ID


def test_without_a_token_or_cache_the_message_says_what_to_do(fake_stack, monkeypatch):
    def from_pretrained(model_id, token):
        raise FileNotFoundError("offline and not cached")

    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    with pytest.raises(diarize.DiarizationUnavailable) as err:
        diarize.load_pipeline(None)
    assert "Hugging Face token in Settings" in str(err.value)
    assert "speaker-diarization-community-1" in str(err.value)


def test_failures_with_a_token_list_the_terms_pages_and_never_the_token(fake_stack, monkeypatch):
    def from_pretrained(model_id, token):
        raise RuntimeError(f"403 Client Error for token {token}")

    monkeypatch.setattr(diarize, "_from_pretrained", from_pretrained)
    with pytest.raises(diarize.DiarizationUnavailable) as err:
        diarize.load_pipeline("hf_secret_token")
    text = str(err.value)
    assert "hf_secret_token" not in text
    for page in ("speaker-diarization-community-1", "speaker-diarization-3.1", "segmentation-3.0"):
        assert page in text


def test_diarize_segments_reports_progress_and_names_known_voices(fake_stack, monkeypatch):
    pipeline = FakePipeline([(0.0, 4.0, "SPEAKER_01"), (4.2, 9.0, "SPEAKER_00")])
    monkeypatch.setattr(diarize, "_from_pretrained", lambda model_id, token: pipeline)
    import sys
    import types

    fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    progress = []

    def identify(audio, turns, labels):
        assert labels == ["SPEAKER_01", "SPEAKER_00"]
        return {"SPEAKER_00": {"name": "Priya", "score": 0.81}, "SPEAKER_01": {"name": None, "score": 0.2}}

    result = diarize.diarize_segments(np.zeros(16000 * 9, dtype=np.float32),
                                      [Segment(0.5, 3.0, "hi"), Segment(5, 8, "hello")], "tok",
                                      progress=progress.append, identify=identify, default="Others")
    assert result.model_id == diarize.COMMUNITY_ID
    assert result.names == {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Priya"}
    assert [s.speaker for s in result.segments] == ["SPEAKER_01", "SPEAKER_00"]
    assert progress[-1] == 1.0 and progress == sorted(progress)
    assert pipeline.calls[0]["sample_rate"] == 16000


def test_readiness_explains_each_state(monkeypatch):
    diarize.reset()
    monkeypatch.setattr(diarize, "installed", lambda: False)
    off = diarize.readiness("tok")
    assert not off["ready"] and "diarize extra" in off["message"]
    monkeypatch.setattr(diarize, "installed", lambda: True)
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: None)
    no_token = diarize.readiness(None)
    assert not no_token["ready"] and "Hugging Face token" in no_token["message"]
    first = diarize.readiness("tok")
    assert first["ready"] and "downloads once" in first["message"]
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: diarize.COMMUNITY_ID)
    cached = diarize.readiness(None)
    assert cached["ready"] and "offline" in cached["message"]
    assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in cached["terms"]


def test_a_missing_model_never_opens_a_socket_without_a_token(fake_stack, monkeypatch):
    """No token means no download attempt: the failure has to happen locally."""
    import socket

    monkeypatch.setattr(diarize, "_from_pretrained",
                        lambda model_id, token: (_ for _ in ()).throw(FileNotFoundError("not cached")))
    touched: list = []

    def connect(self, *args, **kwargs):
        touched.append(args)
        raise AssertionError("the network was touched")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: touched.append(a))
    diarize.readiness(None)
    with pytest.raises(diarize.DiarizationUnavailable):
        diarize.load_pipeline(None)
    assert touched == []
    assert hub.hub_offline()


def test_readiness_describes_this_machine_and_separates_the_download(monkeypatch):
    """Readiness ("can label now") and "will download on first use" are different facts."""
    diarize.reset()
    monkeypatch.setattr(diarize, "installed", lambda: True)
    monkeypatch.setattr(diarize, "cached_pipeline", lambda: None)

    with_token = diarize.readiness("tok")
    assert with_token["ready"] and with_token["needs_download"]
    assert "downloads once" in with_token["message"]

    no_token = diarize.readiness(None)
    assert not no_token["ready"] and not no_token["needs_download"]
    assert "Hugging Face token" in no_token["message"]

    monkeypatch.setattr(diarize, "cached_pipeline", lambda: diarize.COMMUNITY_ID)
    cached = diarize.readiness(None)
    assert cached["ready"] and not cached["needs_download"] and "offline" in cached["message"]

    for state in (with_token, no_token, cached):
        assert "Mac" not in state["message"], state["message"]


def test_the_speaker_model_runs_on_the_cpu_unless_apple_silicon_is_there(monkeypatch):
    import sys
    import types

    cpu_only = types.SimpleNamespace(backends=types.SimpleNamespace(),
                                     device=lambda name: types.SimpleNamespace(type=name))
    monkeypatch.setitem(sys.modules, "torch", cpu_only)
    assert diarize.pick_device().type == "cpu"

    mps = types.SimpleNamespace(type="mps", is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(backends=types.SimpleNamespace(mps=mps),
                                              device=lambda name: types.SimpleNamespace(type=name)))
    assert diarize.pick_device().type == "mps"


def test_pyannote_telemetry_is_forced_off(monkeypatch):
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")
    diarize.telemetry_off()
    # pyannote counts only "true" / "1" as on; its own switch writes "false".
    assert os.environ["PYANNOTE_METRICS_ENABLED"].lower() in ("0", "false")
    from listn.config import disable_network_telemetry

    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    disable_network_telemetry()
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"


def test_hub_offline_switches_the_huggingface_constant_too():
    import huggingface_hub.constants as constants

    hub.enable_hub_offline()
    assert constants.HF_HUB_OFFLINE is True and hub.hub_offline()
    hub.disable_hub_offline()
    assert constants.HF_HUB_OFFLINE is False and "HF_HUB_OFFLINE" not in os.environ
    hub.enable_hub_offline()
    assert hub.is_cached("listn-test/does-not-exist", "config.yaml") is False


def test_redact_hides_tokens():
    assert hub.redact("bad token hf_abcdef in url", "hf_abcdef") == "bad token [token] in url"
    assert hub.redact("nothing", None) == "nothing"
