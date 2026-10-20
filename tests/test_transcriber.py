import numpy as np
import pytest

from listn import transcriber as tr
from listn.transcriber import RawSegment, _WindowedBackend, is_hallucination

from .conftest import FakeTranscriber


@pytest.mark.parametrize("text", ["Thanks for watching!", "  thank you for watching. ",
                                  "Subtitles by the Amara.org community", "[BLANK_AUDIO]", "(music)", "...", ""])
def test_known_hallucinations_dropped(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", ["OK.", "No.", "Yes", "Thank you.", "So the budget is fine."])
def test_real_short_answers_kept(text):
    assert not is_hallucination(text, no_speech_prob=0.1)


def test_fillers_dropped_only_when_no_speech_likely():
    assert is_hallucination("Thank you.", no_speech_prob=0.7)
    assert is_hallucination("you", quiet=True)
    assert not is_hallucination("you", no_speech_prob=0.05)


def test_transcribe_applies_offset_and_filters():
    t = FakeTranscriber()
    audio = (0.2 * np.sin(np.arange(32000) / 5)).astype(np.float32)
    segs = t.transcribe(audio, offset=12.0)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(12.0)
    assert segs[0].end == pytest.approx(13.0)
    assert segs[0].text == "chunk 1 first half"


def test_transcribe_reports_progress_to_completion():
    seen = []
    FakeTranscriber().transcribe(np.full(16000, 0.1, dtype=np.float32), progress=seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_empty_audio_returns_nothing():
    assert FakeTranscriber().transcribe(np.zeros(0, dtype=np.float32)) == []


class Windowed(_WindowedBackend):
    window_seconds = 2

    def __init__(self):
        super().__init__("tiny")
        self.prompts = []

    def _run(self, audio, language, prompt):
        self.prompts.append(prompt)
        return [{"start": 0.5, "end": 1.0, "text": "hello"}]


def test_windowed_backend_offsets_windows_and_skips_silence():
    audio = np.concatenate([np.full(32000, 0.1), np.zeros(32000), np.full(32000, 0.1)]).astype(np.float32)
    w = Windowed()
    segs = list(w.segments(audio))
    assert [s.start for s in segs] == [0.5, 4.5]
    assert w.prompts == [None, "hello"]


def test_resolve_backend_on_linux_defaults_to_faster_whisper(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "linux")
    assert tr.resolve_backend("auto") == "faster-whisper"
    assert tr.resolve_backend("mlx") == "mlx"


def test_get_transcriber_caches_and_reports_missing_backend():
    made = []

    def factory(model, download=None):
        made.append(model)
        return FakeTranscriber(model)

    tr.register_backend("fake", factory)
    a = tr.get_transcriber("tiny", "fake")
    assert tr.get_transcriber("tiny", "fake") is a
    tr.get_transcriber("base", "fake")
    assert made == ["tiny", "base"]

    def missing(model, download=None):
        raise ImportError("nope")

    tr.register_backend("mlx", missing)
    with pytest.raises(RuntimeError, match="listn\\[mlx\\]"):
        tr.get_transcriber("tiny", "mlx")
    tr.register_backend("mlx", tr.MlxWhisper)


def test_raw_segment_defaults():
    assert RawSegment(0, 1, "x").no_speech_prob == 0.0
