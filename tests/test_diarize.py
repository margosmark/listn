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
