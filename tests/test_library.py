import json

import pytest

from listn.config import Settings
from listn.library import Library, Segment, export, fmt_clock


@pytest.fixture
def lib(data_dir):
    return Library()


def test_create_save_get_list(lib):
    a = lib.create("Standup", "live", model="small")
    lib.save_segments(a.id, [Segment(0, 2, "Morning all"), Segment(2, 5, "Let's begin")])
    b = lib.create("Interview", "file")
    assert {m.id for m in lib.list()} == {a.id, b.id}
    assert lib.get(a.id).model == "small"
    assert [s.text for s in lib.segments(a.id)] == ["Morning all", "Let's begin"]


def test_data_lives_under_data_dir(lib, data_dir):
    meta = lib.create("x", "live")
    assert (data_dir / "sessions" / meta.id / "meta.json").exists()


def test_search_text_and_title(lib):
    a = lib.create("Budget review", "live")
    lib.save_segments(a.id, [Segment(0, 1, "The Q3 forecast is up"), Segment(1, 2, "great")])
    b = lib.create("Hiring sync", "live")
    lib.save_segments(b.id, [Segment(0, 1, "forecast for hiring")])
    res = lib.search("forecast")
    assert {r["meta"]["id"] for r in res} == {a.id, b.id}
    assert lib.search("budget")[0]["meta"]["id"] == a.id
    assert lib.search("nothing-here") == []
    assert len(lib.search("")) == 2


def test_rejects_path_traversal(lib):
    for bad in ("../etc", "..", "20240101-000000-zzzzzz", ""):
        with pytest.raises(KeyError):
            lib.get(bad)


def test_delete_and_update(lib):
    m = lib.create("t", "live")
    assert lib.update(m.id, title="Renamed", id="evil").title == "Renamed"
    assert lib.get(m.id).id == m.id
    lib.delete(m.id)
    with pytest.raises(KeyError):
        lib.get(m.id)


def test_recover_interrupted(lib):
    m = lib.create("crashed", "live", status="recording")
    assert lib.recover_interrupted() == 1
    assert lib.get(m.id).status == "error"


def test_import_audio_copies_into_session(lib, tmp_path):
    src = tmp_path / "call.MP3"
    src.write_bytes(b"123")
    m = lib.create("t", "file")
    dest = lib.import_audio(m, src)
    assert dest.name == "audio.mp3" and src.exists()
    assert lib.audio_path(lib.get(m.id)) == dest


def test_exports():
    from listn.library import SessionMeta

    meta = SessionMeta(id="20240101-000000-abcdef", title="Call", created_at="2024-01-01T10:00:00",
                       source="live", duration=3725, speakers={"SPEAKER_00": "Ana"})
    segs = [Segment(1.5, 3.25, "Hello", "SPEAKER_00"), Segment(3725.0, 3726.5, "Bye", None)]
    assert export(meta, segs, "txt") == "[00:01] Ana: Hello\n[1:02:05] Bye\n"
    srt = export(meta, segs, "srt")
    assert "1\n00:00:01,500 --> 00:00:03,250\nAna: Hello" in srt
    assert "01:02:05,000 --> 01:02:06,500" in srt
    vtt = export(meta, segs, "vtt")
    assert vtt.startswith("WEBVTT") and "<v Ana>Hello" in vtt and "00:00:01.500" in vtt
    assert json.loads(export(meta, segs, "json"))["segments"][0]["speaker"] == "SPEAKER_00"
    assert export(meta, segs, "md").startswith("# Call")
    with pytest.raises(ValueError):
        export(meta, segs, "docx")


def test_fmt_clock():
    assert fmt_clock(0) == "00:00"
    assert fmt_clock(61.9) == "01:01"
    assert fmt_clock(3600) == "1:00:00"


def test_settings_roundtrip_and_sanitize(data_dir):
    s = Settings.load()
    assert s.model == "small"
    s.model, s.language, s.hf_token = "base", " EN ", "hf_secret"
    s.validated().save()
    loaded = Settings.load()
    assert loaded.model == "base" and loaded.language == "en"
    assert loaded.public()["hf_token"] is True
    assert "hf_secret" not in json.dumps(loaded.public())
    (data_dir / "settings.json").write_text('{"model": "huge", "chunk_seconds": 999, "junk": 1}')
    bad = Settings.load()
    assert bad.model == "small" and bad.chunk_seconds == 30.0
    (data_dir / "settings.json").write_text("{not json")
    assert Settings.load().model == "small"
