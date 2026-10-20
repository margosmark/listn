from fastapi.testclient import TestClient

from listn.config import Settings
from listn.engine import Engine
from listn.index import SearchIndex
from listn.library import Library, Segment
from listn.server import create_app
from listn.ask import ask

from .conftest import MAC_DEVICES, FakeSD


def _lib():
    lib = Library()
    planning = lib.create("Planning", "live")
    lib.save_segments(planning.id, [
        Segment(1.0, 3.0, "Let's ship the export on Friday.", "Me"),
        Segment(4.0, 7.0, "Priya will review the numbers.", "Others"),
    ])
    other = lib.create("Lunch", "live")
    lib.save_segments(other.id, [Segment(0.0, 2.0, "The cafe is closed on Monday.", "Me")])
    return lib, planning, other


def test_semantic_search_finds_a_line_without_the_exact_phrase(data_dir):
    lib, planning, other = _lib()
    index = SearchIndex(lib)
    hits = index.search("friday ship date")
    assert hits
    assert hits[0]["session_id"] == planning.id
    assert hits[0]["start"] == 1.0
    assert all(hit["session_id"] != other.id for hit in hits)
    assert index.search("xylophone quantum") == []


def test_index_updates_only_changed_sessions(data_dir):
    lib, planning, _other = _lib()
    index = SearchIndex(lib)
    first = index.sync()
    assert first["updated"] == 2
    assert index.sync()["updated"] == 0
    lib.save_segments(planning.id, [Segment(1.0, 3.0, "Let's ship the export next Monday.", "Me")])
    again = index.sync()
    assert again["updated"] == 1
    lib.delete(planning.id)
    assert index.sync()["removed"] == 1
    assert all(hit["session_id"] != planning.id for hit in index.search("ship the export"))


def test_ask_cites_a_specific_line(data_dir):
    lib, planning, _other = _lib()
    index = SearchIndex(lib)

    def complete(prompt):
        assert "Friday" in prompt
        return '{"answer":"The export ships Friday.","citations":[1]}'

    result = ask(index, "When does the export ship?", Settings(), complete=complete)
    assert "Friday" in result["answer"]
    assert result["citations"][0]["session_id"] == planning.id
    assert result["citations"][0]["start"] == 1.0


def test_ask_without_a_model_still_cites_audio(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, default=(0, 3))
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    lib, planning, _other = _lib_on(engine)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        body = client.post("/api/ask", json={"question": "friday ship date"}).json()
        assert body["citations"]
        assert body["citations"][0]["session_id"] == planning.id
        listed = client.get("/api/sessions", params={"q": "friday ship date"}).json()["results"]
        assert listed[0]["meta"]["id"] == planning.id


def _lib_on(engine):
    lib = engine.library
    planning = lib.create("Planning", "live")
    lib.save_segments(planning.id, [Segment(1.0, 3.0, "Let's ship the export on Friday.", "Me")])
    other = lib.create("Lunch", "live")
    lib.save_segments(other.id, [Segment(0.0, 2.0, "The cafe is closed on Monday.", "Me")])
    return lib, planning, other
