import json

from fastapi.testclient import TestClient

from listn.config import Settings
from listn.engine import Engine
from listn.library import Library, Segment
from listn.server import create_app
from listn.summary import build_notes, extractive_notes, parse_model_notes, write_notes

from .conftest import MAC_DEVICES, FakeSD


def _meeting(lib: Library):
    meta = lib.create("Planning", "live", speakers={"Me": "Me", "Others": "Others"})
    lib.save_segments(meta.id, [
        Segment(0, 2, "Let's ship the export on Friday.", "Me"),
        Segment(2, 5, "Agreed, we'll go with Markdown.", "Others"),
        Segment(5, 8, "I'll send the notes to Priya.", "Me"),
        Segment(8, 11, "Can you schedule the review?", "Others"),
    ])
    return meta


def test_extractive_notes_attribute_me_and_others(data_dir):
    lib = Library()
    meta = _meeting(lib)
    notes = extractive_notes(lib.segments(meta.id), meta.speakers)
    assert "Markdown" in notes["summary"]
    assert any("Markdown" in item for item in notes["decisions"])
    owners = {item["owner"] for item in notes["actions"]}
    assert "Me" in owners
    assert "Others" in owners
    assert notes["backend"] == "extractive"


def test_model_json_is_parsed_and_a_bad_reply_falls_back(data_dir):
    lib = Library()
    meta = _meeting(lib)
    raw = """Sure.
```json
{"summary":"Ship Friday.","decisions":["Markdown"],"actions":[{"owner":"Me","text":"Send the notes"}]}
```"""
    parsed = parse_model_notes(raw)
    assert parsed["summary"] == "Ship Friday."
    assert parsed["actions"][0]["owner"] == "Me"
    notes = build_notes(lib.segments(meta.id), meta.speakers, complete=lambda prompt: raw)
    assert notes["backend"] == "model"
    assert notes["summary"] == "Ship Friday."
    fallen = build_notes(lib.segments(meta.id), meta.speakers, complete=lambda prompt: "not json")
    assert fallen["backend"] == "extractive"
    assert any(item["owner"] == "Me" for item in fallen["actions"])


def test_notes_roundtrip_and_edit_api(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, default=(0, 3))
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        lib = engine.library
        meta = _meeting(lib)
        notes = write_notes(lib, meta.id, Settings(summarize_after=False), complete=lambda prompt: "nope")
        assert notes["backend"] == "extractive"
        saved = client.put(f"/api/sessions/{meta.id}/summary", json={
            "summary": "Edited recap.",
            "decisions": ["Keep it local"],
            "actions": [{"owner": "Priya", "text": "Review the export"}],
        })
        assert saved.status_code == 200
        body = saved.json()
        assert body["edited"] is True
        assert body["summary"] == "Edited recap."
        assert body["actions"][0]["owner"] == "Priya"
        again = client.get(f"/api/sessions/{meta.id}").json()["notes"]
        assert again["decisions"] == ["Keep it local"]
        forced = client.post(f"/api/sessions/{meta.id}/summary")
        assert forced.status_code == 200
        assert forced.json()["edited"] is False
        assert json.loads((data_dir / "sessions" / meta.id / "summary.json").read_text())["summary"]


def test_summarize_command(data_dir, capsys):
    from listn import cli

    meta = _meeting(Library())
    assert cli.main(["summarize", meta.id]) == 0
    out = capsys.readouterr().out
    assert "I'll send the notes" in out or "send the notes" in out.lower()
    assert cli.main(["summarize", "20990101-000000-000000"]) == 1
