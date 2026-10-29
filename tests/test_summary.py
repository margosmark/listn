import json
import re

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


# --------------------------------------------------------------------------- notes v2

import threading

from listn import llm, summary
from listn.summary import build_turns, chunk_turns, edited_notes, finish_notes


def _team(lib: Library):
    meta = lib.create("Sync", "live", speakers={"Me": "Me", "SPEAKER_00": "Alex", "SPEAKER_01": "Priya"})
    lib.save_segments(meta.id, [
        Segment(0, 4, "What input do you need from us to make the report useful?", "SPEAKER_01"),
        Segment(4, 9, "I will message Dana and Lee about the expectations by Friday.", "SPEAKER_00"),
        Segment(9, 12, "Okay, so we go with a monthly review instead of weekly.", "Me"),
        Segment(12, 15, "Agreed, monthly it is.", "SPEAKER_01"),
        Segment(15, 20, "I'll draft the handoff doc and share it in Slack.", "Me"),
        Segment(20, 24, "Let's talk about the handoff meeting next Monday.", "SPEAKER_01"),
    ])
    return meta


def test_turns_are_numbered_and_merge_one_speaker():
    segs = [Segment(0, 1, "Hello there.", "Me"), Segment(1, 2, "Still me.", "Me"),
            Segment(2, 3, "[paused]", "Me"), Segment(3, 4, "Now you.", None)]
    turns = build_turns(segs, {"Me": "Me"})
    assert [(t.number, t.speaker) for t in turns] == [(1, "Me"), (2, "Others")]
    assert turns[0].text == "Hello there. Still me."
    assert "L1 [00:00] Me: Hello there." in summary.prompt_for(segs, {"Me": "Me"})


def test_model_notes_are_grounded_and_owners_resolved(data_dir):
    lib = Library()
    meta = _team(lib)
    reply = json.dumps({
        "summary": "Alex will clarify expectations; the team moves to a monthly review.",
        "key_points": [{"text": "Priya asked what input Alex needs for the report", "line": 1}],
        "decisions": [{"text": "Review monthly instead of weekly", "line": 3},
                      {"text": "Agreed to discuss the handoff meeting", "line": 6}],
        "actions": [
            {"owner": "alex", "text": "Alex will message Dana and Lee about expectations", "due": "Friday", "line": 2},
            {"owner": "I", "text": "Draft the handoff doc and share it in Slack", "due": "null", "line": 5},
            {"owner": "Me", "text": "draft the handoff doc and share it in Slack", "due": None, "line": 5},
            {"owner": "Me", "text": "Book a flight to Denver", "due": None, "line": 3},
        ],
        "open_questions": [],
    })
    notes = build_notes(lib.segments(meta.id), meta.speakers, complete=lambda prompt: reply)
    assert notes["backend"] == "model" and notes["version"] == 2
    assert notes["key_points"][0]["start"] == 0
    assert notes["decisions"] == ["Review monthly instead of weekly"]
    assert notes["decision_items"][0]["start"] == 9
    assert [q["text"] for q in notes["open_questions"]] == ["Agreed to discuss the handoff meeting"]
    owners = [(a["owner"], a["text"], a.get("due"), a.get("start")) for a in notes["actions"]]
    assert owners == [
        ("Alex", "Message Dana and Lee about expectations", "Friday", 4),
        ("Me", "Draft the handoff doc and share it in Slack", None, 15),
    ]
    assert all("line" not in a for a in notes["actions"])


def test_prompt_names_people_and_explains_me(data_dir):
    lib = Library()
    meta = _team(lib)
    seen = []
    build_notes(lib.segments(meta.id), meta.speakers, complete=lambda p: seen.append(p) or "{}")
    assert "Speakers: Priya, Alex, Me" in seen[0]
    assert '"Me" is the person these notes are for' in seen[0]
    assert "L2 [00:04] Alex: I will message Dana" in seen[0]


def test_bad_json_is_retried_once_then_falls_back(data_dir):
    lib = Library()
    meta = _team(lib)
    replies = iter(["Sure! Here are the notes:", '{"summary": "Fine.", "actions": [],}'])
    calls = []

    def runner(prompt):
        calls.append(prompt)
        return next(replies)

    notes = build_notes(lib.segments(meta.id), meta.speakers, complete=runner)
    assert notes["summary"] == "Fine." and len(calls) == 2
    assert "previous reply was not a valid JSON" in calls[1]
    calls.clear()
    fallen = build_notes(lib.segments(meta.id), meta.speakers, complete=lambda p: calls.append(p) or "nope")
    assert fallen["backend"] == "extractive" and len(calls) == 2


def test_long_meetings_are_chunked_and_merged(data_dir, monkeypatch):
    monkeypatch.setattr(summary, "SINGLE_PASS_CHARS", 600)
    monkeypatch.setattr(summary, "CHUNK_CHARS", 400)
    lib = Library()
    meta = lib.create("Long", "live", speakers={"Me": "Me", "SPEAKER_00": "Sam"})
    segs = []
    for i in range(30):
        who = "Me" if i % 2 else "SPEAKER_00"
        text = f"Point number {i} about the rollout plan and the package configuration details."
        if i == 4:
            text = "I'm going to clone the configuration profile and test it with Jo."
        if i == 25:
            text = "I'll forward the Slack thread to Sam after this call."
        segs.append(Segment(i * 10, i * 10 + 9, text, who))
    lib.save_segments(meta.id, segs)
    prompts = []

    def runner(prompt):
        prompts.append(prompt)
        if prompt.startswith("These are notes taken on consecutive parts"):
            return json.dumps({"summary": "Sam and Me planned the rollout.",
                               "key_points": [{"text": "Clone the profile for a test", "time": "00:40"}]})
        part_lines = [int(n) for n in re.findall(r"^L(\d+) ", prompt, re.M)]
        actions = []
        if 5 in part_lines:
            actions.append({"owner": "Sam", "text": "Clone the configuration profile and test it with Jo", "line": 5})
        if 26 in part_lines:
            actions.append({"owner": "Me", "text": "Forward the Slack thread to Sam", "line": 26})
        return json.dumps({"summary": "A part.", "key_points": [], "decisions": [], "actions": actions})

    notes = build_notes(lib.segments(meta.id), meta.speakers, complete=runner)
    assert notes["chunks"] > 2
    assert sum(p.startswith("Below is part") for p in prompts) == notes["chunks"]
    assert notes["summary"] == "Sam and Me planned the rollout."
    assert notes["key_points"][0]["start"] == 40
    got = [(a["owner"], a["start"]) for a in notes["actions"]]
    assert got == [("Sam", 40), ("Me", 250)]  # overlap duplicates merged


def test_chunks_overlap_and_cover_everything():
    segs = [Segment(i, i + 1, "word " * 30, "Me" if i % 2 else None) for i in range(40)]
    turns = build_turns(segs, {})
    chunks = chunk_turns(turns, max_chars=500, overlap=2)
    assert len(chunks) > 3
    covered = {t.number for c in chunks for t in c}
    assert covered == {t.number for t in turns}
    assert chunks[1][0] is chunks[0][-2]


def test_auto_notes_keep_hand_edits(data_dir):
    lib = Library()
    meta = _team(lib)
    lib.save_notes(meta.id, {"summary": "Mine.", "decisions": [], "actions": [], "edited": True})
    kept = write_notes(lib, meta.id, Settings(), complete=lambda p: '{"summary":"New"}', force=False)
    assert kept["summary"] == "Mine."
    forced = write_notes(lib, meta.id, Settings(), complete=lambda p: '{"summary":"New"}')
    assert forced["summary"] == "New" and forced["edited"] is False


def test_edits_keep_source_times(data_dir):
    previous = {"summary": "x", "decisions": ["Go monthly"], "decision_items": [{"text": "Go monthly", "start": 9}],
                "actions": [{"owner": "Me", "text": "Draft the doc", "start": 15, "due": "Friday"}],
                "key_points": [{"text": "A point", "start": 3}], "backend": "mlx"}
    notes = edited_notes("y", ["Go monthly", "New one"], [{"owner": "Me", "text": "Draft the doc"},
                                                          {"owner": "Priya", "text": "Other"}],
                         previous, key_points=["A point"])
    assert notes["decision_items"] == [{"text": "Go monthly", "start": 9}, {"text": "New one"}]
    assert notes["actions"][0] == {"owner": "Me", "text": "Draft the doc", "start": 15, "due": "Friday"}
    assert notes["key_points"] == [{"text": "A point", "start": 3}]
    assert notes["edited"] is True and notes["backend"] == "mlx"


def test_extractive_skips_small_talk():
    segs = [Segment(0, 1, "I just need to grab some water, I'm going to step away.", None),
            Segment(1, 2, "I'll be out of here soon, so all good.", None),
            Segment(2, 3, "I'll send the rollout notes to Sam today.", "Me"),
            Segment(3, 4, "Thank you.", None), Segment(4, 5, "Thank you.", None)]
    notes = extractive_notes(segs, {"Me": "Me"})
    assert [a["text"] for a in notes["actions"]] == ["I'll send the rollout notes to Sam today."]
    assert notes["actions"][0]["start"] == 2
    assert "Thank you" not in notes["summary"]


def test_owner_resolution():
    turns = build_turns([Segment(0, 1, "hi", "SPEAKER_00")], {"SPEAKER_00": "Sam Lee"})
    names = ["Me", "Sam Lee", "Others"]
    assert summary.resolve_owner("myself", names, None) == "Me"
    assert summary.resolve_owner("sam", names, None) == "Sam Lee"
    assert summary.resolve_owner("Sam (IT)", names, None) == "Sam Lee"
    assert summary.resolve_owner("unknown", names, turns[0]) == "Sam Lee"
    assert summary.resolve_owner("Dana", names, None) == "Dana"


def test_generate_runs_on_one_llm_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_run", lambda *a: seen.append(threading.current_thread().name) or "ok")
    monkeypatch.setattr(llm, "_schedule_unload", lambda: None)
    out = []
    workers = [threading.Thread(target=lambda: out.append(llm.generate("hi", "x"))) for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert out == ["ok"] * 3
    assert len(set(seen)) == 1 and seen[0].startswith("listn-llm")


def test_chat_prompt_turns_thinking_off_when_supported():
    class Tok:
        def __init__(self, strict):
            self.strict, self.calls = strict, []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **extra):
            self.calls.append(extra)
            if self.strict and extra:
                raise TypeError("unexpected keyword")
            return f"{len(messages)}:{messages[-1]['content']}"

    loose, strict = Tok(False), Tok(True)
    assert llm.chat_prompt(loose, "hi", "sys") == "2:hi" and loose.calls == [{"enable_thinking": False}]
    assert llm.chat_prompt(strict, "hi") == "1:hi" and strict.calls[-1] == {}


def test_finish_notes_moves_vague_decisions():
    turns = build_turns([Segment(0, 1, "We agreed to discuss pricing later.", None)], {})
    out = finish_notes({"summary": "s", "decisions": [{"text": "Agreed to discuss pricing", "line": 1}]}, turns)
    assert out["decision_items"] == [] and out["open_questions"][0]["text"] == "Agreed to discuss pricing"


def test_reduce_step_merges_but_never_loses_candidates(data_dir, monkeypatch):
    monkeypatch.setattr(summary, "SINGLE_PASS_CHARS", 300)
    monkeypatch.setattr(summary, "CHUNK_CHARS", 250)
    lib = Library()
    meta = lib.create("Long", "live", speakers={"Me": "Me", "SPEAKER_00": "Ravi"})
    segs = [Segment(i * 10, i * 10 + 9, f"I'll fix the package script number {i % 3} today for the rollout.",
                    "SPEAKER_00" if i % 2 else "Me") for i in range(12)]
    lib.save_segments(meta.id, segs)

    def runner(prompt):
        if prompt.startswith("These are notes taken"):
            assert "A1 [" in prompt and "Candidate action items" in prompt
            return json.dumps({"summary": "Whole.", "key_points": [],
                               "decisions": [],
                               "actions": [{"ids": ["A2", "A1", "A99"], "owner": "ravi", "text": "Fix the package script",
                                            "due": "today"}]})
        lines = [int(n) for n in re.findall(r"^L(\d+) ", prompt, re.M)]
        return json.dumps({"summary": "Part.", "decisions": [{"text": "Fix the package script", "line": lines[0]}],
                           "actions": [{"owner": "Ravi", "text": f"Fix the package script number {lines[0] % 3}",
                                        "line": lines[0]} for _ in range(1)]})

    notes = build_notes(lib.segments(meta.id), meta.speakers, complete=runner)
    assert notes["chunks"] >= 2
    assert notes["decisions"] == ["Fix the package script"]  # an empty reply keeps the candidates
    action = notes["actions"][0]
    assert (action["owner"], action["text"], action["due"]) == ("Ravi", "Fix the package script", "today")
    assert action["start"] == 0
