from datetime import datetime

from listn.ipc import BOOKMARK, START, encode_json
from listn.meetings import meeting_label, pick_event, title_and_attendees

from .conftest import MAC_DEVICES, FakeSD
from .test_engine import wait_for
from .test_ingest import FakeClient


def test_meeting_detection_matches_calls_not_ordinary_browser_audio():
    assert meeting_label("us.zoom.xos") == "Zoom"
    assert meeting_label("com.microsoft.teams2") == "Teams"
    assert meeting_label("com.apple.FaceTime") == "FaceTime"
    assert meeting_label("com.tinyspeck.slackmacgap", window_titles=["Design huddle"]) == "Slack huddle"
    assert meeting_label("com.tinyspeck.slackmacgap", window_titles=["general"]) is None
    assert meeting_label("com.google.Chrome", window_titles=["Weekly — Google Meet"]) == "Browser meeting"
    assert meeting_label("com.google.Chrome", window_titles=["A song on YouTube"]) is None
    assert meeting_label("com.apple.Safari", app_name="meet.google.com") == "Browser meeting"


def test_calendar_picks_the_shortest_overlapping_event_and_titles_the_session():
    moment = datetime(2026, 10, 20, 15, 5)
    events = [
        {"title": "Out of office", "start": "2026-10-20T00:00:00", "end": "2026-10-21T00:00:00", "attendees": []},
        {"title": "Design review", "start": "2026-10-20T15:00:00", "end": "2026-10-20T15:30:00",
         "attendees": ["Priya", "Sam"]},
        {"title": "Tomorrow", "start": "2026-10-21T15:00:00", "end": "2026-10-21T16:00:00", "attendees": ["Ada"]},
    ]
    chosen = pick_event(events, moment)
    assert chosen["title"] == "Design review"
    title, people = title_and_attendees(None, events, moment)
    assert title == "Design review"
    assert people == ["Priya", "Sam"]
    kept, people = title_and_attendees("My title", events, moment)
    assert kept == "My title" and people == ["Priya", "Sam"]
    assert pick_event(events, datetime(2026, 10, 22, 9, 0)) is None


def test_bookmark_and_attendees_arrive_over_the_socket(data_dir, fake_factory):
    from listn.engine import Engine
    from listn.ingest import IngestServer

    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    server = IngestServer(engine, port=8765)
    server.start()
    try:
        client = FakeClient(server.path)
        client.hello()
        client.send(encode_json(START, {
            "source": "system", "mic": False, "model": "tiny", "chunk_seconds": 2,
            "title": "Design review", "attendees": ["Priya", "Sam"],
        }))
        wait_for(lambda: engine.live_active() and engine.live.meta.attendees == ["Priya", "Sam"])
        assert engine.live.meta.title == "Design review"
        client.send(encode_json(BOOKMARK, {"note": "Decision"}))
        wait_for(lambda: engine.library.bookmarks(engine.live.meta.id) == [{"t": engine.library.bookmarks(engine.live.meta.id)[0]["t"], "note": "Decision"}] if engine.library.bookmarks(engine.live.meta.id) else False)
        marks = engine.library.bookmarks(engine.live.meta.id)
        assert marks[0]["note"] == "Decision"
        assert marks[0]["t"] >= 0
        client.close()
    finally:
        server.stop()
