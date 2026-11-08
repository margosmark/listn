"""Captions: the rules that make a live transcript safe to put on screen.

* :class:`CaptionModel` — pure: which lines show, and when the window
  opens, hides, clears and closes. Only the active session can contribute;
  the ``[paused]`` gap is never rendered as speech.
* Privacy — caption text never reaches a log record, the module contains
  no clipboard call, and the event stream is connected only while the
  setting is on.
* The supervisor — a real thread, a fake window and a fake stream, so the
  subscribe/unsubscribe and close-on-stop behaviour is exercised without a
  display.
* :class:`EventStreamClient` against the real server: the captions come
  from the engine's own ``/api/events`` stream over loopback, proven with
  a running uvicorn rather than a mock.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

import pytest

from listn.captions import (CLOSE, HIDE, OPEN, SHOW, UPDATE,
                            CaptionModel, CaptionSupervisor,
                            EventStreamClient)


def wait_for(predicate, timeout=5.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def started(session="s1"):
    return {"type": "live.started", "session": session}


def segments(*texts, session="s1", speaker=None):
    return {"type": "live.segments", "session": session,
            "segments": [{"start": 0.0, "end": 1.0, "text": t,
                          "speaker": speaker} for t in texts]}


def test_opens_on_start_and_shows_lines():
    model = CaptionModel()
    assert model.handle(started()) == OPEN
    assert model.handle(segments("hello there")) == UPDATE
    assert model.lines == ["hello there"]


def test_speaker_prefix_is_only_added_when_present():
    model = CaptionModel()
    model.handle(started())
    model.handle(segments("plain"))
    model.handle(segments("with a name", speaker="Me"))
    assert model.lines == ["plain", "Me: with a name"]


def test_pause_clears_and_hides_resume_shows_again():
    model = CaptionModel()
    model.handle(started())
    model.handle(segments("something said"))
    assert model.handle({"type": "live.paused", "session": "s1"}) == HIDE
    assert model.lines == []
    assert model.handle({"type": "live.resumed", "session": "s1"}) == SHOW
    assert model.lines == []


def test_stop_clears_and_closes():
    model = CaptionModel()
    model.handle(started())
    model.handle(segments("secret words"))
    assert model.handle({"type": "live.stopped", "session": "s1"}) == CLOSE
    assert model.lines == []
    assert model.running is False
    # Nothing after the stop can reopen or repopulate it.
    assert model.handle(segments("late line")) is None
    assert model.lines == []


def test_the_pause_gap_is_never_rendered_as_speech():
    model = CaptionModel()
    model.handle(started())
    assert model.handle(segments("real words")) == UPDATE
    assert model.handle(segments("[paused]")) is None
    assert model.lines == ["real words"]


def test_only_the_active_session_can_write_captions():
    model = CaptionModel()
    model.handle(started("s1"))
    assert model.handle(segments("from s1")) == UPDATE
    assert model.handle(segments("from another session", session="s2")) is None
    assert model.handle({"type": "live.paused", "session": "s2"}) is None
    assert model.lines == ["from s1"]


def test_lines_are_bounded():
    model = CaptionModel(max_lines=3)
    model.handle(started())
    model.handle(segments("a", "b", "c", "d", "e"))
    assert model.lines == ["c", "d", "e"]


def test_events_before_a_start_are_ignored():
    model = CaptionModel()
    assert model.handle(segments("should not appear")) is None
    assert model.lines == []


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def test_no_caption_text_reaches_a_log_record(caplog):
    model = CaptionModel()
    model.handle(started())
    # A stream failure *while* captions are on is the one place this module
    # logs at all: it must report the error type, never the text.
    client = EventStreamClient(8765)

    def boom(url, timeout=None):
        client.stop()
        raise OSError("connection refused")

    client._opener = boom
    with caplog.at_level(logging.DEBUG):
        model.handle(segments("the merger closes on friday"))
        model.handle({"type": "live.paused", "session": "s1"})
        model.handle({"type": "live.stopped", "session": "s1"})
        assert list(client.events()) == []
    assert caplog.records
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "merger" not in blob
    assert "friday" not in blob
    assert any("reconnect" in r.getMessage() for r in caplog.records)


def test_the_module_has_no_clipboard_path():
    import listn.captions as captions_module

    source = open(captions_module.__file__).read()
    # The Tk clipboard API (and every CLI copier) must be absent, not just
    # unused: nothing in this module can put caption text on the clipboard.
    for forbidden in ("clipboard_append", "clipboard_get", "clipboard_clear",
                      "tkinter.clipboard", "xclip", "xsel", "wl-copy",
                      "pyperclip"):
        assert forbidden not in source, forbidden


class FakeResponse:
    """A loopback SSE body: bytes lines, entered like urlopen()."""

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_caption_text_never_reaches_the_stream_log(caplog):
    body = [b"retry: 2000\n\n",
            b'data: {"type": "live.segments", "session": "s1", '
            b'"segments": [{"text": "board approves the merger", "speaker": null}]}\n\n']
    client = EventStreamClient(8765)
    client._opener = lambda url, timeout=None: FakeResponse(body)
    with caplog.at_level(logging.DEBUG):
        got = []
        for event in client.events():
            got.append(event)
            break                      # closes the generator: no reconnect
        client.stop()
    assert got and got[0]["type"] == "live.segments"
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "merger" not in blob and "approves" not in blob


# ---------------------------------------------------------------------------
# Supervisor: opt-in subscription, close on stop/disable
# ---------------------------------------------------------------------------

class FakeWindow:
    def __init__(self):
        self.actions: list = []
        self.closed = False

    def apply(self, action, lines):
        self.actions.append((action, list(lines)))
        if action == CLOSE:
            self.closed = True

    def close(self):
        self.closed = True
        if (CLOSE, []) not in self.actions:
            self.actions.append((CLOSE, []))


class FakeStream:
    def __init__(self):
        self.inbox: "queue.Queue" = queue.Queue()
        self.stopped = False
        self.started = False

    def events(self):
        self.started = True
        while not self.stopped:
            try:
                item = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item

    def stop(self):
        self.stopped = True
        self.inbox.put(None)


def test_captions_stay_unsubscribed_while_the_setting_is_off():
    settings = {"captions_enabled": False}
    streams: list = []
    window = FakeWindow()

    def factory():
        stream = FakeStream()
        streams.append(stream)
        return stream

    supervisor = CaptionSupervisor(
        port=8765, get_settings=lambda: dict(settings),
        window_factory=lambda: window, stream_factory=factory,
        poll_seconds=0.02, live_getter=lambda: {},
        available=lambda: (True, ""))
    supervisor.start()
    try:
        assert wait_for(lambda: supervisor._window is not None)
        time.sleep(0.15)
        assert streams == []           # never connected: captions are off
        assert window.actions == []
    finally:
        supervisor.stop()
    assert window.closed


def test_enabling_subscribes_and_stop_closes_the_window():
    settings = {"captions_enabled": True}
    streams: list = []
    window = FakeWindow()

    def factory():
        stream = FakeStream()
        streams.append(stream)
        return stream

    supervisor = CaptionSupervisor(
        port=8765, get_settings=lambda: dict(settings),
        window_factory=lambda: window, stream_factory=factory,
        poll_seconds=0.02, live_getter=lambda: {},
        available=lambda: (True, ""))
    supervisor.start()
    try:
        assert wait_for(lambda: len(streams) == 1)
        stream = streams[0]
        stream.inbox.put(started())
        stream.inbox.put(segments("local only"))
        assert wait_for(lambda: (OPEN, []) in window.actions)
        assert wait_for(lambda: any(a == UPDATE and "local only" in lines
                                    for a, lines in window.actions))
        stream.inbox.put({"type": "live.stopped", "session": "s1"})
        assert wait_for(lambda: window.closed)
        assert supervisor.model.lines == []
    finally:
        supervisor.stop()


def test_switching_captions_off_disconnects_and_clears():
    settings = {"captions_enabled": True}
    streams: list = []
    window = FakeWindow()

    def factory():
        stream = FakeStream()
        streams.append(stream)
        return stream

    supervisor = CaptionSupervisor(
        port=8765, get_settings=lambda: dict(settings),
        window_factory=lambda: window, stream_factory=factory,
        poll_seconds=0.02, live_getter=lambda: {},
        available=lambda: (True, ""))
    supervisor.start()
    try:
        assert wait_for(lambda: len(streams) == 1)
        streams[0].inbox.put(started())
        streams[0].inbox.put(segments("visible until switched off"))
        assert wait_for(lambda: supervisor.model.running)
        settings["captions_enabled"] = False
        assert wait_for(lambda: window.closed)
        assert wait_for(lambda: streams[0].stopped)
        assert supervisor.model.lines == []
    finally:
        supervisor.stop()


def test_no_display_is_reported_instead_of_failing():
    window = FakeWindow()
    supervisor = CaptionSupervisor(
        port=8765, get_settings=lambda: {"captions_enabled": True},
        window_factory=lambda: window,
        stream_factory=lambda: FakeStream(),
        poll_seconds=0.02, live_getter=lambda: {},
        available=lambda: (False, "no display is available to this process"))
    supervisor.start()
    try:
        assert wait_for(lambda: supervisor.disabled_reason is not None)
        assert "no display" in supervisor.disabled_reason
        assert window.actions == []      # nothing was opened
    finally:
        supervisor.stop()


def test_snapshot_reopens_captions_for_a_running_recording():
    settings = {"captions_enabled": True}
    streams: list = []
    window = FakeWindow()

    def factory():
        stream = FakeStream()
        streams.append(stream)
        return stream

    live = {"live": {"id": "s9", "state": "listening", "paused": False},
            "segments": [{"start": 0.0, "end": 1.0, "text": "already said",
                          "speaker": None}]}
    supervisor = CaptionSupervisor(
        port=8765, get_settings=lambda: dict(settings),
        window_factory=lambda: window, stream_factory=factory,
        poll_seconds=0.02, live_getter=lambda: live,
        available=lambda: (True, ""))
    supervisor.start()
    try:
        assert wait_for(lambda: any(a == OPEN for a, _ in window.actions))
        assert wait_for(lambda: any("already said" in lines
                                    for _, lines in window.actions))
    finally:
        supervisor.stop()


# ---------------------------------------------------------------------------
# The real event stream over loopback
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server(data_dir, fake_factory):
    import uvicorn
    from fastapi.testclient import TestClient  # noqa: F401  (ensures deps)

    from listn.engine import Engine
    from listn.server import create_app
    from .conftest import MAC_DEVICES, FakeSD

    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    app.state.http_port = port
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_for(lambda: server.started, timeout=10.0), "server did not start"
    yield port, engine
    server.should_exit = True
    thread.join(timeout=10.0)


def test_captions_consume_the_engines_own_event_stream(live_server):
    port, engine = live_server
    client = EventStreamClient(port, timeout=5.0)
    got: list = []
    done = threading.Event()

    def read():
        for event in client.events():
            got.append(event)
            if event.get("type") == "live.stopped":
                done.set()
                return

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    # Wait for the SSE subscription before publishing: the stream is a real
    # HTTP connection that takes a moment to attach.
    assert wait_for(lambda: len(engine.bus._subs) > 0, timeout=10.0)
    engine.bus.publish({"type": "live.started", "session": "s1"})
    engine.bus.publish({"type": "live.segments", "session": "s1",
                        "segments": [{"start": 0.0, "end": 1.0,
                                      "text": "through the real server",
                                      "speaker": None}]})
    engine.bus.publish({"type": "live.stopped", "session": "s1"})
    assert done.wait(10.0), f"stream did not deliver: {got}"
    client.stop()
    kinds = [e.get("type") for e in got]
    assert "live.started" in kinds and "live.segments" in kinds
    text = json.dumps(got)
    assert "through the real server" in text
    thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# The Tk window (needs a display; skipped on a headless box)
# ---------------------------------------------------------------------------

def test_caption_window_shows_and_clears_text():
    from listn.captions import CaptionWindow

    usable, reason = CaptionWindow.available()
    if not usable:
        pytest.skip(f"no caption window here: {reason}")
    window = CaptionWindow.create()
    try:
        window.apply(OPEN, ["first line", "second line"])
        assert "first line" in window.text.get("1.0", "end")
        assert window.visible
        window.apply(HIDE, [])
        assert not window.visible
        window.apply(UPDATE, ["only this"])
        body = window.text.get("1.0", "end")
        assert "only this" in body and "first line" not in body
        window.apply(CLOSE, [])
        assert window.destroyed
        assert not window.visible
        # close() empties the widget and then destroys the window: reading
        # it afterwards must fail, which is the point — the text is gone
        # with the window, not left behind somewhere.
        import tkinter

        with pytest.raises(tkinter.TclError):
            window.text.get("1.0", "end")
    finally:
        window.close()
