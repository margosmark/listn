"""Silent capture is flagged, late frames are dropped, versions agree."""

import json
import plistlib
import re
import time
import tomllib
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import listn
from listn.engine import Engine
from listn.ingest import IngestServer
from listn.ipc import EVENT, PING, PONG, START, STOP, encode_audio, encode_frame, encode_json
from listn.privacy import PANES, SILENT_SESSION, silence_message
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD, FakeTranscriber
from .test_engine import wait_for
from .test_ingest import FakeClient, ingest, tone  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


def _events_until(client, kind, timeout=10):
    seen = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        event = json.loads(client.wait_kind(EVENT, timeout=end - time.monotonic()))
        seen.append(event)
        if event["type"] == kind:
            return seen
    raise AssertionError(f"no {kind} event; saw {[e['type'] for e in seen]}")


def test_silence_message_points_at_the_right_pane():
    message, pane = silence_message("mic")
    assert pane == "microphone" and "Microphone" in message and "capture.log" in message
    message, pane = silence_message("system")
    assert pane == "audio-capture" and "System Audio Recording" in message and "capture.log" in message
    assert all(url.startswith("x-apple.systempreferences:") for url in PANES.values())


def test_all_zero_microphone_is_flagged_and_the_session_is_not_done(ingest):  # noqa: F811
    server, engine = ingest
    client = FakeClient(server.path)
    client.hello()
    client.send(encode_json(START, {"source": "system", "mic": True, "model": "tiny", "chunk_seconds": 2}))
    zeros = np.zeros(16000 * 3, dtype=np.float32)
    client.send(encode_audio(0, 16000, zeros))
    client.send(encode_audio(1, 16000, zeros))
    events = _events_until(client, "live.silent")
    silent = events[-1]
    assert silent["track"] == "mic" and silent["pane"] == "microphone"
    assert engine.live.status()["silent_tracks"] == ["mic"]

    client.send(encode_frame(STOP))
    _events_until(client, "live.stopped")
    meta = engine.library.get(engine.live.meta.id)
    assert meta.status == "error"
    assert meta.silent is True
    assert meta.error == SILENT_SESSION
    client.close()


class Mute(FakeTranscriber):
    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        self.calls += 1
        return iter(())


def test_audio_without_speech_is_done_with_a_warning(data_dir):
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=lambda model, backend, status=None: Mute(model))
    server = IngestServer(engine, port=8765)
    server.start()
    client = FakeClient(server.path)
    try:
        client.hello()
        client.send(encode_json(START, {"source": "system", "mic": True, "model": "tiny", "chunk_seconds": 2}))
        client.send(encode_audio(0, 16000, tone(2.6, 16000, 220)))
        client.send(encode_audio(1, 16000, tone(2.6, 16000, 330)))
        wait_for(lambda: engine.live_active() and len(engine.live._heard) == 2)
        session = engine.live
        client.send(encode_frame(STOP))
        _events_until(client, "live.stopped")
        meta = engine.library.get(session.meta.id)
        assert meta.status == "done"
        assert meta.silent is False
        assert meta.warning == "No speech was recognised in this recording."
    finally:
        client.close()
        server.stop()


def test_audio_after_stop_is_dropped_and_the_connection_survives(ingest):  # noqa: F811
    server, engine = ingest
    client = FakeClient(server.path)
    client.hello()
    client.send(encode_json(START, {"source": "system", "mic": False, "model": "tiny", "chunk_seconds": 2}))
    client.send(encode_audio(0, 16000, tone(2.2, 16000, 200)))
    wait_for(lambda: engine.live_active() and engine.live.segments)
    client.send(encode_frame(STOP))
    for _ in range(20):
        client.send(encode_audio(0, 16000, tone(0.1, 16000, 200)))
    _events_until(client, "live.stopped")
    for _ in range(5):
        client.send(encode_audio(0, 16000, tone(0.1, 16000, 200)))
    client.send(encode_frame(PING, b"still here"))
    assert client.wait_kind(PONG) == b"still here"
    assert server.late_frames > 0
    client.close()


def test_privacy_endpoint_rejects_unknown_panes(data_dir, fake_factory, monkeypatch):
    import listn.privacy as privacy

    opened = []
    monkeypatch.setattr(privacy, "open_pane", lambda pane: opened.append(pane) or True)
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    app = create_app(engine=Engine(sd=sd, transcriber_factory=fake_factory), sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        assert c.post("/api/privacy/open", json={"pane": "camera"}).status_code == 400
        r = c.post("/api/privacy/open", json={"pane": "audio-capture"})
        assert r.status_code == 200
        assert r.json()["url"] == PANES["audio-capture"]
    assert opened == ["audio-capture"]


def test_app_engine_and_package_versions_agree():
    plist = plistlib.loads((ROOT / "macos" / "Info.plist").read_bytes())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert plist["CFBundleShortVersionString"] == project == listn.__version__
    lock = (ROOT / "uv.lock").read_text()
    assert re.search(r'name = "listn"\nversion = "' + re.escape(project) + '"', lock)
