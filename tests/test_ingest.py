"""Capture-socket sessions driven by a fake menu-bar client."""

import json
import os
import queue
import socket
import struct
import threading
import time
import wave

import numpy as np
import pytest

from listn.config import default_model
from listn.engine import Engine
from listn.ingest import IngestServer
from listn.ipc import (
    ERROR,
    EVENT,
    HELLO,
    PING,
    PONG,
    START,
    STOP,
    WELCOME,
    FrameDecoder,
    ProtocolError,
    encode_audio,
    encode_frame,
    encode_json,
)

from .conftest import MAC_DEVICES, FakeSD
from .test_engine import wait_for


class FakeClient:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(path))
        self.dec = FrameDecoder()
        self.frames: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        self.sock.settimeout(0.3)
        while not self._stop.is_set():
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                self.frames.put(None)
                return
            if not data:
                self.frames.put(None)
                return
            try:
                for frame in self.dec.feed(data):
                    self.frames.put(frame)
            except ProtocolError as exc:
                self.frames.put(exc)
                return

    def send(self, frame: bytes) -> None:
        self.sock.sendall(frame)

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

    def next_frame(self, timeout=5):
        frame = self.frames.get(timeout=timeout)
        if isinstance(frame, Exception):
            raise frame
        if frame is None:
            raise EOFError("capture socket closed")
        return frame

    def wait_kind(self, kind, timeout=5):
        end = time.monotonic() + timeout
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out waiting for frame {kind}")
            got, payload = self.next_frame(remaining)
            if got == kind:
                return payload

    def hello(self):
        self.send(encode_json(HELLO, {"client": "fake", "version": 1}))
        payload = self.wait_kind(WELCOME)
        return json.loads(payload)


def tone(seconds, rate, freq):
    t = np.arange(int(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def ingest(data_dir, fake_factory):
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    server = IngestServer(engine, port=8765)
    server.start()
    try:
        yield server, engine
    finally:
        server.stop()


def test_default_model_is_small_without_mlx():
    # This environment is not an Apple Silicon Mac with mlx-whisper.
    assert default_model() == "small"


def test_socket_is_private_and_handshake_works(ingest, data_dir):
    server, _engine = ingest
    mode = os.stat(server.path).st_mode & 0o777
    assert mode == 0o600
    info = json.loads((data_dir / "server.json").read_text())
    assert info["sock"] == str(server.path)
    assert info["port"] == 8765
    assert info["pid"] == os.getpid()
    client = FakeClient(server.path)
    welcome = client.hello()
    assert welcome["recording"] is False
    assert welcome["tracks"] == {"0": "Others", "1": "Me"}
    client.send(encode_frame(PING, b"ping"))
    assert client.wait_kind(PONG) == b"ping"
    client.close()


def test_commands_before_hello_are_rejected(ingest):
    server, _engine = ingest
    client = FakeClient(server.path)
    client.send(encode_frame(PING, b"x"))
    message = json.loads(client.wait_kind(ERROR))["message"]
    assert "HELLO" in message
    client.close()


def test_absurd_frame_length_is_rejected(ingest):
    server, _engine = ingest
    client = FakeClient(server.path)
    client.hello()
    client.send(struct.pack("<I", 50_000_000))
    message = json.loads(client.wait_kind(ERROR))["message"]
    assert "length" in message.lower()
    client.close()


def test_dual_track_session_labels_me_and_others(ingest):
    server, engine = ingest
    client = FakeClient(server.path)
    client.hello()
    client.send(encode_json(START, {
        "source": "app",
        "app_name": "Zoom",
        "mic": True,
        "title": "Standup",
        "model": "tiny",
        "chunk_seconds": 2,
    }))
    client.send(encode_audio(0, 48000, tone(2.6, 48000, 220)))
    client.send(encode_audio(1, 16000, tone(2.6, 16000, 330)))

    speakers = set()
    end = time.monotonic() + 10
    while time.monotonic() < end and speakers != {"Me", "Others"}:
        payload = client.wait_kind(EVENT, timeout=end - time.monotonic())
        event = json.loads(payload)
        if event["type"] == "live.segments":
            speakers.update(seg["speaker"] for seg in event["segments"])
            assert all("Thanks for watching" not in seg["text"] for seg in event["segments"])
    assert speakers == {"Me", "Others"}

    client.send(encode_frame(STOP))
    end = time.monotonic() + 10
    while time.monotonic() < end:
        event = json.loads(client.wait_kind(EVENT, timeout=end - time.monotonic()))
        if event["type"] == "live.stopped":
            break
    else:
        raise AssertionError("session did not stop")

    meta = engine.library.get(engine.live.meta.id)
    assert meta.status == "done"
    assert meta.source_name == "Zoom + microphone"
    assert meta.duration > 2
    segs = engine.library.segments(meta.id)
    assert {s.speaker for s in segs} == {"Me", "Others"}
    assert segs == sorted(segs, key=lambda s: (s.start, s.speaker or ""))
    root = engine.library.session_path(meta.id)
    for name in ("audio-system.wav", "audio-mic.wav", "audio.wav"):
        assert (root / name).exists()
    with wave.open(str(root / "audio.wav"), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getframerate() == 16000
        assert wf.getnframes() > 16000
    client.close()


def test_replacing_the_client_does_not_drop_the_recording(ingest):
    server, engine = ingest
    first = FakeClient(server.path)
    first.hello()
    first.send(encode_json(START, {"source": "system", "mic": False, "model": "tiny", "chunk_seconds": 2}))
    first.send(encode_audio(0, 16000, tone(2.4, 16000, 180)))
    wait_for(lambda: engine.live_active() and engine.live.segments)

    second = FakeClient(server.path)
    welcome = second.hello()
    assert welcome["recording"] is True
    assert engine.live_active()
    second.send(encode_frame(STOP))
    end = time.monotonic() + 10
    while time.monotonic() < end:
        event = json.loads(second.wait_kind(EVENT, timeout=end - time.monotonic()))
        if event["type"] == "live.stopped":
            break
    else:
        raise AssertionError("replacement client could not stop the session")
    assert not engine.live_active()
    first.close()
    second.close()


def test_legacy_device_start_uses_the_input_device(ingest):
    server, engine = ingest
    client = FakeClient(server.path)
    client.hello()
    client.send(encode_json(START, {
        "source": "device",
        "device": "BlackHole",
        "model": "tiny",
        "chunk_seconds": 2,
        "title": "Legacy",
    }))
    saw_segment = False
    end = time.monotonic() + 10
    while time.monotonic() < end and not saw_segment:
        event = json.loads(client.wait_kind(EVENT, timeout=end - time.monotonic()))
        if event["type"] == "live.segments":
            saw_segment = True
            assert all(seg["speaker"] is None for seg in event["segments"])
        elif event["type"] == "live.error":
            raise AssertionError(event["message"])
    assert saw_segment
    assert engine.live.mode == "device"
    assert engine.live.device_info["name"] == "BlackHole 2ch"
    client.send(encode_audio(1, 16000, tone(0.2, 16000, 440)))
    message = json.loads(client.wait_kind(ERROR))["message"]
    assert "input device" in message
    client.send(encode_frame(STOP))
    end = time.monotonic() + 10
    while time.monotonic() < end:
        event = json.loads(client.wait_kind(EVENT, timeout=end - time.monotonic()))
        if event["type"] == "live.stopped":
            break
    else:
        raise AssertionError("legacy session did not stop")
    client.close()
