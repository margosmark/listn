"""Capture-socket sessions driven by a fake menu-bar client."""

import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from listn.config import default_model
from listn.engine import Engine
from listn.ingest import IngestServer, path_fits_unix_socket, resolve_capture_socket, unix_socket_max_bytes
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


def test_default_model_is_small_without_mlx(monkeypatch):
    """Apple Silicon with MLX installed must still be able to prove the fallback."""
    import builtins

    import listn.config as config

    monkeypatch.setattr(config, "apple_silicon", lambda: True)
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx_whisper" or name.startswith("mlx_whisper."):
            raise ImportError("mlx unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert config.default_model() == "small"


def test_default_model_is_turbo_when_mlx_imports(monkeypatch):
    import sys
    import types

    import listn.config as config

    monkeypatch.setattr(config, "apple_silicon", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.ModuleType("mlx_whisper"))
    assert config.default_model() == "large-v3-turbo"


def test_default_model_is_small_off_apple_silicon(monkeypatch):
    import listn.config as config

    monkeypatch.setattr(config, "apple_silicon", lambda: False)
    assert config.default_model() == "small"


def test_unix_socket_limit_matches_the_platform():
    if sys.platform == "darwin":
        assert unix_socket_max_bytes() <= 103
    else:
        assert unix_socket_max_bytes() == 107
    macos_sized = Path("/tmp") / ("p" * 100)  # "/tmp/" + 100 == 105 bytes
    assert len(os.fsencode(macos_sized)) == 105
    if sys.platform == "darwin":
        assert not path_fits_unix_socket(macos_sized)
    else:
        assert path_fits_unix_socket(macos_sized)
    assert not path_fits_unix_socket(Path("/tmp") / ("p" * 120))


def test_long_data_dir_falls_back_to_a_short_socket(monkeypatch, fake_factory):
    import shutil

    long_root = Path("/tmp") / ("n" * 90)
    data = long_root / "data"
    monkeypatch.setenv("LISTN_DATA_DIR", str(data))
    monkeypatch.setenv("TMPDIR", "/tmp/" + ("t" * 90))
    preferred = data / "capture.sock"
    assert not path_fits_unix_socket(preferred)
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    server = IngestServer(engine, port=8765)
    try:
        server.start()
        assert server.path == Path(f"/tmp/listn-{os.getuid()}") / "capture.sock"
        assert path_fits_unix_socket(server.path)
        assert server.path.is_socket()
        assert os.stat(server.path).st_mode & 0o777 == 0o600
        assert server.path.parent.stat().st_mode & 0o777 == 0o700
        info = json.loads((data / "server.json").read_text())
        assert info["sock"] == str(server.path)
        assert info["pid"] == os.getpid()
        client = FakeClient(server.path)
        client.hello()
        client.close()
    finally:
        server.stop()
        shutil.rmtree(long_root, ignore_errors=True)


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
    try:
        client.hello()
        client.send(encode_json(START, {
            "source": "app",
            "app_name": "Zoom",
            "mic": True,
            "title": "Standup",
            "model": "tiny",
            "chunk_seconds": 2,
        }))
        wait_for(lambda: engine.live_active())
        live = engine.live
        client.send(encode_audio(0, 48000, tone(2.6, 48000, 220)))
        client.send(encode_audio(1, 16000, tone(2.6, 16000, 330)))
        # AUDIO and STOP share one stream, so the server cannot stop
        # before both tones are queued. Joining the workers waits until
        # the flush and the labels exist. A socket read is not that
        # signal: the event bus drops frames once a subscriber is full,
        # which happened under a full-suite load.
        client.send(encode_frame(STOP))
        assert live is not None
        for thread in live._threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), thread.name
        with live._segment_lock:
            segments = list(live.segments)
        assert {segment.speaker for segment in segments} >= {"Me", "Others"}
        assert all("Thanks for watching" not in segment.text for segment in segments)
        while True:
            try:
                item = client.frames.get_nowait()
            except queue.Empty:
                break
            if not isinstance(item, tuple):
                continue
            kind, payload = item
            if kind != EVENT:
                continue
            event = json.loads(payload)
            if event["type"] != "live.segments":
                continue
            assert all("Thanks for watching" not in segment["text"] for segment in event["segments"])

        wait_for(lambda: engine.live is not None and not engine.live_active(), timeout=30)

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
    finally:
        client.close()
        if engine.live_active():
            engine.stop_live(wait=True)


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
