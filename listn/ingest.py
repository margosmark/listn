"""Unix-socket ingest for the macOS menu bar app.

The socket lives at ``<data dir>/capture.sock`` with mode 0600. The data
directory itself is 0700, so other users on the machine cannot connect.
If that path is longer than a Unix socket address allows (104 bytes on
macOS, including the NUL), the socket moves to a short per-user directory
such as ``/tmp/listn-<uid>/`` (mode 0700). ``server.json`` stays in the data
directory and records the path that is actually listening. Nothing listens
on a network interface.

One client at a time. A new connection replaces the old one without stopping
the recording, so the menu bar can reconnect after a hiccup and still send
STOP. Audio from the client is pushed into the live session; transcript
events are written back as EVENT frames.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import struct
import sys
import threading
from pathlib import Path

from listn import __version__
from listn.config import Settings, data_dir
from listn.engine import Engine
from listn.ipc import (
    AUDIO,
    ERROR,
    EVENT,
    HELLO,
    MAX_FRAME,
    PING,
    PONG,
    PROTOCOL_VERSION,
    START,
    STOP,
    TRACK_SPEAKER,
    WELCOME,
    FrameDecoder,
    ProtocolError,
    decode_audio,
    decode_json,
    encode_frame,
    encode_json,
)

log = logging.getLogger("listn")

# CPython rejects an AF_UNIX path whose length is >= sizeof(sun_path).
# macOS sun_path is 104 bytes; Linux is 108. The bound we use is one
# shorter than that so the terminating NUL fits on both.
_UNIX_SOCKET_MAX = 103 if sys.platform == "darwin" else 107


def unix_socket_max_bytes() -> int:
    return _UNIX_SOCKET_MAX


def path_fits_unix_socket(path: Path) -> bool:
    try:
        encoded = os.fsencode(path)
    except (OSError, ValueError):
        return False
    return 0 < len(encoded) <= _UNIX_SOCKET_MAX


def short_socket_dir() -> Path:
    """A 0700 directory whose ``capture.sock`` fits in a Unix address.

    ``$TMPDIR`` is preferred when that path is short enough. macOS points
    ``TMPDIR`` at ``/var/folders/...``, which is often already too long, so
    ``/tmp/listn-<uid>`` is the fallback that always fits.
    """
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    name = f"listn-{uid}"
    candidates: list[Path] = []
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        candidates.append(Path(tmpdir) / name)
    fallback = Path("/tmp") / name
    if fallback not in candidates:
        candidates.append(fallback)
    chosen = next((directory for directory in candidates if path_fits_unix_socket(directory / "capture.sock")), fallback)
    chosen.mkdir(parents=True, exist_ok=True)
    try:
        chosen.chmod(0o700)
    except OSError:
        pass
    return chosen


def resolve_capture_socket(preferred: Path | None = None) -> Path:
    """Return ``preferred`` when it can be bound, otherwise a short per-user path."""
    path = Path(preferred) if preferred is not None else data_dir() / "capture.sock"
    if path_fits_unix_socket(path):
        return path
    fallback = short_socket_dir() / "capture.sock"
    try:
        length = len(os.fsencode(path))
    except (OSError, ValueError):
        length = -1
    log.warning(
        "Capture socket path is too long for a Unix socket (%s, %s bytes). Using %s",
        path,
        length,
        fallback,
    )
    return fallback


def server_info_path() -> Path:
    return data_dir() / "server.json"


def write_server_info(port: int | None, sock: Path | None) -> None:
    path = server_info_path()
    payload = {"pid": os.getpid(), "port": port, "sock": None if sock is None else str(sock), "version": __version__}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.chmod(0o600)
    tmp.replace(path)


def clear_server_info() -> None:
    path = server_info_path()
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if info.get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def _same_user(conn: socket.socket) -> bool:
    """True when the peer is this user, or when the OS cannot say (macOS)."""
    try:
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    except (OSError, AttributeError):
        return True
    _pid, uid, _gid = struct.unpack("3i", raw)
    return uid == os.getuid()


class _Client:
    def __init__(self):
        self.replaced = False
        self.owns_session = False
        self.done = threading.Event()


class IngestServer:
    def __init__(self, engine: Engine, path: Path | None = None, port: int | None = None):
        self.engine = engine
        self.path = Path(path) if path else None
        self.port = port
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._conn: socket.socket | None = None
        self._client: _Client | None = None
        self._handlers: list[threading.Thread] = []

    def start(self) -> None:
        if self.path is None:
            self.path = data_dir() / "capture.sock"
        self.path = resolve_capture_socket(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._reclaim_stale_socket()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(self.path))
        except OSError:
            srv.close()
            raise
        os.chmod(self.path, 0o600)
        srv.listen(1)
        srv.settimeout(0.5)
        self._server = srv
        write_server_info(self.port, self.path)
        self._accept_thread = threading.Thread(target=self._accept_loop, name="listn-ingest", daemon=True)
        self._accept_thread.start()
        log.info("Capture socket listening at %s", self.path)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        with self._lock:
            conn = self._conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        for thread in list(self._handlers):
            thread.join(timeout=2)
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
        if self.path is not None:
            try:
                if self.path.exists():
                    self.path.unlink()
            except OSError:
                pass
        clear_server_info()

    def _reclaim_stale_socket(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self.path))
        except OSError:
            try:
                self.path.unlink()
            except OSError:
                pass
        else:
            raise OSError(f"Capture socket {self.path} is already served by another listn.")
        finally:
            probe.close()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not _same_user(conn):
                conn.close()
                continue
            state = _Client()
            with self._lock:
                previous = self._conn
                previous_state = self._client
                self._conn = conn
                self._client = state
            if previous_state is not None:
                previous_state.replaced = True
                if previous_state.owns_session:
                    state.owns_session = True
                    previous_state.owns_session = False
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            thread = threading.Thread(target=self._handle, args=(conn, state), name="listn-capture-client", daemon=True)
            self._handlers.append(thread)
            thread.start()

    def _send(self, conn: socket.socket, frame: bytes) -> None:
        with self._send_lock:
            conn.sendall(frame)

    def _handle(self, conn: socket.socket, state: _Client) -> None:
        decoder = FrameDecoder()
        events: queue.Queue | None = None
        forwarder: threading.Thread | None = None
        conn.settimeout(0.5)
        hello = False
        try:
            while not self._stop.is_set() and not state.done.is_set():
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                try:
                    frames = decoder.feed(data)
                except ProtocolError as exc:
                    self._error(conn, str(exc))
                    break
                for kind, payload in frames:
                    if not hello and kind != HELLO:
                        self._error(conn, "Send HELLO before any other frame.")
                        return
                    try:
                        if kind == HELLO:
                            hello = self._on_hello(conn, payload)
                            if hello and events is None:
                                events = self.engine.bus.subscribe()
                                forwarder = threading.Thread(
                                    target=self._forward, args=(conn, state, events),
                                    name="listn-capture-events", daemon=True,
                                )
                                forwarder.start()
                            if not hello:
                                return
                        elif kind == START:
                            self._on_start(conn, state, payload)
                        elif kind == STOP:
                            self._on_stop(state)
                        elif kind == AUDIO:
                            self._on_audio(conn, payload)
                        elif kind == PING:
                            self._send(conn, encode_frame(PONG, payload))
                        elif kind == PONG:
                            pass
                        else:
                            self._error(conn, f"Unknown frame type {kind}.")
                    except ProtocolError as exc:
                        self._error(conn, str(exc))
                    except OSError:
                        return
        finally:
            state.done.set()
            if forwarder is not None:
                forwarder.join(timeout=1)
            if events is not None:
                self.engine.bus.unsubscribe(events)
            if state.owns_session and not state.replaced and self.engine.live_active():
                try:
                    self.engine.stop_live(wait=False)
                except Exception:
                    log.exception("Could not stop the session the capture client owned")
            try:
                conn.close()
            except OSError:
                pass

    def _forward(self, conn: socket.socket, state: _Client, events: queue.Queue) -> None:
        while not state.done.is_set() and not self._stop.is_set():
            try:
                event = events.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._send(conn, encode_json(EVENT, event))
            except OSError:
                break

    def _error(self, conn: socket.socket, message: str) -> None:
        try:
            self._send(conn, encode_json(ERROR, {"message": message}))
        except OSError:
            pass

    def _on_hello(self, conn: socket.socket, payload: bytes) -> bool:
        body = decode_json(payload)
        version = int(body.get("version") or 0)
        if version != PROTOCOL_VERSION:
            self._error(conn, f"Protocol version {version} is not supported (this engine speaks {PROTOCOL_VERSION}).")
            return False
        self._send(conn, encode_json(WELCOME, {
            "version": PROTOCOL_VERSION,
            "server": "listn",
            "pid": os.getpid(),
            "recording": self.engine.live_active(),
            "tracks": {str(k): v for k, v in TRACK_SPEAKER.items()},
            "max_frame": MAX_FRAME,
        }))
        return True

    def _settings(self, body: dict) -> Settings:
        settings = Settings.load()
        for key in ("model", "language", "backend", "device"):
            if body.get(key):
                setattr(settings, key, body[key])
        if body.get("chunk_seconds") is not None:
            settings.chunk_seconds = body["chunk_seconds"]
        if "save_audio" in body and body["save_audio"] is not None:
            settings.save_audio = bool(body["save_audio"])
        return settings.validated()

    def _on_start(self, conn: socket.socket, state: _Client, payload: bytes) -> None:
        body = decode_json(payload)
        source = str(body.get("source") or "system")
        if source not in ("system", "app", "device"):
            raise ProtocolError("source must be system, app, or device.")
        settings = self._settings(body)
        title = body.get("title") or None
        if isinstance(title, str):
            title = title.strip()[:200] or None
        try:
            if source == "device":
                self.engine.start_live(settings, title)
            else:
                self.engine.start_ingest(settings, {
                    "source": source,
                    "app_name": body.get("app_name") or None,
                    "mic": bool(body.get("mic", True)),
                }, title)
        except RuntimeError as exc:
            self._error(conn, str(exc))
            return
        state.owns_session = True

    def _on_stop(self, state: _Client) -> None:
        if not state.owns_session:
            raise ProtocolError("This client did not start the recording.")
        if self.engine.live is None:
            raise ProtocolError("Nothing is recording.")
        self.engine.stop_live(wait=False)
        state.owns_session = False

    def _on_audio(self, conn: socket.socket, payload: bytes) -> None:
        session = self.engine.live
        if session is None or not self.engine.live_active():
            self._error(conn, "Not recording.")
            return
        if session.mode != "ingest":
            self._error(conn, "This recording is using an input device, not the capture socket.")
            return
        track, rate, _channels, samples = decode_audio(payload)
        if track not in TRACK_SPEAKER:
            self._error(conn, "Track must be 0 (system) or 1 (microphone).")
            return
        session.push_audio(track, rate, samples)
