"""Binary frames for the menu bar app's capture socket.

The socket is a filesystem Unix domain socket (see ``listn.ingest``). Every
frame is little-endian, which is the native byte order on Apple Silicon::

    uint32 length          byte count of (type + payload), 1 .. 8 MiB
    uint8  type
    bytes  payload

Types
-----
1 HELLO    client → server. JSON object, ``{"client": "listn-menu", "version": 1}``.
2 WELCOME  server → client. JSON object, ``version``, ``pid``, ``recording``,
           ``tracks`` mapping ``"0"`` to ``"Others"`` and ``"1"`` to ``"Me"``.
3 AUDIO    client → server. Captured PCM::

               uint8  track          0 system (others), 1 microphone (me)
               uint32 sample_rate    Hz, greater than 0
               uint16 channels       1 .. 8
               float32 PCM           interleaved, length divisible by channels

4 START    client → server. JSON. ``source`` is ``"system"``, ``"app"``, or
           ``"device"`` (legacy input, including BlackHole). Optional:
           ``app_name``, ``app_pid``, ``mic`` (bool), ``title``, plus any of
           ``model``, ``language``, ``backend``, ``chunk_seconds``, ``device``,
           ``save_audio`` applied to this session only.
5 STOP     client → server. Empty payload. Ends the session this client started.
6 EVENT    server → client. JSON object, the same event the web UI receives
           (``type`` is ``live.level``, ``live.segments``, ``live.stopped``, ...).
7 PING     either direction. Payload is echoed back.
8 PONG     the echo.
9 ERROR    server → client. JSON ``{"message": "..."}``.
10 BOOKMARK client → server. JSON ``{"note": "optional"}``. Marks the
   current moment in the transcript.
11 PAUSE    client → server. Empty. Drops speech until RESUME and leaves a
   ``[paused]`` gap in the transcript. The timeline keeps moving.
12 RESUME   client → server. Empty. Recording continues after the gap.

A new connection replaces the previous client. Replacing a client does not
stop a recording that client started; the new client can send STOP.
"""

from __future__ import annotations

import json
import struct

import numpy as np

PROTOCOL_VERSION = 1
MAX_FRAME = 8 * 1024 * 1024
MAX_CHANNELS = 8

HELLO = 1
WELCOME = 2
AUDIO = 3
START = 4
STOP = 5
EVENT = 6
PING = 7
PONG = 8
ERROR = 9
BOOKMARK = 10
PAUSE = 11
RESUME = 12

TRACK_SYSTEM = 0
TRACK_MIC = 1
TRACK_SPEAKER = {TRACK_SYSTEM: "Others", TRACK_MIC: "Me"}

_FRAME = struct.Struct("<IB")
_AUDIO = struct.Struct("<BIH")


class ProtocolError(ValueError):
    pass


def encode_frame(kind: int, payload: bytes = b"") -> bytes:
    if not 0 <= int(kind) <= 255:
        raise ProtocolError(f"Bad frame type {kind}.")
    payload = bytes(payload)
    length = 1 + len(payload)
    if length > MAX_FRAME:
        raise ProtocolError(f"Frame is {length} bytes; the limit is {MAX_FRAME}.")
    return _FRAME.pack(length, int(kind)) + payload


def encode_json(kind: int, obj: dict) -> bytes:
    return encode_frame(kind, json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def encode_audio(track: int, sample_rate: int, samples: np.ndarray, channels: int | None = None) -> bytes:
    """Pack one block of float PCM. ``samples`` is mono, or shaped (frames, channels)."""
    audio = np.asarray(samples, dtype="<f4")
    if audio.ndim == 1:
        ch = 1 if channels is None else int(channels)
        if ch != 1:
            raise ProtocolError("A 1-D block has one channel.")
        flat = audio
    elif audio.ndim == 2:
        ch = int(audio.shape[1] if channels is None else channels)
        if ch != audio.shape[1]:
            raise ProtocolError("Channel count does not match the array shape.")
        flat = np.ascontiguousarray(audio, dtype="<f4").reshape(-1)
    else:
        raise ProtocolError("Audio must be a 1-D or 2-D array.")
    if not 0 <= int(track) <= 255:
        raise ProtocolError(f"Bad track {track}.")
    if int(sample_rate) <= 0 or int(sample_rate) > 384000:
        raise ProtocolError(f"Bad sample rate {sample_rate}.")
    if not 1 <= ch <= MAX_CHANNELS:
        raise ProtocolError(f"Bad channel count {ch}.")
    payload = _AUDIO.pack(int(track), int(sample_rate), ch) + flat.tobytes()
    return encode_frame(AUDIO, payload)


def decode_json(payload: bytes) -> dict:
    try:
        obj = json.loads(payload.decode("utf-8")) if payload else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Frame was not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("JSON frame must be an object.")
    return obj


def decode_audio(payload: bytes) -> tuple[int, int, int, np.ndarray]:
    if len(payload) < _AUDIO.size:
        raise ProtocolError("Audio frame is too short.")
    track, rate, channels = _AUDIO.unpack_from(payload, 0)
    raw = payload[_AUDIO.size:]
    if len(raw) % 4 != 0:
        raise ProtocolError("Audio payload is not a whole number of float32 samples.")
    if channels < 1 or channels > MAX_CHANNELS:
        raise ProtocolError(f"Bad channel count {channels}.")
    if rate <= 0:
        raise ProtocolError(f"Bad sample rate {rate}.")
    samples = np.frombuffer(raw, dtype="<f4")
    if channels > 1:
        if samples.size % channels != 0:
            raise ProtocolError("Sample count is not divisible by the channel count.")
        samples = samples.reshape(-1, channels)
    return int(track), int(rate), int(channels), samples


class FrameDecoder:
    """Reassemble frames from arbitrary socket reads."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        if not data:
            return []
        self._buf.extend(data)
        frames = []
        while len(self._buf) >= 4:
            (length,) = struct.unpack_from("<I", self._buf, 0)
            if length < 1 or length > MAX_FRAME:
                self._buf.clear()
                raise ProtocolError(f"Frame length {length} is outside 1..{MAX_FRAME}.")
            total = 4 + length
            if len(self._buf) < total:
                break
            kind = self._buf[4]
            payload = bytes(self._buf[5:total])
            del self._buf[:total]
            frames.append((kind, payload))
        return frames
