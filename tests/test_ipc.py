"""Frame codec tests. No socket and no audio hardware."""

import struct

import numpy as np
import pytest

from listn.ipc import (
    AUDIO,
    HELLO,
    MAX_FRAME,
    PING,
    TRACK_MIC,
    FrameDecoder,
    ProtocolError,
    decode_audio,
    decode_json,
    encode_audio,
    encode_frame,
    encode_json,
)


def test_json_frame_roundtrip():
    raw = encode_json(HELLO, {"client": "listn-menu", "version": 1})
    frames = FrameDecoder().feed(raw)
    assert len(frames) == 1
    kind, payload = frames[0]
    assert kind == HELLO
    assert decode_json(payload) == {"client": "listn-menu", "version": 1}


def test_decoder_reassembles_split_reads_and_back_to_back_frames():
    first = encode_frame(PING, b"abc")
    second = encode_json(HELLO, {"version": 1})
    blob = first + second
    decoder = FrameDecoder()
    assert decoder.feed(blob[:3]) == []
    assert decoder.feed(blob[3:7]) == []
    frames = decoder.feed(blob[7:])
    assert [kind for kind, _ in frames] == [PING, HELLO]
    assert frames[0][1] == b"abc"
    assert decode_json(frames[1][1])["version"] == 1


def test_decoder_rejects_an_absurd_length():
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError):
        decoder.feed(struct.pack("<I", MAX_FRAME + 1))
    # A bad length is fatal for that buffer, and a later valid frame still parses.
    frames = decoder.feed(encode_frame(PING, b"z"))
    assert frames == [(PING, b"z")]


def test_audio_frame_mono_and_stereo():
    mono = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)
    kind, payload = FrameDecoder().feed(encode_audio(TRACK_MIC, 48000, mono))[0]
    assert kind == AUDIO
    track, rate, channels, samples = decode_audio(payload)
    assert (track, rate, channels) == (TRACK_MIC, 48000, 1)
    assert samples.shape == (4,)
    assert np.allclose(samples, mono)

    stereo = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
    _, payload = FrameDecoder().feed(encode_audio(0, 16000, stereo))[0]
    track, rate, channels, samples = decode_audio(payload)
    assert (track, rate, channels) == (0, 16000, 2)
    assert samples.shape == (2, 2)
    assert np.allclose(samples, stereo)


def test_audio_frame_rejects_a_truncated_payload():
    with pytest.raises(ProtocolError):
        decode_audio(b"\x00\x80\x3e\x00\x00\x01")  # header claims data but is short of 7 bytes? 
    header = struct.pack("<BIH", 0, 16000, 1)
    with pytest.raises(ProtocolError):
        decode_audio(header + b"\x00\x00")  # not a whole float
