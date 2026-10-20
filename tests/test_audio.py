import wave

import numpy as np
import pytest

from listn.audio import (Chunker, StreamResampler, WavWriter, decode_file, resample, rms,
                         to_int16, to_mono)


def sine(freq, seconds, rate, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono_ignores_silent_channels():
    voice = sine(300, 0.1, 16000)
    block = np.stack([voice, voice, np.zeros_like(voice)], axis=1)
    mono = to_mono(block)
    assert mono.shape == voice.shape
    assert rms(mono) == pytest.approx(rms(voice), rel=1e-5)


def test_to_mono_averages_active_channels_and_handles_silence():
    a, b = sine(300, 0.1, 16000, 0.4), sine(300, 0.1, 16000, 0.2)
    assert rms(to_mono(np.stack([a, b], axis=1))) == pytest.approx(rms((a + b) / 2), rel=1e-5)
    assert not to_mono(np.zeros((100, 2), dtype=np.float32)).any()
    assert to_mono(np.ones((5, 1), dtype=np.float32)).shape == (5,)


@pytest.mark.parametrize("src", [48000, 44100, 16000])
def test_resample_preserves_duration_and_pitch(src):
    out = resample(sine(440, 1.0, src), src, 16000)
    assert abs(out.size - 16000) <= 1
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spectrum) * 16000 / out.size
    assert peak_hz == pytest.approx(440, abs=3)


def test_stream_resampler_keeps_every_sample_across_blocks():
    audio = sine(440, 1.0, 48000)
    rs = StreamResampler(48000)
    pieces = [rs.process(audio[i:i + 1000]) for i in range(0, audio.size, 1000)]
    joined = np.concatenate(pieces)
    assert joined.size == 16000
    np.testing.assert_allclose(joined, resample(audio, 48000), atol=1e-6)


def test_to_int16_clips_instead_of_wrapping():
    out = to_int16(np.array([1.5, -1.5, 0.0, 1.0], dtype=np.float32))
    assert out.tolist() == [32767, -32767, 0, 32767]


def test_wav_writer_header_valid_while_recording(tmp_path):
    path = tmp_path / "rec.wav"
    w = WavWriter(path)
    w.write(sine(440, 0.5, 16000))
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 8000
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
    w.write(sine(440, 0.5, 16000))
    w.close()
    assert w.seconds == pytest.approx(1.0)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 16000


def test_chunker_cuts_in_quiet_gap_and_loses_nothing():
    rate = 16000
    speech = sine(300, 5.2, rate)
    gap = np.zeros(int(0.3 * rate), dtype=np.float32)
    audio = np.concatenate([speech, gap, sine(300, 3.0, rate)])
    ch = Chunker(chunk_seconds=6.0, search_seconds=1.5)
    chunks = []
    for i in range(0, audio.size, 1600):
        chunks += ch.feed(audio[i:i + 1600])
    chunks += ch.flush()
    first_start, first = chunks[0]
    assert first_start == 0.0
    assert 5.2 <= first.size / rate <= 5.5
    assert sum(c.size for _, c in chunks) == audio.size
    assert chunks[1][0] == pytest.approx(first.size / rate)


def test_chunker_flush_drops_tiny_tail():
    ch = Chunker(chunk_seconds=4.0)
    assert ch.feed(np.zeros(1000, dtype=np.float32)) == []
    assert ch.flush() == []


def test_decode_wav_resamples_to_16k_mono(tmp_path):
    path = tmp_path / "stereo44.wav"
    audio = sine(440, 1.0, 44100)
    stereo = np.stack([audio, audio], axis=1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(to_int16(stereo).tobytes())
    out = decode_file(path)
    assert abs(out.size - 16000) < 400
    assert rms(out) == pytest.approx(rms(audio), rel=0.05)


def test_decode_mp3(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "tone.mp3"
    try:
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mp3", rate=44100)
            stream.layout = "mono"
            samples = sine(440, 1.0, 44100)
            frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = 44100
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except Exception as exc:
        pytest.skip(f"mp3 encoder unavailable: {exc}")
    out = decode_file(path)
    assert 0.9 < out.size / 16000 < 1.2


def test_decode_rejects_non_audio(tmp_path):
    path = tmp_path / "notes.wav"
    path.write_text("not audio")
    with pytest.raises(Exception):
        decode_file(path)
