"""Pure audio helpers: mixing, resampling, chunking, WAV writing, decoding."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_RATE = 16000
SILENT_CHANNEL_RMS = 1e-4


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def to_mono(block: np.ndarray) -> np.ndarray:
    """Mix a (frames, channels) block to mono float32.

    A plain mean halves the level when only one side of an aggregate device
    is active (e.g. BlackHole carries the call and the mic channel is quiet),
    which pushes real speech under the silence gate. Average only channels
    that carry signal instead.
    """
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0].copy()
    levels = np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=0))
    active = levels > SILENT_CHANNEL_RMS
    if not active.any():
        return block.mean(axis=1).astype(np.float32)
    return block[:, active].mean(axis=1).astype(np.float32)


def lowpass_taps(src_rate: float, dst_rate: float, n_taps: int = 95) -> np.ndarray:
    """Windowed-sinc anti-alias filter just below the target Nyquist."""
    cutoff = 0.45 * dst_rate / src_rate
    m = np.arange(n_taps) - (n_taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * m) * np.blackman(n_taps)
    return (h / h.sum()).astype(np.float32)


class StreamResampler:
    """Resample a block stream continuously (filter state and phase carry over).

    Downsampling low-passes first: BlackHole usually runs at 48 kHz, and
    naive decimation folds 8-24 kHz content back into the speech band.
    """

    def __init__(self, src_rate: float, dst_rate: int = TARGET_RATE):
        self.src_rate = float(src_rate)
        self.dst_rate = dst_rate
        self.passthrough = int(round(self.src_rate)) == dst_rate
        self.step = self.src_rate / dst_rate
        self.taps = lowpass_taps(self.src_rate, dst_rate) if self.src_rate > dst_rate else None
        self._hist = np.zeros(0 if self.taps is None else self.taps.size - 1, dtype=np.float32)
        self._prev: float | None = None
        self._pos = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.passthrough or block.size == 0:
            return block
        if self.taps is not None:
            x = np.concatenate([self._hist, block])
            y = np.convolve(x, self.taps, mode="valid").astype(np.float32)
            self._hist = x[-(self.taps.size - 1):]
        else:
            y = block
        if self._prev is None:
            arr, base = y, 0
        else:
            arr, base = np.concatenate([[self._prev], y]), -1
        last = y.size - 1
        count = int(np.floor((last - self._pos) / self.step)) + 1 if self._pos <= last else 0
        positions = self._pos + self.step * np.arange(count)
        out = np.interp(positions - base, np.arange(arr.size), arr).astype(np.float32)
        self._pos = self._pos + self.step * count - y.size
        self._prev = float(y[-1])
        return out


def resample(audio: np.ndarray, src_rate: float, dst_rate: int = TARGET_RATE) -> np.ndarray:
    return StreamResampler(src_rate, dst_rate).process(audio)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Float [-1, 1] to int16 with clipping (no wrap-around on loud peaks)."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class WavWriter:
    """Incremental mono 16-bit WAV writer.

    ``wave`` patches the header on every write, so the file stays playable
    even if the process dies mid-recording.
    """

    def __init__(self, path: Path, rate: int = TARGET_RATE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.frames = 0
        self._wf = wave.open(str(self.path), "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(rate)

    def write(self, audio: np.ndarray) -> None:
        if audio.size:
            self._wf.writeframes(to_int16(audio).tobytes())
            self.frames += int(audio.size)

    @property
    def seconds(self) -> float:
        return self.frames / self.rate

    def close(self) -> None:
        if self._wf is not None:
            self._wf.close()
            self._wf = None


class PauseChunker:
    """Cut live audio on a pause, and never later than ``max_seconds``.

    A fixed window splits words that happen to straddle the boundary. This
    waits until the signal stays under ``threshold`` for ``pause_seconds``,
    as long as the chunk already has ``min_seconds`` of audio. If nobody
    pauses, it cuts at ``max_seconds`` so latency stays bounded.
    """

    def __init__(self, max_seconds: float = 15.0, pause_seconds: float = 0.45,
                 min_seconds: float = 0.35, threshold: float = 0.012,
                 rate: int = TARGET_RATE, window_seconds: float = 0.02):
        self.rate = rate
        self.max = max(1, int(max_seconds * rate))
        self.pause = max(1, int(pause_seconds * rate))
        self.min = max(1, int(min_seconds * rate))
        self.threshold = float(threshold)
        self.window = max(1, int(window_seconds * rate))
        self._buf = np.zeros(0, dtype=np.float32)
        self.consumed = 0

    def _loud(self, frame: np.ndarray) -> bool:
        return rms(frame) >= self.threshold

    def _find_cut(self) -> int | None:
        if self._buf.size >= self.max:
            return self.max
        speech = 0
        silence = 0
        pos = 0
        limit = self._buf.size - (self._buf.size % self.window)
        while pos < limit:
            frame = self._buf[pos:pos + self.window]
            if self._loud(frame):
                speech += self.window
                silence = 0
            else:
                silence += self.window
                if speech >= self.min and silence >= self.pause:
                    return pos + self.window
            pos += self.window
        return None

    def feed(self, audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
        self._buf = np.concatenate([self._buf, np.asarray(audio, dtype=np.float32).reshape(-1)])
        out = []
        while True:
            cut = self._find_cut()
            if cut is None or cut <= 0:
                break
            chunk = self._buf[:cut].copy()
            start = self.consumed / self.rate
            self._buf = self._buf[cut:]
            self.consumed += cut
            if self._loud(chunk):
                out.append((start, chunk))
        return out

    def flush(self, min_seconds: float = 0.2) -> list[tuple[float, np.ndarray]]:
        if self._buf.size < int(min_seconds * self.rate) or not self._loud(self._buf):
            self.consumed += self._buf.size
            self._buf = np.zeros(0, dtype=np.float32)
            return []
        chunk = (self.consumed / self.rate, self._buf.copy())
        self.consumed += self._buf.size
        self._buf = np.zeros(0, dtype=np.float32)
        return [chunk]


class Chunker:
    """Collect 16 kHz mono audio and cut it into transcription chunks.

    Rather than slicing at a fixed length (which splits words in half), each
    chunk ends at the quietest short window within the last ``search_seconds``
    before the target length.
    """

    def __init__(self, chunk_seconds: float = 6.0, rate: int = TARGET_RATE,
                 search_seconds: float = 1.5, window_seconds: float = 0.1):
        self.rate = rate
        self.target = int(chunk_seconds * rate)
        self.search = min(int(search_seconds * rate), self.target // 2)
        self.window = max(1, int(window_seconds * rate))
        self._buf = np.zeros(0, dtype=np.float32)
        self.consumed = 0

    def _cut_point(self, data: np.ndarray) -> int:
        lo = self.target - self.search
        region = data[lo:self.target]
        if region.size < self.window * 2:
            return self.target
        n_windows = region.size // self.window
        energies = np.square(region[: n_windows * self.window]).reshape(n_windows, self.window).mean(axis=1)
        best = int(np.argmin(energies))
        return lo + best * self.window + self.window // 2

    def feed(self, audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """Add audio; return ready chunks as (start_seconds, samples)."""
        self._buf = np.concatenate([self._buf, np.asarray(audio, dtype=np.float32)])
        out = []
        while self._buf.size >= self.target:
            cut = self._cut_point(self._buf)
            out.append((self.consumed / self.rate, self._buf[:cut].copy()))
            self._buf = self._buf[cut:]
            self.consumed += cut
        return out

    def flush(self, min_seconds: float = 0.5) -> list[tuple[float, np.ndarray]]:
        if self._buf.size < int(min_seconds * self.rate):
            self.consumed += self._buf.size
            self._buf = np.zeros(0, dtype=np.float32)
            return []
        chunk = (self.consumed / self.rate, self._buf.copy())
        self.consumed += self._buf.size
        self._buf = np.zeros(0, dtype=np.float32)
        return [chunk]


def decode_file(path: Path, rate: int = TARGET_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file (wav, mp3, m4a, flac...) to 16 kHz mono.

    Uses PyAV, which bundles its own ffmpeg, so no Homebrew ffmpeg is needed.
    """
    import av

    chunks = []
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError(f"{Path(path).name} has no audio stream.")
        # Resample in the source layout and mix down ourselves: ffmpeg's stereo
        # downmix adds channels at -3 dB each, which boosts and can clip.
        resampler = av.AudioResampler(format="fltp", rate=rate)

        def take(frames):
            for out in frames:
                planes = out.to_ndarray()
                chunks.append(to_mono(planes.T) if planes.ndim == 2 else planes.reshape(-1))

        for frame in container.decode(streams[0]):
            take(resampler.resample(frame))
        take(resampler.resample(None))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def read_wav(path: Path) -> np.ndarray:
    """Read a 16-bit WAV file listn wrote, as mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise ValueError(f"{Path(path).name} is not 16-bit PCM.")
        raw = wf.readframes(wf.getnframes())
    if not raw:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = to_mono(pcm.reshape(-1, channels))
    return np.ascontiguousarray(pcm, dtype=np.float32)


def mix_mono(tracks: list[np.ndarray]) -> np.ndarray:
    """Sum mono tracks that already share a timeline, clipping peaks.

    A missing or empty track contributes nothing, so a mic-only or
    system-only recording is not attenuated.
    """
    present = [np.asarray(t, dtype=np.float32).reshape(-1) for t in tracks if np.size(t)]
    if not present:
        return np.zeros(0, dtype=np.float32)
    length = max(t.size for t in present)
    mixed = np.zeros(length, dtype=np.float32)
    for track in present:
        mixed[: track.size] += track
    return np.clip(mixed, -1.0, 1.0)
