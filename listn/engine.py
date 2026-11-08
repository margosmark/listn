"""Live recording sessions, file transcription jobs, and an event bus.

Threads in a live session:

* PortAudio callback: copies the block into a bounded queue and returns.
  It never blocks; if the queue is full the block is counted as dropped.
* pipeline thread: mixes to mono, resamples to 16 kHz, writes the WAV,
  updates the level meter, and cuts chunks for transcription.
* transcribe thread: runs Whisper on chunks in order. Capture starts
  before the model finishes loading, so nothing said early is lost.

Both the UI (via server-sent events) and the CLI consume the same events.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from listn import diarize
from listn.audio import TARGET_RATE, Chunker, PauseChunker, StreamResampler, WavWriter, decode_file, mix_mono, read_wav, rms, to_mono
from listn.vocab import hotword_string, prompt_with_vocabulary
from listn.ipc import TRACK_MIC, TRACK_SPEAKER, TRACK_SYSTEM
from listn.capture import open_input
from listn.config import Settings
from listn.library import Library, Segment, SessionMeta
from listn.transcriber import get_transcriber, resolve_backend

log = logging.getLogger("listn")

# A real microphone never delivers exact zeros; its noise floor is far above
# this. Exact zeros for a couple of seconds mean macOS is withholding audio.
DIGITAL_SILENCE = 1e-5
SILENCE_CHECK_SECONDS = 2.0
TRACK_KEYS = {TRACK_SYSTEM: "system", TRACK_MIC: "mic"}


def _clean_attendees(attendees: list) -> list[str]:
    names = []
    for item in attendees:
        text = str(item).strip()
        if text and text not in names:
            names.append(text[:80])
        if len(names) >= 40:
            break
    return names


class Cancelled(Exception):
    pass


class EventBus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class LiveSession:
    def __init__(self, library: Library, bus: EventBus, settings: Settings, title: str | None = None,
                 sd=None, transcriber_factory: Callable | None = None,
                 on_finished: Callable[["LiveSession"], bool] | None = None):
        self.library = library
        self.bus = bus
        self.settings = settings
        self.sd = sd
        self._factory = transcriber_factory or get_transcriber
        self._on_finished = on_finished
        self._book = None
        self.meta: SessionMeta = library.create(
            title or time.strftime("Recording %b %d, %H:%M"), "live",
            model=settings.model, backend=resolve_backend(settings.backend), language=settings.language,
            status="recording",
        )
        self.segments: list[Segment] = []
        self.state = "starting"
        self.level_db = -120.0
        self.dropped_blocks = 0
        self.overflows = 0
        self.pending_chunks = 0
        self.device_info: dict | None = None
        self._blocks: queue.Queue = queue.Queue(maxsize=2000)
        self._chunks: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self.mode = "device"
        self._writer: WavWriter | None = None
        self._writers: dict[int, WavWriter] = {}
        self._track_levels: dict[int, float] = {}
        self._ingest_started: float | None = None
        self._stream = None
        self._threads: list[threading.Thread] = []
        self._started = time.monotonic()
        self._samples = 0
        self.paused = False
        self._pause_started: float | None = None
        self.late_blocks = 0
        self._peaks: dict[str, float] = {}
        self._heard: dict[str, float] = {}
        self.silent_tracks: list[str] = []
        self._segment_lock = threading.Lock()
        self._last_disk = 0.0

    def _emit(self, kind: str, **data) -> None:
        self.bus.publish({"type": kind, "session": self.meta.id, **data})

    def status(self) -> dict:
        return {
            "id": self.meta.id,
            "title": self.meta.title,
            "state": self.state,
            "elapsed": round(self._elapsed_seconds(), 1),
            "level_db": round(self.level_db, 1),
            "capture": self.mode,
            "dropped_blocks": self.dropped_blocks,
            "overflows": self.overflows,
            "pending_chunks": self.pending_chunks,
            "device": self.device_info,
            "segments": len(self.segments),
            "model": self.meta.model,
            "backend": self.meta.backend,
            "paused": self.paused,
            "silent_tracks": list(self.silent_tracks),
        }

    def _elapsed_seconds(self) -> float:
        if self.mode == "ingest" and self._ingest_started:
            return time.monotonic() - self._ingest_started
        return self._samples / TARGET_RATE

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        try:
            self._blocks.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self, attendees: list | None = None) -> None:
        if self.settings.save_audio:
            self._writer = WavWriter(self.library.new_audio_path(self.meta.id, ".wav"))
            self.meta.audio_file = self._writer.path.name
            self.library.save_meta(self.meta)
        try:
            opened = open_input(self.settings.device, self._callback, sd=self.sd)
        except Exception as exc:
            self._fail(str(exc))
            raise
        self._stream = opened.stream
        self._rate = opened.samplerate
        self.device_info = {"name": opened.device.name, "channels": opened.channels,
                            "samplerate": int(opened.samplerate), "kind": opened.device.kind}
        self.meta.source_name = opened.device.name
        if attendees:
            self.meta.attendees = _clean_attendees(attendees)
        self.library.save_meta(self.meta)
        self.state = "loading-model"
        self._threads = [
            threading.Thread(target=self._pipeline, name="listn-pipeline", daemon=True),
            threading.Thread(target=self._transcribe_loop, name="listn-transcribe", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._emit("live.started", status=self.status())

    def start_ingest(self, source: dict, attendees: list | None = None) -> None:
        """Record PCM pushed in through the capture socket instead of an input device.

        ``source`` has ``source`` (``system`` or ``app``), optional ``app_name``,
        and ``mic``. System audio is track 0 and labelled Others. The microphone
        is track 1 and labelled Me. pyannote, if enabled, runs only on the
        system track so it does not rename Me.
        """
        self.mode = "ingest"
        self._ingest_started = time.monotonic()
        kind = source.get("source") or "system"
        if kind == "app" and source.get("app_name"):
            label = str(source["app_name"])
        else:
            label = "All system audio"
        if source.get("mic", True):
            label += " + microphone"
        self.meta.source_name = label
        self.meta.speakers = {"Me": "Me", "Others": "Others"}
        if attendees:
            self.meta.attendees = _clean_attendees(attendees)
        self.device_info = {
            "name": label,
            "kind": "process-tap",
            "channels": 2 if source.get("mic", True) else 1,
            "samplerate": None,
            "source": kind,
            "mic": bool(source.get("mic", True)),
        }
        self.library.save_meta(self.meta)
        self.state = "loading-model"
        self._threads = [
            threading.Thread(target=self._pipeline, name="listn-pipeline", daemon=True),
            threading.Thread(target=self._transcribe_loop, name="listn-transcribe", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._emit("live.started", status=self.status())

    def push_audio(self, track: int, sample_rate: int, samples: np.ndarray) -> bool:
        """Queue captured PCM. Returns False (and drops it) once the session is stopping."""
        if self.mode != "ingest" or self._stop.is_set():
            self.late_blocks += 1
            return False
        block = np.array(samples, dtype=np.float32, copy=True)
        try:
            self._blocks.put_nowait((int(track), int(sample_rate), block))
        except queue.Full:
            self.dropped_blocks += 1
        return True

    def _note_signal(self, key: str, audio: np.ndarray) -> None:
        """Watch for digital silence: macOS sends exact zeros when privacy access is missing."""
        if audio.size == 0:
            return
        peak = float(np.max(np.abs(audio)))
        self._peaks[key] = max(self._peaks.get(key, 0.0), peak)
        self._heard[key] = self._heard.get(key, 0.0) + audio.size / TARGET_RATE
        if key in self.silent_tracks or key == "system":
            return
        if self._heard[key] >= SILENCE_CHECK_SECONDS and self._peaks[key] <= DIGITAL_SILENCE:
            self.silent_tracks.append(key)
            from listn.privacy import silence_message

            message, pane = silence_message(key)
            self._emit("live.silent", track=key, message=message, pane=pane)

    def all_silent(self) -> bool:
        if not self._heard:
            return True
        return all(self._peaks.get(key, 0.0) <= DIGITAL_SILENCE for key in self._heard)

    def _chunker(self):
        if self.settings.vad_chunks:
            return PauseChunker(max_seconds=max(self.settings.chunk_seconds, 2.0))
        return Chunker(self.settings.chunk_seconds)

    def _pipeline(self) -> None:
        try:
            if self.mode == "ingest":
                self._pipeline_ingest()
            else:
                self._pipeline_device()
        except Exception as exc:
            log.exception("pipeline failed")
            self._emit("live.error", message=f"Audio pipeline error: {exc}")
        finally:
            self._close_writers()
            self._capture_done.set()
            self._chunks.put(None)

    def _pipeline_device(self) -> None:
        resampler = StreamResampler(self._rate)
        chunker = self._chunker()
        last_level = 0.0
        gap = 0.0
        was_paused = False
        while not (self._stop.is_set() and self._blocks.empty()):
                self._guard_disk()
                if self.mode == "device":
                    try:
                        self._reopen_input()
                    except Exception as exc:
                        self._emit("live.error", message=f"The input device dropped: {exc}")
                try:
                    block = self._blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                mono = resampler.process(to_mono(block))
                self._samples += mono.size
                if self.paused:
                    if not was_paused:
                        for start, chunk in chunker.flush():
                            self.pending_chunks += 1
                            self._chunks.put((None, start + gap, chunk))
                        was_paused = True
                    if self._writer:
                        self._writer.write(np.zeros(mono.size, dtype=np.float32))
                    gap += mono.size / TARGET_RATE
                    self.level_db = -120.0
                    continue
                was_paused = False
                self._note_signal("input", mono)
                if self._writer:
                    self._writer.write(mono)
                level = rms(mono)
                self.level_db = 20 * np.log10(level) if level > 1e-6 else -120.0
                for start, chunk in chunker.feed(mono):
                    self.pending_chunks += 1
                    self._chunks.put((None, start + gap, chunk))
                now = time.monotonic()
                if now - last_level > 0.25:
                    last_level = now
                    self._emit("live.level", level_db=round(self.level_db, 1),
                               elapsed=round(self._elapsed_seconds(), 1),
                               pending=self.pending_chunks, dropped=self.dropped_blocks)
        for start, chunk in chunker.flush():
            self.pending_chunks += 1
            self._chunks.put((None, start + gap, chunk))

    def _pipeline_ingest(self) -> None:
        resamplers: dict[int, StreamResampler] = {}
        chunkers: dict[int, Chunker] = {}
        offsets: dict[int, float] = {}
        gaps: dict[int, float] = {}
        last_level = 0.0
        was_paused = False
        while not (self._stop.is_set() and self._blocks.empty()):
            self._guard_disk()
            try:
                track, rate, block = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            if track not in (TRACK_SYSTEM, TRACK_MIC) or rate <= 0:
                continue
            mono = to_mono(block)
            if mono.size == 0:
                continue
            resampler = resamplers.get(track)
            if resampler is None or int(round(resampler.src_rate)) != int(round(rate)):
                resampler = StreamResampler(float(rate))
                resamplers[track] = resampler
            audio = resampler.process(mono)
            if audio.size == 0:
                continue
            if track not in offsets:
                origin = self._ingest_started or time.monotonic()
                offsets[track] = max(0.0, time.monotonic() - origin)
                if self.settings.save_audio:
                    self._open_track_writer(track, offsets[track])
            writer = self._writers.get(track)
            if self.paused:
                if not was_paused:
                    for held, held_chunker in chunkers.items():
                        for start, chunk in held_chunker.flush():
                            self.pending_chunks += 1
                            self._chunks.put((held, offsets.get(held, 0.0) + gaps.get(held, 0.0) + start, chunk))
                    was_paused = True
                if writer is not None:
                    writer.write(np.zeros(audio.size, dtype=np.float32))
                gaps[track] = gaps.get(track, 0.0) + audio.size / TARGET_RATE
                self._track_levels[track] = -120.0
                self.level_db = -120.0
                continue
            was_paused = False
            self._note_signal(TRACK_KEYS[track], audio)
            if writer is not None:
                writer.write(audio)
            level = rms(audio)
            self._track_levels[track] = 20 * np.log10(level) if level > 1e-6 else -120.0
            self.level_db = max(self._track_levels.values())
            chunker = chunkers.setdefault(track, self._chunker())
            for start, chunk in chunker.feed(audio):
                self.pending_chunks += 1
                self._chunks.put((track, offsets[track] + gaps.get(track, 0.0) + start, chunk))
            now = time.monotonic()
            if now - last_level > 0.25:
                last_level = now
                self._emit("live.level", level_db=round(self.level_db, 1),
                           levels={"system": round(self._track_levels.get(TRACK_SYSTEM, -120.0), 1),
                                   "mic": round(self._track_levels.get(TRACK_MIC, -120.0), 1)},
                           elapsed=round(self._elapsed_seconds(), 1),
                           pending=self.pending_chunks, dropped=self.dropped_blocks)
        for track, chunker in chunkers.items():
            for start, chunk in chunker.flush():
                self.pending_chunks += 1
                self._chunks.put((track, offsets.get(track, 0.0) + gaps.get(track, 0.0) + start, chunk))

    def _open_track_writer(self, track: int, offset: float) -> None:
        name = "audio-system.wav" if track == TRACK_SYSTEM else "audio-mic.wav"
        writer = WavWriter(self.library.session_path(self.meta.id) / name)
        pad = int(round(max(0.0, offset) * TARGET_RATE))
        if pad:
            writer.write(np.zeros(pad, dtype=np.float32))
        self._writers[track] = writer
        if not self.meta.audio_file:
            self.meta.audio_file = name
            self.library.save_meta(self.meta)

    def _transcribe_loop(self) -> None:
        try:
            model = self._factory(self.settings.model, self.settings.backend,
                                  status=lambda msg: self._emit("live.status", message=msg))
        except Exception as exc:
            self._emit("live.error", message=f"Could not load the model: {exc}")
            self.state = "recording-only"
            while self._chunks.get() is not None:
                self.pending_chunks -= 1
            return
        if self.state == "loading-model":
            self.state = "listening"
        self._emit("live.status", message="Listening", state=self.state)
        prompts: dict = {}
        recent: dict = {}
        while True:
            item = self._chunks.get()
            if item is None:
                break
            track, start, chunk = item
            speaker = TRACK_SPEAKER.get(track)
            if speaker == "Others" and self.settings.match_speakers:
                matched = self._match_known(chunk)
                if matched:
                    speaker = matched
                    self.meta.speakers = dict(self.meta.speakers or {})
                    if self.meta.speakers.get(matched) != matched:
                        self.meta.speakers[matched] = matched
                        self.library.save_meta(self.meta)
            try:
                raw = model.transcribe(
                    chunk, language=self.settings.language,
                    prompt=prompt_with_vocabulary(self.settings.vocabulary, prompts.get(track)),
                    hotwords=hotword_string(self.settings.vocabulary), offset=start)
            except Exception as exc:
                self._emit("live.error", message=f"Transcription failed for one chunk: {exc}")
                raw = []
            finally:
                self.pending_chunks -= 1
            new = [Segment(round(s.start, 2), round(s.end, 2), s.text, speaker) for s in raw]
            if new:
                bucket = recent.setdefault(track, [])
                bucket.extend(new)
                prompts[track] = " ".join(s.text for s in bucket[-3:])[-200:]
                with self._segment_lock:
                    self.segments.extend(new)
                    self.segments.sort(key=lambda s: (s.start, s.speaker or ""))
                    self.library.save_segments(self.meta.id, self.segments)
                self._emit("live.segments", segments=[s.to_dict() for s in new])

    def _match_known(self, chunk: np.ndarray) -> str | None:
        """Name a whole live chunk after a known voice. Never downloads during a call."""
        from listn.speakers import SpeakerBook, threshold_for, voice_embedder

        try:
            if self._book is None:
                book = SpeakerBook()
                if not book.profiles():
                    self._book = False
                    return None
                book.embedder = voice_embedder(allow_download=False)
                self._book = book
            if self._book is False or chunk.size < int(self._book.embedder.min_seconds * TARGET_RATE):
                return None
            matched, _score = self._book.match(chunk, threshold_for(self._book.embedder, self.settings))
            return matched
        except Exception as exc:
            log.debug("Live voice matching skipped: %s", exc)
            return None

    def pause(self) -> bool:
        if self._stop.is_set() or self.paused or self.state in ("done", "error", "finishing"):
            return False
        self.paused = True
        self._pause_started = self._elapsed_seconds()
        self.state = "paused"
        self._emit("live.paused", t=round(self._pause_started, 2), state=self.state)
        self._emit("live.status", message="Paused. Speech during the gap is not transcribed.", state=self.state)
        return True

    def resume(self) -> bool:
        if not self.paused or self._stop.is_set():
            return False
        end = self._elapsed_seconds()
        start = self._pause_started if self._pause_started is not None else end
        self.paused = False
        self._pause_started = None
        self.state = "listening"
        # A pause is always marked, even one shorter than the audio that has
        # arrived since, so the transcript shows that it happened.
        gap = Segment(round(start, 2), round(max(end, start + 0.05), 2), "[paused]", None)
        with self._segment_lock:
            self.segments.append(gap)
            self.segments.sort(key=lambda s: (s.start, s.speaker or ""))
            self.library.save_segments(self.meta.id, self.segments)
        self._emit("live.segments", segments=[gap.to_dict()])
        self._emit("live.resumed", state=self.state)
        self._emit("live.status", message="Listening", state=self.state)
        return True

    def _guard_disk(self) -> None:
        now = time.monotonic()
        if now - self._last_disk < 30:
            return
        self._last_disk = now
        from listn.power import DiskSpaceError, require_free_space

        try:
            require_free_space(self.library.root, self.settings.min_free_gb)
        except DiskSpaceError as exc:
            self.meta.error = str(exc)
            self._emit("live.error", message=str(exc))
            self._stop.set()

    def _reopen_input(self) -> None:
        stream = self._stream
        if stream is None or getattr(stream, "active", True):
            return
        now = time.monotonic()
        if now - getattr(self, "_reopen_at", 0.0) < 2:
            return
        self._reopen_at = now
        try:
            if stream is not None:
                stop = getattr(stream, "stop", None)
                if stop:
                    stop()
                stream.close()
        except Exception:
            pass
        opened = open_input(self.settings.device, self._callback, sd=self.sd)
        self._stream = opened.stream
        self._emit("live.status", message="The input device changed. Recording continued.")

    def stop(self, wait: bool = True) -> None:
        if self._stop.is_set():
            return
        self.state = "finishing"
        self._emit("live.status", message="Finishing the last chunks...", state=self.state)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stop.set()
        finisher = threading.Thread(target=self._finish, name="listn-finish", daemon=True)
        finisher.start()
        if wait:
            finisher.join()

    def _close_writers(self) -> None:
        if self._writer is not None:
            self._writer.close()
        for writer in self._writers.values():
            writer.close()

    def _write_mixdown(self) -> None:
        tracks = []
        root = self.library.session_path(self.meta.id)
        for name in ("audio-system.wav", "audio-mic.wav"):
            path = root / name
            if not path.exists():
                continue
            audio = read_wav(path)
            if audio.size:
                tracks.append(audio)
        if not tracks:
            return
        mixed = mix_mono(tracks)
        if mixed.size == 0:
            return
        out = WavWriter(root / "audio.wav")
        try:
            out.write(mixed)
        finally:
            out.close()
        self.meta.audio_file = "audio.wav"

    def _finish(self) -> None:
        for t in self._threads:
            t.join()
        if self.mode == "ingest":
            self._write_mixdown()
            ends = [writer.seconds for writer in self._writers.values()]
            self.meta.duration = round(max(ends) if ends else self._elapsed_seconds(), 2)
        else:
            self.meta.duration = round(self._samples / TARGET_RATE, 2)
        self.library.save_segments(self.meta.id, self.segments)
        from listn.privacy import SILENT_SESSION

        spoken = [s for s in self.segments if s.text != "[paused]"]
        silent = self.all_silent()
        if silent:
            self.meta.status = "error"
            self.meta.silent = True
            self.meta.error = SILENT_SESSION
        else:
            self.meta.status = "done"
            if not spoken:
                self.meta.warning = "No speech was recognised in this recording."
        self.library.save_meta(self.meta)
        self.state = "done"
        self._emit("live.stopped", session=self.meta.id, meta=self.meta.to_dict())
        from listn.retention import apply_retention
        from listn.summary import schedule_notes

        speakers_queued = False
        if spoken and not silent and self._on_finished is not None:
            try:
                speakers_queued = bool(self._on_finished(self))
            except Exception:
                log.exception("Could not queue speaker labels")
        try:
            apply_retention(self.library, self.settings.audio_retention_days)
        except OSError:
            log.exception("Retention pass failed")
        if spoken and not speakers_queued:
            schedule_notes(self.library, self.meta.id, self.settings, self.bus)

    def _fail(self, message: str) -> None:
        self.state = "error"
        self.meta.status = "error"
        self.meta.error = message
        self._close_writers()
        self.library.save_meta(self.meta)
        self._emit("live.error", message=message)


class FileJob:
    kind = "transcribe"

    def __init__(self, meta: SessionMeta, audio_path: Path, model: str, backend: str,
                 language: str | None, diarize_speakers: bool):
        self.meta = meta
        self.audio_path = audio_path
        self.model = model
        self.backend = backend
        self.language = language
        self.diarize = diarize_speakers
        self.progress = 0.0
        self.stage = "queued"
        self.detail: str | None = None
        self.cancelled = False
        self.hf_token: str | None = None
        self.match_speakers = True
        self.speaker_threshold: float | None = None
        self.spectrum_threshold: float | None = None

    def to_dict(self) -> dict:
        return {"id": self.meta.id, "title": self.meta.title, "progress": round(self.progress, 3),
                "stage": self.stage, "model": self.model, "kind": self.kind, "detail": self.detail}


class SpeakerJob(FileJob):
    """Label speakers on a saved session without transcribing it again."""

    kind = "speakers"

    def __init__(self, meta: SessionMeta, audio_path: Path | None, notes_after: bool = False):
        super().__init__(meta, audio_path, "pyannote", "local", None, True)
        self.notes_after = notes_after


SPEAKER_STAGE = "speakers"


class Engine:
    def __init__(self, library: Library | None = None, sd=None, transcriber_factory: Callable | None = None):
        self.library = library or Library()
        self.bus = EventBus()
        self.sd = sd
        self._factory = transcriber_factory or get_transcriber
        self.live: LiveSession | None = None
        self.jobs: dict[str, FileJob] = {}
        self._job_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.library.recover_interrupted()
        try:
            from listn.retention import apply_retention

            apply_retention(self.library, Settings.load().audio_retention_days)
        except OSError:
            log.exception("Retention pass failed")
        threading.Thread(target=self._job_worker, name="listn-jobs", daemon=True).start()

    def live_active(self) -> bool:
        return self.live is not None and self.live.state not in ("done", "error")

    def _tuned(self, settings: Settings, on_battery: bool, battery_percent: float | None) -> Settings:
        from dataclasses import replace

        from listn.power import model_for_power, require_free_space

        require_free_space(self.library.root, settings.min_free_gb)
        model, allow_notes = model_for_power(settings, on_battery, battery_percent)
        return replace(settings, model=model, summarize_after=bool(settings.summarize_after) and allow_notes)

    def start_live(self, settings: Settings, title: str | None = None, attendees: list | None = None,
                   on_battery: bool = False, battery_percent: float | None = None) -> LiveSession:
        with self._lock:
            if self.live_active():
                raise RuntimeError("A recording is already running.")
            tuned = self._tuned(settings, on_battery, battery_percent)
            session = LiveSession(self.library, self.bus, tuned, title, sd=self.sd,
                                  transcriber_factory=self._factory, on_finished=self._live_finished)
            session.start(attendees)
            self.live = session
            return session

    def start_ingest(self, settings: Settings, source: dict, title: str | None = None,
                     attendees: list | None = None, on_battery: bool = False,
                     battery_percent: float | None = None) -> LiveSession:
        with self._lock:
            if self.live_active():
                raise RuntimeError("A recording is already running.")
            tuned = self._tuned(settings, on_battery, battery_percent)
            session = LiveSession(self.library, self.bus, tuned, title, sd=self.sd,
                                  transcriber_factory=self._factory, on_finished=self._live_finished)
            session.start_ingest(source, attendees)
            self.live = session
            return session

    def pause_live(self) -> bool:
        session = self.live
        if session is None or not self.live_active():
            return False
        return session.pause()

    def resume_live(self) -> bool:
        session = self.live
        if session is None or not self.live_active():
            return False
        return session.resume()

    def add_bookmark(self, note: str = "", source: str | None = None) -> dict | None:
        """Mark the current moment. ``source`` records where the marker came
        from ("tray", "hotkey", …); it is stored only when the caller gives
        it, so existing bookmark records keep their exact shape."""
        session = self.live
        if session is None or not self.live_active():
            return None
        item = {"t": round(session._elapsed_seconds(), 2), "note": (note or "Bookmark").strip()[:200] or "Bookmark"}
        if source:
            item["source"] = str(source).strip()[:32]
        marks = self.library.bookmarks(session.meta.id)
        marks.append(item)
        self.library.save_bookmarks(session.meta.id, marks)
        self.bus.publish({"type": "live.bookmark", "session": session.meta.id, "bookmark": item})
        return item

    def stop_live(self, wait: bool = False) -> SessionMeta | None:
        session = self.live
        if session is None:
            return None
        session.stop(wait=wait)
        return session.meta

    def enqueue_file(self, src: Path, title: str, settings: Settings, diarize_speakers: bool,
                     move: bool = False, existing: SessionMeta | None = None) -> FileJob:
        if existing is not None:
            meta = existing
            self.library.update(meta.id, status="transcribing", error=None, model=settings.model,
                                backend=resolve_backend(settings.backend), language=settings.language)
            meta = self.library.get(meta.id)
            audio_path = src
        else:
            meta = self.library.create(title, "file", source_name=Path(src).name, model=settings.model,
                                       backend=resolve_backend(settings.backend), language=settings.language,
                                       status="transcribing")
            audio_path = self.library.import_audio(meta, Path(src), move=move)
        job = FileJob(meta, audio_path, settings.model, settings.backend, settings.language, diarize_speakers)
        self._speaker_settings(job, settings)
        if diarize_speakers:
            self.library.update(meta.id, speaker_state="queued", speaker_note=None)
        self.jobs[meta.id] = job
        self._job_queue.put(job)
        self.bus.publish({"type": "job.update", "job": job.to_dict()})
        return job

    @staticmethod
    def _speaker_settings(job: FileJob, settings: Settings) -> None:
        job.hf_token = settings.resolved_hf_token()
        job.match_speakers = bool(settings.match_speakers)
        job.speaker_threshold = settings.speaker_threshold
        job.spectrum_threshold = settings.spectrum_threshold

    def speaker_source(self, session_id: str) -> tuple[Path | None, bool]:
        """The audio to diarize and whether it is the Others track of a two-track recording."""
        root = self.library.session_path(session_id)
        system = root / "audio-system.wav"
        if system.exists():
            return system, True
        if (root / "audio-mic.wav").exists():
            return None, True
        return self.library.audio_path(self.library.get(session_id)), False

    def enqueue_speakers(self, session_id: str, settings: Settings, notes_after: bool = False) -> SpeakerJob:
        """Queue speaker labels for a saved session (after a recording, or from the library/CLI)."""
        meta = self.library.get(session_id)
        if session_id in self.jobs:
            raise RuntimeError("This session is busy.")
        if self.live_active() and self.live.meta.id == session_id:
            raise RuntimeError("This session is still recording.")
        path, _two_track = self.speaker_source(session_id)
        if path is None:
            raise FileNotFoundError("This session has no saved audio of the other people to label.")
        job = SpeakerJob(meta, path, notes_after=notes_after)
        self._speaker_settings(job, settings)
        job.detail = "Queued"
        self.library.update(session_id, speaker_state="queued", speaker_note=None)
        self.jobs[session_id] = job
        self._job_queue.put(job)
        self.bus.publish({"type": "job.update", "job": job.to_dict()})
        return job

    def _live_finished(self, session: LiveSession) -> bool:
        """Queue speaker labels for a finished recording. True when a job was queued."""
        settings = session.settings
        if not settings.speakers_after():
            return False
        ready = diarize.readiness(settings.resolved_hf_token())
        if not ready["ready"]:
            self.library.update(session.meta.id, speaker_state="skipped", speaker_note=ready["message"])
            self.bus.publish({"type": "speakers.skipped", "session": session.meta.id, "message": ready["message"]})
            return False
        path, _two_track = self.speaker_source(session.meta.id)
        if path is None:
            return False
        lines = [s for s in session.segments if s.speaker != "Me" and s.text != "[paused]"]
        if not lines:
            return False
        try:
            self.enqueue_speakers(session.meta.id, settings, notes_after=True)
        except (RuntimeError, FileNotFoundError, KeyError) as exc:
            log.info("Speaker labels not queued: %s", exc)
            return False
        return True

    def cancel_job(self, session_id: str) -> bool:
        job = self.jobs.get(session_id)
        if not job:
            return False
        job.cancelled = True
        return True

    def _job_update(self, job: FileJob, stage: str | None = None, progress: float | None = None) -> None:
        if job.cancelled:
            raise Cancelled()
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        self.bus.publish({"type": "job.update", "job": job.to_dict()})

    def _job_worker(self) -> None:
        while True:
            job: FileJob = self._job_queue.get()
            if job.cancelled:
                self._finish_job(job, error="Cancelled.")
                continue
            try:
                while self.live_active():
                    if job.stage != "waiting":
                        self._job_update(job, "waiting")
                    time.sleep(0.5)
                    if job.cancelled:
                        raise Cancelled()
                if job.kind == "speakers":
                    self._run_speaker_job(job)
                else:
                    self._run_job(job)
                self._finish_job(job)
            except (Cancelled, diarize.Cancelled):
                self._finish_job(job, error="Cancelled.")
            except Exception as exc:
                log.debug(traceback.format_exc())
                from listn.hub import redact

                self._finish_job(job, error=redact(str(exc) or exc.__class__.__name__, job.hf_token))

    def _identify(self, job: FileJob):
        """Callback for diarize_segments: name diarized speakers after known voices."""
        from listn.speakers import SpeakerBook, load_neural_embedder, threshold_for

        def identify(audio: np.ndarray, turns, labels: list[str]) -> dict:
            book = SpeakerBook()
            if not book.profiles():
                return {}
            book.embedder = load_neural_embedder(job.hf_token, allow_download=True)
            book.reenroll(self.library)
            probes = {}
            for label in labels:
                clip = diarize.speaker_audio(audio, turns, label)
                if clip.size >= int(book.embedder.min_seconds * TARGET_RATE):
                    probes[label] = book.embedder.embed(clip)
            limits = SimpleNamespace(speaker_threshold=job.speaker_threshold or Settings.speaker_threshold,
                                     spectrum_threshold=job.spectrum_threshold or Settings.spectrum_threshold)
            return book.identify(probes, threshold_for(book.embedder, limits))

        return identify

    def _label_speakers(self, job: FileJob, audio: np.ndarray | None, base: float = 0.0, span: float = 1.0) -> None:
        """Diarize the session's Others audio and relabel its lines. Me is never touched."""
        sid = job.meta.id
        path, two_track = self.speaker_source(sid)
        if path is None:
            raise FileNotFoundError("This session has no saved audio of the other people to label.")
        if two_track or audio is None:
            audio = decode_file(path)
        segments = self.library.segments(sid)
        if two_track:
            target = [s for s in segments if s.speaker != "Me" and s.text != "[paused]"]
        else:
            target = [s for s in segments if s.text != "[paused]"]
        chosen = {id(s) for s in target}
        keep = [s for s in segments if id(s) not in chosen]
        if not target:
            self.library.update(sid, speaker_state="skipped", speaker_note="No lines from other people to label.")
            return
        self.library.update(sid, speaker_state="running", speaker_note=None)

        def detail(message: str) -> None:
            job.detail = message.rstrip(".")
            self._job_update(job, SPEAKER_STAGE)

        def progress(fraction: float) -> None:
            self._job_update(job, SPEAKER_STAGE, base + span * fraction)

        result = diarize.diarize_segments(
            audio, target, job.hf_token, status=detail, progress=progress,
            cancelled=lambda: job.cancelled, default="Others" if two_track else None,
            identify=self._identify(job) if job.match_speakers else None)
        labelled = sorted(keep + result.segments, key=lambda s: (s.start, s.speaker or ""))
        names = {"Me": "Me"} if two_track else {}
        names.update(result.names)
        if any(s.speaker == "Others" for s in labelled):
            names["Others"] = "Others"
        known = sorted({m["name"] for m in result.matches.values() if m.get("name")})
        count = len(result.names)
        note = f"{count} speaker{'s' if count != 1 else ''} on the {'Others track' if two_track else 'recording'}"
        if known:
            note += f"; recognised {', '.join(known)}"
        self.library.save_segments(sid, labelled)
        self.library.update(sid, diarized=True, speakers=names, speaker_state="done", speaker_note=note + ".",
                            speaker_model=result.model_id, speaker_matches=result.matches)

    def _run_speaker_job(self, job: SpeakerJob) -> None:
        self._job_update(job, SPEAKER_STAGE, 0.0)
        job.detail = "Reading audio"
        self._job_update(job, SPEAKER_STAGE, 0.02)
        try:
            self._label_speakers(job, None, base=0.02, span=0.98)
        except (Cancelled, diarize.Cancelled):
            raise
        except Exception as exc:
            raise RuntimeError(self._speaker_failed(job, exc)) from exc

    def _run_job(self, job: FileJob) -> None:
        self._job_update(job, "decoding", 0.0)
        audio = decode_file(job.audio_path)
        duration = audio.size / TARGET_RATE
        self.library.update(job.meta.id, duration=round(duration, 2))
        if duration < 0.2:
            raise ValueError("The file contains no audible audio.")
        self._job_update(job, "loading-model")
        model = self._factory(job.model, job.backend,
                              status=lambda msg: self._job_update(job, msg))
        self._job_update(job, "transcribing", 0.0)
        raw = model.transcribe(audio, language=job.language,
                               progress=lambda p: self._job_update(job, None, p * (0.85 if job.diarize else 1.0)))
        segments = [Segment(round(s.start, 2), round(s.end, 2), s.text) for s in raw]
        self.library.save_segments(job.meta.id, segments)
        if job.diarize and segments:
            self._job_update(job, SPEAKER_STAGE, 0.85)
            try:
                self._label_speakers(job, audio, base=0.85, span=0.15)
            except (Cancelled, diarize.Cancelled):
                raise
            except Exception as exc:
                self._speaker_failed(job, exc)

    def _speaker_failed(self, job: FileJob, exc: Exception) -> str:
        from listn.hub import redact

        message = redact(str(exc) or exc.__class__.__name__, job.hf_token)
        state = "skipped" if isinstance(exc, (diarize.DiarizationUnavailable, FileNotFoundError)) else "error"
        note = message if state == "skipped" else f"Speaker labels failed: {message}"
        log.info("Speaker labels for %s: %s", job.meta.id, note)
        self.library.update(job.meta.id, speaker_state=state, speaker_note=note)
        return note

    def _finish_job(self, job: FileJob, error: str | None = None) -> None:
        from listn.summary import schedule_notes

        job.stage = "error" if error else "done"
        job.progress = job.progress if error else 1.0
        if job.kind == "speakers":
            # The transcript stays usable whatever happens to the speaker labels.
            if error == "Cancelled.":
                meta = self.library.update(job.meta.id, speaker_state="cancelled", speaker_note="Speaker labelling was cancelled.")
            elif error:
                meta = self.library.get(job.meta.id)
                if meta.speaker_state not in ("skipped", "error"):
                    self._speaker_failed(job, RuntimeError(error))
                meta = self.library.get(job.meta.id)
            else:
                meta = self.library.get(job.meta.id)
            self.jobs.pop(job.meta.id, None)
            self.bus.publish({"type": "job.done", "job": job.to_dict(), "meta": meta.to_dict(), "error": error})
            if job.notes_after:
                schedule_notes(self.library, job.meta.id, Settings.load(), self.bus)
            return
        if error and job.diarize:
            current = self.library.get(job.meta.id)
            if current.speaker_state in ("queued", "running"):
                self.library.update(job.meta.id, speaker_state=None)
        meta = self.library.update(job.meta.id, status="error" if error else "done", error=error)
        self.jobs.pop(job.meta.id, None)
        self.bus.publish({"type": "job.done", "job": job.to_dict(), "meta": meta.to_dict(), "error": error})
        if not error:
            schedule_notes(self.library, job.meta.id, Settings.load(), self.bus)
