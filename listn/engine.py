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

import numpy as np

from listn import diarize
from listn.audio import TARGET_RATE, Chunker, StreamResampler, WavWriter, decode_file, rms, to_mono
from listn.capture import open_input
from listn.config import Settings
from listn.library import Library, Segment, SessionMeta
from listn.transcriber import get_transcriber, resolve_backend

log = logging.getLogger("listn")


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
                 sd=None, transcriber_factory: Callable | None = None):
        self.library = library
        self.bus = bus
        self.settings = settings
        self.sd = sd
        self._factory = transcriber_factory or get_transcriber
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
        self._writer: WavWriter | None = None
        self._stream = None
        self._threads: list[threading.Thread] = []
        self._started = time.monotonic()
        self._samples = 0

    def _emit(self, kind: str, **data) -> None:
        self.bus.publish({"type": kind, "session": self.meta.id, **data})

    def status(self) -> dict:
        return {
            "id": self.meta.id,
            "title": self.meta.title,
            "state": self.state,
            "elapsed": round(self._samples / TARGET_RATE, 1),
            "level_db": round(self.level_db, 1),
            "dropped_blocks": self.dropped_blocks,
            "overflows": self.overflows,
            "pending_chunks": self.pending_chunks,
            "device": self.device_info,
            "segments": len(self.segments),
            "model": self.meta.model,
            "backend": self.meta.backend,
        }

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        try:
            self._blocks.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self) -> None:
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
        self.library.save_meta(self.meta)
        self.state = "loading-model"
        self._threads = [
            threading.Thread(target=self._pipeline, name="listn-pipeline", daemon=True),
            threading.Thread(target=self._transcribe_loop, name="listn-transcribe", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._emit("live.started", status=self.status())

    def _pipeline(self) -> None:
        resampler = StreamResampler(self._rate)
        chunker = Chunker(self.settings.chunk_seconds)
        last_level = 0.0
        try:
            while not (self._stop.is_set() and self._blocks.empty()):
                try:
                    block = self._blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                mono = resampler.process(to_mono(block))
                self._samples += mono.size
                if self._writer:
                    self._writer.write(mono)
                level = rms(mono)
                self.level_db = 20 * np.log10(level) if level > 1e-6 else -120.0
                for start, chunk in chunker.feed(mono):
                    self.pending_chunks += 1
                    self._chunks.put((start, chunk))
                now = time.monotonic()
                if now - last_level > 0.25:
                    last_level = now
                    self._emit("live.level", level_db=round(self.level_db, 1),
                               elapsed=round(self._samples / TARGET_RATE, 1),
                               pending=self.pending_chunks, dropped=self.dropped_blocks)
            for start, chunk in chunker.flush():
                self.pending_chunks += 1
                self._chunks.put((start, chunk))
        except Exception as exc:
            log.exception("pipeline failed")
            self._emit("live.error", message=f"Audio pipeline error: {exc}")
        finally:
            if self._writer:
                self._writer.close()
            self._capture_done.set()
            self._chunks.put(None)

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
        prompt = None
        while True:
            item = self._chunks.get()
            if item is None:
                break
            start, chunk = item
            try:
                raw = model.transcribe(chunk, language=self.settings.language, prompt=prompt, offset=start)
            except Exception as exc:
                self._emit("live.error", message=f"Transcription failed for one chunk: {exc}")
                raw = []
            finally:
                self.pending_chunks -= 1
            new = [Segment(round(s.start, 2), round(s.end, 2), s.text) for s in raw]
            if new:
                self.segments.extend(new)
                prompt = " ".join(s.text for s in self.segments[-3:])[-200:]
                self.library.save_segments(self.meta.id, self.segments)
                self._emit("live.segments", segments=[s.to_dict() for s in new])

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

    def _finish(self) -> None:
        for t in self._threads:
            t.join()
        self.meta.duration = round(self._samples / TARGET_RATE, 2)
        self.library.save_segments(self.meta.id, self.segments)
        if self.settings.diarize_after and self.segments and self._writer:
            self.state = "diarizing"
            self._emit("live.status", message="Identifying speakers...", state=self.state)
            try:
                audio = decode_file(self._writer.path)
                labelled, names = diarize.diarize_segments(audio, self.segments, self.settings.resolved_hf_token())
                self.segments = labelled
                self.meta.speakers = names
                self.meta.diarized = True
                self.library.save_segments(self.meta.id, self.segments)
            except Exception as exc:
                self._emit("live.error", message=f"Speaker labels skipped: {exc}")
        self.meta.status = "done"
        self.library.save_meta(self.meta)
        self.state = "done"
        self._emit("live.stopped", session=self.meta.id, meta=self.meta.to_dict())

    def _fail(self, message: str) -> None:
        self.state = "error"
        self.meta.status = "error"
        self.meta.error = message
        if self._writer:
            self._writer.close()
        self.library.save_meta(self.meta)
        self._emit("live.error", message=message)


class FileJob:
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
        self.cancelled = False
        self.hf_token: str | None = None

    def to_dict(self) -> dict:
        return {"id": self.meta.id, "title": self.meta.title, "progress": round(self.progress, 3),
                "stage": self.stage, "model": self.model}


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
        threading.Thread(target=self._job_worker, name="listn-jobs", daemon=True).start()

    def live_active(self) -> bool:
        return self.live is not None and self.live.state not in ("done", "error")

    def start_live(self, settings: Settings, title: str | None = None) -> LiveSession:
        with self._lock:
            if self.live_active():
                raise RuntimeError("A recording is already running.")
            session = LiveSession(self.library, self.bus, settings, title, sd=self.sd,
                                  transcriber_factory=self._factory)
            session.start()
            self.live = session
            return session

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
        job.hf_token = settings.resolved_hf_token()
        self.jobs[meta.id] = job
        self._job_queue.put(job)
        self.bus.publish({"type": "job.update", "job": job.to_dict()})
        return job

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
                self._run_job(job)
                self._finish_job(job)
            except Cancelled:
                self._finish_job(job, error="Cancelled.")
            except Exception as exc:
                log.debug(traceback.format_exc())
                self._finish_job(job, error=str(exc) or exc.__class__.__name__)

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
            self._job_update(job, "speakers", 0.9)
            labelled, names = diarize.diarize_segments(audio, segments, job.hf_token)
            self.library.save_segments(job.meta.id, labelled)
            self.library.update(job.meta.id, diarized=True, speakers=names)

    def _finish_job(self, job: FileJob, error: str | None = None) -> None:
        job.stage = "error" if error else "done"
        job.progress = job.progress if error else 1.0
        meta = self.library.update(job.meta.id, status="error" if error else "done", error=error)
        self.jobs.pop(job.meta.id, None)
        self.bus.publish({"type": "job.done", "job": job.to_dict(), "meta": meta.to_dict(), "error": error})
