"""Native crash isolation: model load + inference in a child process.

Every crash here is a real SIGSEGV/SIGABRT delivered inside a real worker
child (see tests/_crash_backends.py, injected through
``transcriber._FACTORY_PATH``), because that is what a broken CUDA stack
does — and it must never reach the caller or the library. Audio is
synthetic; no network, no real recordings, no GPU required.

Covered: crash after model load, crash during segment iteration, crash
during load itself; the auto->CPU fallback (exactly one retry, said out
loud), explicit cuda (failure, no fallback), plain cpu (error, nothing to
fall back to); and that a crashed attempt never leaves a partial or
duplicate transcript behind.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from listn import cli
from listn import transcriber as tr
from listn.config import Settings
from listn.engine import Engine
from listn.library import Library
from listn.transcriber import CudaUnavailable, IsolatedTranscriber

from .conftest import MAC_DEVICES, FakeSD
from .test_engine import wait_for, write_wav

FACTORIES = "tests._crash_backends"


@pytest.fixture(autouse=True)
def clean_worker_cache():
    """Each test gets its own cached wrapper (and worker) lifecycle."""
    tr._close_cached()
    tr._cache.clear()
    yield
    tr._close_cached()
    tr._cache.clear()


def _iso(factory: str, compute_device: str, monkeypatch, data_dir) -> IsolatedTranscriber:
    """A real isolated wrapper whose worker builds ``factory`` instead of a model."""
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:{factory}")
    return tr.get_transcriber("tiny", "faster-whisper", compute_device=compute_device)


def _audio() -> np.ndarray:
    return np.full(16000, 0.1, dtype=np.float32)


def _loads(path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


def _load_log(tmp_path, monkeypatch):
    path = tmp_path / "loads.txt"
    monkeypatch.setenv("LISTN_TEST_LOAD_LOG", str(path))
    return path


# --- wrapper level: the child takes the crash --------------------------------

def test_round_trip_runs_in_a_child_process(monkeypatch, data_dir):
    wrapper = _iso("roundtrip_factory", "cpu", monkeypatch, data_dir)
    assert isinstance(wrapper, IsolatedTranscriber)
    seen = []
    segs = wrapper.transcribe(_audio(), language="en", progress=seen.append)
    assert [s.text for s in segs] == ["hello from the worker", "second line"]
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    assert wrapper.device_message == "device: cpu"
    # The worker really is another process — and it is still alive after use.
    assert wrapper._session is not None and wrapper._session.proc.pid != os.getpid()
    assert wrapper._session.proc.poll() is None
    wrapper.close()
    assert wrapper._session is None


def test_sigsegv_after_model_load_auto_falls_back_to_cpu(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("segfault_unless_cpu_factory", "auto", monkeypatch, data_dir)
    assert _loads(log) == 1  # the GPU load succeeded; the crash comes at inference
    segs = wrapper.transcribe(_audio())
    assert [s.text for s in segs] == ["cpu result one", "cpu result two"]
    assert wrapper.device == "cpu"
    assert "SIGSEGV" in wrapper.device_message
    assert "running on CPU" in wrapper.device_message
    assert _loads(log) == 2  # exactly one CPU retry, no loop


def test_sigsegv_during_iteration_never_returns_partial_segments(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("abort_mid_iteration_factory", "auto", monkeypatch, data_dir)
    segs = wrapper.transcribe(_audio())
    # The aborted attempt's first segment must be gone, not half-kept.
    assert [s.text for s in segs] == ["complete one", "complete two"]
    assert not any("partial" in s.text for s in segs)
    assert "SIGABRT" in wrapper.device_message
    assert _loads(log) == 2


def test_sigsegv_during_model_load_auto_rebuilds_on_cpu(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("crash_on_load_factory", "auto", monkeypatch, data_dir)
    assert _loads(log) == 2  # first child died building; CPU child rebuilt
    segs = wrapper.transcribe(_audio())
    assert [s.text for s in segs] == ["rebuilt on cpu"]
    assert wrapper.device == "cpu"
    assert "SIGSEGV" in wrapper.device_message and "loading the model" in wrapper.device_message


def test_explicit_cuda_sigsegv_fails_with_advice_and_never_retries(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("segfault_unless_cpu_factory", "cuda", monkeypatch, data_dir)
    with pytest.raises(CudaUnavailable) as excinfo:
        wrapper.transcribe(_audio())
    message = str(excinfo.value)
    assert "SIGSEGV" in message and "transcribing" in message
    assert "--compute-device cpu" in message and "docs/GPU_LINUX.md" in message
    assert _loads(log) == 1  # no silent CPU retry when CUDA was explicit
    # The failure is sticky: later calls fail fast, they never respawn a worker.
    with pytest.raises(CudaUnavailable, match="SIGSEGV"):
        wrapper.transcribe(_audio())
    assert _loads(log) == 1


def test_sigsegv_on_cpu_is_an_error_not_a_fallback(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("always_crash_factory", "cpu", monkeypatch, data_dir)
    with pytest.raises(RuntimeError) as excinfo:
        wrapper.transcribe(_audio())
    message = str(excinfo.value)
    assert "SIGSEGV" in message and "no other device to fall back to" in message
    assert _loads(log) == 1  # cpu never "falls back" to anything


def test_python_cuda_error_auto_falls_back_but_explicit_cuda_fails(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    wrapper = _iso("cuda_error_unless_cpu_factory", "auto", monkeypatch, data_dir)
    segs = wrapper.transcribe(_audio())
    assert [s.text for s in segs] == ["recovered on cpu"]
    assert "libcublas.so.12" in wrapper.device_message
    assert "running on CPU" in wrapper.device_message
    assert _loads(log) == 2

    explicit = _iso("cuda_error_unless_cpu_factory", "cuda", monkeypatch, data_dir)
    with pytest.raises(CudaUnavailable) as excinfo:
        explicit.transcribe(_audio())
    assert "libcublas.so.12" in str(excinfo.value)
    assert "--compute-device cpu" in str(excinfo.value)
    assert _loads(log) == 3  # only the auto attempt retried; explicit loaded once


def test_missing_worker_factory_is_a_readable_error(monkeypatch, data_dir):
    monkeypatch.setattr(tr, "_FACTORY_PATH", "tests.test_isolation:no_such_factory")
    with pytest.raises(RuntimeError, match="no_such_factory"):
        tr.get_transcriber("tiny", "faster-whisper", compute_device="cpu")


# --- engine level: library writes only after a clean child answer ------------

def _run_file_job(monkeypatch, data_dir, tmp_path, factory: str, device: str):
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", device)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:{factory}")
    engine = Engine(transcriber_factory=tr.get_transcriber)
    events = []
    queue = engine.bus.subscribe()
    job = engine.enqueue_file(write_wav(tmp_path / "input.wav"), "Input",
                              Settings(model="tiny"), diarize_speakers=False)
    while True:
        event = queue.get(timeout=30)
        events.append(event)
        if event["type"] == "job.done":
            break
    return engine, job, events


def test_file_job_sigsegv_leaves_no_partial_or_duplicate_transcript(monkeypatch, data_dir, tmp_path):
    engine, job, events = _run_file_job(monkeypatch, data_dir, tmp_path,
                                         "always_crash_factory", "cpu")
    done = events[-1]
    assert done["error"] and "SIGSEGV" in done["error"]
    assert "Traceback" not in done["error"]
    meta = engine.library.get(job.meta.id)
    assert meta.status == "error"
    # No partial write, no duplicate write: the file still holds exactly the
    # empty list create() put there — the crashed attempt produced nothing.
    assert engine.library.segments(meta.id) == []
    raw = json.loads((engine.library.session_path(meta.id) / "transcript.json").read_text())
    assert raw == []


def test_file_job_sigabrt_auto_falls_back_and_writes_the_transcript_once(monkeypatch, data_dir, tmp_path):
    log = _load_log(tmp_path, monkeypatch)
    engine, job, events = _run_file_job(monkeypatch, data_dir, tmp_path,
                                         "abort_mid_iteration_factory", "auto")
    assert events[-1]["error"] is None
    meta = engine.library.get(job.meta.id)
    assert meta.status == "done"
    texts = [s.text for s in engine.library.segments(meta.id)]
    assert texts == ["complete one", "complete two"]  # once: no duplicates, no partial
    stages = [e["job"]["stage"] for e in events if e["type"] == "job.update"]
    assert any("running on CPU" in stage for stage in stages)  # said out loud
    assert _loads(log) == 2


# --- live session: crash during a call ---------------------------------------

def test_live_native_crash_auto_falls_back_and_keeps_transcribing(monkeypatch, data_dir, tmp_path):
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "auto")
    _load_log(tmp_path, monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:abort_mid_iteration_factory")
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=tr.get_transcriber)
    queue = engine.bus.subscribe()
    session = engine.start_live(Settings(device="Aggregate", model="tiny", chunk_seconds=2.0))
    wait_for(lambda: session.segments, timeout=20.0)
    engine.stop_live(wait=True)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    statuses = [e["message"] for e in events if e["type"] == "live.status"]
    assert any("running on CPU" in m for m in statuses)  # truthful, not silent
    assert not any(e["type"] == "live.error" for e in events)
    meta = engine.library.get(session.meta.id)
    assert meta.status == "done"
    # One pair of lines per transcribed chunk; the crashed first attempt
    # contributed nothing — above all, no partial segment.
    texts = [s.text for s in engine.library.segments(meta.id)]
    assert len(texts) >= 2
    assert set(texts) <= {"complete one", "complete two"}
    assert texts.count("complete one") == texts.count("complete two")
    assert not any("partial" in t for t in texts)


def test_live_explicit_cuda_failure_is_said_once_and_keeps_recording(monkeypatch, data_dir, tmp_path):
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "cuda")
    _load_log(tmp_path, monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:cuda_error_unless_cpu_factory")
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=tr.get_transcriber)
    queue = engine.bus.subscribe()
    session = engine.start_live(Settings(device="Aggregate", model="tiny", chunk_seconds=2.0))
    wait_for(lambda: session.state == "recording-only", timeout=20.0)
    engine.stop_live(wait=True)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    errors = [e["message"] for e in events if e["type"] == "live.error"]
    assert len(errors) == 1, errors  # once, not once per chunk
    assert "--compute-device cpu" in errors[0] and "libcublas.so.12" in errors[0]
    meta = engine.library.get(session.meta.id)
    assert meta.status == "done" and engine.library.audio_path(meta) is not None


# --- CLI: one readable line, exit 1, no transcript ---------------------------

def test_cli_sigsegv_fails_with_one_line_and_writes_no_transcript(data_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "cpu")
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:always_crash_factory")
    wav = write_wav(tmp_path / "note.wav")
    assert cli.main(["transcribe", str(wav), "--model", "tiny"]) == 1
    captured = capsys.readouterr()
    assert "failed" in captured.err and "SIGSEGV" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out.strip() == ""  # no transcript text was emitted
    sessions = Library().list()
    assert [s.status for s in sessions] == ["error"]
    assert sessions[0].error and "SIGSEGV" in sessions[0].error
