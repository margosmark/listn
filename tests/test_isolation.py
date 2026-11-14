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
import threading
import time

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


def test_explicit_cuda_without_a_device_fails_loudly_before_any_build(data_dir):
    """Contract item 5, isolated path: the probe's refusal crosses the pipe
    as what it is (CudaUnavailable), not demoted to a bare RuntimeError —
    one actionable line, no CPU attempt, nothing cached.
    """
    with pytest.raises(CudaUnavailable) as excinfo:
        tr.get_transcriber("tiny", "faster-whisper",
                           compute_device="cuda", isolate=True)
    message = str(excinfo.value)
    assert "compute_device=cuda" in message
    assert "--compute-device cpu" in message
    # No model was built and no fallback happened: the build never started.
    assert tr._cache == {}
    assert tr.CPU_FALLBACK_MESSAGE not in message


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

# --- S1: bounded worker requests --------------------------------------------
# The security review executed a child that never answers: transcribe(), the
# load path and close() all blocked forever, recovery only via an external
# kill, and the engine's single job-worker thread then wedged every later
# file job. Every request now carries a silence-based deadline, close()'s
# lock wait is bounded, and the expiry stops the child itself.

def _worker_alive(pid: int) -> bool:
    """True while the worker process still exists (zombies count as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _spy_worker_pids(monkeypatch) -> list:
    """Record every worker child pid this test spawns (orphan evidence)."""
    pids = []
    original = tr._WorkerSession.spawn.__func__

    def spawn(cls):
        session = original(cls)
        pids.append(session.proc.pid)
        return session

    monkeypatch.setattr(tr._WorkerSession, "spawn", classmethod(spawn))
    return pids


def test_hung_worker_at_load_fails_within_the_bound_and_leaves_no_orphan(
        monkeypatch, data_dir):
    """A child that never answers the load: the parent returns within the
    bound with a clean error, and the worker is gone — not orphaned."""
    monkeypatch.setattr(tr, "_LOAD_IDLE_TIMEOUT", 10.0)
    pids = _spy_worker_pids(monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:hang_load_factory")
    start = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        tr.get_transcriber("tiny", "faster-whisper", compute_device="cpu")
    elapsed = time.monotonic() - start
    message = str(excinfo.value)
    assert elapsed < 30.0  # bounded: the old code blocked forever
    assert "stopped responding" in message
    assert "loading the model" in message
    assert "Traceback" not in message
    assert tr._cache == {}  # nothing half-built was cached
    assert pids, "the test must have spawned a worker"
    assert all(not _worker_alive(pid) for pid in pids)  # no orphan worker


def test_hung_worker_at_transcribe_fails_within_the_bound_and_leaves_no_orphan(
        monkeypatch, data_dir):
    """A child that loads fine but never answers the transcription: same
    bound, same clean teardown, and the wedged session is dropped."""
    monkeypatch.setattr(tr, "_REQUEST_IDLE_TIMEOUT", 3.0)
    pids = _spy_worker_pids(monkeypatch)
    wrapper = _iso("hang_transcribe_factory", "cpu", monkeypatch, data_dir)
    start = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        wrapper.transcribe(_audio())
    elapsed = time.monotonic() - start
    message = str(excinfo.value)
    assert elapsed < 30.0
    assert "stopped responding" in message
    assert "while transcribing" in message
    assert "Traceback" not in message
    assert wrapper._session is None  # the wedged session was torn down
    assert pids and all(not _worker_alive(pid) for pid in pids)


def test_close_is_bounded_while_a_request_is_wedged(monkeypatch, data_dir):
    """close() must not queue behind a wedged request: it waits at most
    _CLOSE_LOCK_TIMEOUT, then stops the worker anyway — which unblocks the
    wedged request with a clean error instead of an external pkill."""
    monkeypatch.setattr(tr, "_REQUEST_IDLE_TIMEOUT", 60.0)  # keep it wedged
    monkeypatch.setattr(tr, "_CLOSE_LOCK_TIMEOUT", 1.0)
    pids = _spy_worker_pids(monkeypatch)
    wrapper = _iso("hang_transcribe_factory", "cpu", monkeypatch, data_dir)
    errors = []
    done = threading.Event()

    def wedge():
        try:
            wrapper.transcribe(_audio())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=wedge, daemon=True).start()
    deadline = time.monotonic() + 10.0
    while not wrapper._lock.locked() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert wrapper._lock.locked(), "the transcribe request never took the lock"

    start = time.monotonic()
    wrapper.close()
    assert time.monotonic() - start < 10.0  # bounded lock wait, then a stop
    assert done.wait(15.0)  # the wedged request fails, it does not hang
    assert errors, "the wedged transcribe should have been unblocked"
    assert "The transcription process" in str(errors[0])
    assert all(not _worker_alive(pid) for pid in pids)


# --- S2/S3: the child confirms the decision, never widens it -----------------

def test_child_gpu_claim_on_a_cpu_request_is_clamped_to_cpu(monkeypatch, data_dir):
    """Requested cpu; the child claims cuda in its device, its device line
    AND a status event (the review's executed probe). None of it may reach
    the user: the job still runs, on the CPU the parent asked for."""
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:gpu_claim_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="cpu", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["ran where it was told"]  # the job still ran
    assert wrapper.device == "cpu"  # the claim was not adopted
    assert wrapper.device_message == "device: cpu"
    assert wrapper.status_note is None  # nothing fell back
    assert not any("cuda" in s.lower() or "gpu" in s.lower()
                   for s in statuses)  # the callback saw no GPU claim either
    wrapper.close()


def test_child_reclaim_after_a_cpu_pin_never_overwrites_the_parent(
        monkeypatch, data_dir, tmp_path):
    """auto pins CPU after the child dies on the GPU attempt; the retry
    child claims cuda (device + message). The pin must survive it: the job
    still runs on CPU and the announcement keeps the SIGSEGV explanation."""
    log = _load_log(tmp_path, monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH",
                        f"{FACTORIES}:reclaim_after_pin_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["cpu work"]  # still ran — on the pinned CPU
    assert _loads(log) == 2  # one GPU attempt (died), one CPU retry
    assert wrapper.device == "cpu"  # the post-pin re-claim was not adopted
    assert wrapper.device_message.startswith("device: cpu")
    assert "device: cuda" not in wrapper.device_message
    assert "SIGSEGV" in (wrapper.status_note or "")  # parent's note stands
    assert not any("cuda" in s.lower() or "gpu acceleration" in s.lower()
                   for s in statuses)
    wrapper.close()


def test_child_fallback_note_is_not_adopted_on_a_cpu_request(
        monkeypatch, data_dir):
    """A consistent cpu claim carrying the canonical fallback note and a GPU
    device line: a request that never fell back must announce neither."""
    monkeypatch.setattr(tr, "_FACTORY_PATH",
                        f"{FACTORIES}:injected_note_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="cpu", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["on cpu"]
    assert wrapper.status_note is None  # nothing fell back
    assert wrapper.device_message == "device: cpu"
    assert tr.CPU_FALLBACK_MESSAGE not in statuses
    wrapper.close()


def test_child_canonical_note_for_a_narrowed_auto_build_is_adopted(
        monkeypatch, data_dir):
    """The control for S2/S3/S2r: an honest auto->CPU narrowing keeps its
    note and its message exactly as the probe reported them.

    S2r: the note is adopted only when THIS parent's own probe narrowed
    auto->cpu too, so the parent-side probe seams are pinned to the
    visible-but-broken-GPU machine that produces a narrowing.
    """
    _narrowed_gpu_probe(monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH",
                        f"{FACTORIES}:narrowed_cpu_note_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    assert wrapper.device == "cpu"
    assert wrapper.status_note == tr.CPU_FALLBACK_MESSAGE
    assert wrapper.device_message == "device: cpu"
    assert statuses == [tr.CPU_FALLBACK_MESSAGE]
    wrapper.close()


def test_kind_device_runtime_fault_is_not_relabelled_as_cpu_fallback(
        monkeypatch, data_dir):
    """S3: kind='device' carrying an OOM must propagate as itself — no CPU
    rebuild, no canonical 'CUDA libraries unavailable' announcement."""
    monkeypatch.setattr(tr, "_FACTORY_PATH",
                        f"{FACTORIES}:kind_device_oom_factory")
    statuses = []
    with pytest.raises(RuntimeError) as excinfo:
        tr.get_transcriber("tiny", "faster-whisper",
                           compute_device="auto", status=statuses.append)
    assert type(excinfo.value) is RuntimeError  # not a CudaUnavailable
    assert "out of memory" in str(excinfo.value)
    assert statuses == []  # nothing was announced as a CPU fallback
    assert tr._cache == {}  # no fallback build, nothing cached


# --- S1r: the send path is bounded too ---------------------------------------

def _wait_until_stopped(pid: int, timeout: float = 10.0) -> bool:
    """True once the child has SIGSTOPped itself (Linux /proc state)."""
    status_path = f"/proc/{pid}/status"
    if not os.path.exists(status_path):
        time.sleep(2.5)  # no /proc (macOS): trust the factory's 1.5 s timer
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(status_path) as fh:
                state = fh.read().split("State:")[1].split("\n")[0].strip()
        except (OSError, IndexError):
            return False
        if state.startswith("T"):
            return True
        time.sleep(0.05)
    return False


def test_send_path_is_bounded_when_child_stops_reading(monkeypatch, data_dir):
    """S1r: a child that stops reading the request pipe OUTSIDE an active
    request (SIGSTOP after load) must not wedge the caller in os.write on
    a payload larger than the pipe buffer: the send is bounded too — one
    clean error, the lock released, a second job not stuck behind it, and
    no orphan worker left behind."""
    monkeypatch.setattr(tr, "_REQUEST_IDLE_TIMEOUT", 3.0)
    pids = _spy_worker_pids(monkeypatch)
    wrapper = _iso("freeze_between_factory", "cpu", monkeypatch, data_dir)
    assert _wait_until_stopped(pids[-1]), "the child never froze between requests"
    big = np.full(16000 * 16, 0.1, dtype=np.float32)  # 1 MB ≫ 64 KB pipe
    start = time.monotonic()
    errors = []
    try:
        wrapper.transcribe(big, language="en")
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)
    elapsed = time.monotonic() - start
    message = str(errors[0]) if errors else ""
    assert elapsed < 12.0  # bounded: the old code blocked in os.write forever
    assert errors, "the wedged send must fail, not return segments"
    assert "stopped responding" in message
    assert "while transcribing" in message
    assert "Traceback" not in message
    assert not wrapper._lock.locked()  # the caller released the lock
    assert wrapper._session is None    # the wedged session was torn down
    # a second job must not queue behind the wedged one
    start2 = time.monotonic()
    with pytest.raises(RuntimeError):
        wrapper.transcribe(_audio(), language="en")
    assert time.monotonic() - start2 < 5.0
    wrapper.close()
    assert pids and all(not _worker_alive(pid) for pid in pids)  # no orphan


# --- S2r: on `auto` the child is not the device authority ---------------------

def _no_gpu_probe(monkeypatch) -> None:
    """Pin the parent-side probe seams to a machine with no usable GPU."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(
        tr.compute_backend, "rocm_capability",
        lambda: tr.compute_backend.Capability(
            "rocm", False, "no AMD GPU and no ROCm runtime are visible "
            "on this machine", "use --compute-device cpu or auto"))


def _narrowed_gpu_probe(monkeypatch) -> None:
    """Pin the parent-side seams to a visible-but-broken CUDA GPU: auto
    narrows to CPU with the canonical reason (the CachyOS laptop)."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        tr, "cuda_runtime_ready",
        lambda *a, **k: (False,
                         "libcublas.so.12: cannot open shared object file"))
    monkeypatch.setattr(
        tr.compute_backend, "rocm_capability",
        lambda: tr.compute_backend.Capability(
            "rocm", False, "no AMD GPU and no ROCm runtime are visible "
            "on this machine", "use --compute-device cpu or auto"))


def test_auto_hostile_gpu_claim_is_dropped_without_a_usable_parent_gpu(
        monkeypatch, data_dir):
    """S2r/C5: on the default device (auto) a child claiming cuda — in its
    device, its device line AND a status event — has all of it dropped when
    this parent's own probe sees no usable GPU. The job still runs."""
    _no_gpu_probe(monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:gpu_claim_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["ran where it was told"]  # the job still ran
    assert wrapper.device == "auto"  # claim dropped: the parent's decision stands
    assert "cuda" not in wrapper.device_message.lower()
    assert wrapper.status_note is None
    assert not any("cuda" in s.lower() or "gpu" in s.lower()
                   for s in statuses)  # neither status event nor announcement
    wrapper.close()


def test_auto_fabricated_fallback_note_is_dropped_without_parent_narrowing(
        monkeypatch, data_dir):
    """S2r/C6: a child fabricating the canonical CPU-fallback note — as
    the response's status_note AND as a status event — on a host whose own
    probe did NOT narrow auto->cpu has both channels dropped. The job runs."""
    _no_gpu_probe(monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:auto_note_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["on cpu"]
    assert wrapper.status_note is None               # fabricated note dropped
    assert tr.CPU_FALLBACK_MESSAGE not in statuses   # and never announced
    wrapper.close()


def test_auto_hostile_status_event_is_dropped_without_a_usable_parent_gpu(
        monkeypatch, data_dir):
    """S2r/C7: a hostile status event alone (the response itself claims
    nothing) must not reach the status callback under auto on a host whose
    parent-side probe sees no usable GPU."""
    _no_gpu_probe(monkeypatch)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:auto_status_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["ok"]
    assert wrapper.device == "auto"  # the response itself claimed nothing
    assert not any("cuda" in s.lower() or "gpu" in s.lower()
                   for s in statuses)
    wrapper.close()


def test_auto_gpu_claim_is_adopted_when_the_parent_probe_sees_a_usable_gpu(
        monkeypatch, data_dir):
    """The other side of S2r: under auto a cuda claim is ADOPTED when this
    parent's own probe corroborates a usable GPU — the cross-check must not
    blanket-ban GPU claims."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 2)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:gpu_claim_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="auto", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["ran where it was told"]
    assert wrapper.device == "cuda"  # corroborated by the parent's own probe
    assert "cuda" in wrapper.device_message.lower()
    wrapper.close()


def test_cpu_request_never_runs_the_parent_auto_probe(monkeypatch, data_dir):
    """S2r blast guard: the parent-side auto probe must never fire for an
    explicit `cpu` ask — `cpu` never probes (it could only ever corroborate
    a GPU claim, which cpu never accepts)."""
    def boom(*_args, **_kwargs):
        raise AssertionError("the parent must not probe for compute_device=cpu")
    monkeypatch.setattr(tr, "cuda_device_count", boom)
    monkeypatch.setattr(tr, "cuda_runtime_ready", boom)
    monkeypatch.setattr(tr.compute_backend, "rocm_capability", boom)
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:gpu_claim_factory")
    statuses = []
    wrapper = tr.get_transcriber("tiny", "faster-whisper",
                                 compute_device="cpu", status=statuses.append)
    segments = [s.text for s in wrapper.transcribe(_audio(), language="en")]
    assert segments == ["ran where it was told"]
    assert wrapper.device == "cpu"
    assert not any("cuda" in s.lower() or "gpu" in s.lower() for s in statuses)
    wrapper.close()

