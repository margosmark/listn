import time
import wave

import numpy as np
import pytest

from listn.audio import to_int16
from listn.config import Settings
from listn.engine import Engine

from .conftest import MAC_DEVICES, FakeSD, join_stray_finishers


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out")


def write_wav(path, seconds=3.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(to_int16(0.3 * np.sin(2 * np.pi * 220 * t)).tobytes())
    return path


@pytest.fixture
def engine(data_dir, fake_factory):
    return Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)


def test_live_session_records_transcribes_and_saves(engine):
    q = engine.bus.subscribe()
    settings = Settings(device="Aggregate", model="tiny", chunk_seconds=2.0)
    session = engine.start_live(settings, "Team call")
    assert session.device_info["channels"] == 3
    wait_for(lambda: len(session.segments) >= 2)
    engine.stop_live(wait=True)

    meta = engine.library.get(session.meta.id)
    assert meta.status == "done" and meta.source_name == "Aggregate Device"
    assert meta.duration > 2.5
    segs = engine.library.segments(meta.id)
    assert segs and all("Thanks for watching" not in s.text for s in segs)
    assert segs == sorted(segs, key=lambda s: s.start)
    assert segs[1].start >= segs[0].end - 0.01
    with wave.open(str(engine.library.audio_path(meta)), "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        assert wf.getnframes() / 16000 == pytest.approx(meta.duration, abs=0.05)

    kinds = set()
    while not q.empty():
        kinds.add(q.get_nowait()["type"])
    assert {"live.started", "live.segments", "live.level", "live.stopped"} <= kinds


def test_only_one_live_session(engine):
    engine.start_live(Settings(device="BlackHole"))
    with pytest.raises(RuntimeError, match="already"):
        engine.start_live(Settings(device="BlackHole"))
    engine.stop_live(wait=True)


def test_live_without_saving_audio(engine):
    s = engine.start_live(Settings(device="BlackHole", save_audio=False, chunk_seconds=2.0))
    wait_for(lambda: s.segments)
    engine.stop_live(wait=True)
    assert engine.library.audio_path(engine.library.get(s.meta.id)) is None


def test_bad_device_marks_session_error(engine):
    with pytest.raises(Exception, match="No input device"):
        engine.start_live(Settings(device="Nope"))
    errored = [m for m in engine.library.list() if m.status == "error"]
    assert len(errored) == 1
    assert not engine.live_active()


def test_model_load_failure_keeps_recording(data_dir):
    def broken(model, backend, status=None):
        raise RuntimeError("model missing")

    eng = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=broken)
    s = eng.start_live(Settings(device="BlackHole", chunk_seconds=2.0))
    wait_for(lambda: s.state == "recording-only")
    time.sleep(0.2)
    eng.stop_live(wait=True)
    meta = eng.library.get(s.meta.id)
    assert meta.status == "done" and meta.duration > 0
    assert eng.library.audio_path(meta) is not None


def test_file_job_transcribes_and_reports_progress(engine, tmp_path):
    src = write_wav(tmp_path / "interview.wav")
    q = engine.bus.subscribe()
    job = engine.enqueue_file(src, "Interview", Settings(model="tiny"), diarize_speakers=False)
    events = []
    while True:
        ev = q.get(timeout=10)
        events.append(ev)
        if ev["type"] == "job.done":
            break
    assert ev["error"] is None
    stages = [e["job"]["stage"] for e in events if e["type"] == "job.update"]
    assert "decoding" in stages and "transcribing" in stages
    meta = engine.library.get(job.meta.id)
    assert meta.status == "done" and meta.duration == pytest.approx(3.0, abs=0.05)
    assert [s.text for s in engine.library.segments(meta.id)] == ["chunk 1 first half"]
    assert src.exists()


def test_file_job_waits_for_live_and_can_be_cancelled(engine, tmp_path):
    engine.start_live(Settings(device="BlackHole"))
    job = engine.enqueue_file(write_wav(tmp_path / "a.wav"), "A", Settings(), False)
    wait_for(lambda: job.stage == "waiting")
    engine.cancel_job(job.meta.id)
    wait_for(lambda: engine.library.get(job.meta.id).status == "error")
    assert engine.library.get(job.meta.id).error == "Cancelled."
    engine.stop_live(wait=True)


def test_file_job_bad_file_reports_error(engine, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"\x00" * 100)
    job = engine.enqueue_file(bad, "Broken", Settings(), False)
    wait_for(lambda: engine.library.get(job.meta.id).status == "error")
    assert engine.library.get(job.meta.id).error


def test_file_job_completes_when_cuda_libraries_are_missing(tmp_path, data_dir, monkeypatch):
    """Mark's box: a GPU CTranslate2 can see, CUDA user-space libraries absent.

    The job must finish on CPU with the fallback in its status stages — not
    die with ``libcublas.so.12 is not found`` — and rebuild exactly once.

    Seam port (contract item 2): the model now loads inside the isolation
    child, so this file's ``sys.modules`` patch can no longer reach it. The
    same fault travels through ``transcriber._FACTORY_PATH`` — the child
    builds ``cuda_fault_factory`` instead of a model, and the ``auto`` build
    still dies with the libcublas error while the ``cpu`` build works.
    Assertions unchanged; only the build evidence moved from the fake
    model's class list to the log the child writes.
    """
    import json

    from listn import transcriber as tr
    from .test_transcriber import _CUBLAS_ERROR, _FAULT_ENV, _FAULT_FACTORY, _FAULT_LOG_ENV

    log = tmp_path / "cuda_fault_builds.log"
    monkeypatch.setenv(_FAULT_LOG_ENV, str(log))
    monkeypatch.setenv(_FAULT_ENV, json.dumps({
        "error": _CUBLAS_ERROR, "fail_at_init": False, "devices": ["auto"]}))
    monkeypatch.setattr(tr, "_FACTORY_PATH", _FAULT_FACTORY)
    tr._cache.clear()
    try:
        eng = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0))   # real get_transcriber
        q = eng.bus.subscribe()
        job = eng.enqueue_file(write_wav(tmp_path / "call.wav"), "Call",
                               Settings(model="small"), diarize_speakers=False)
        events = []
        while True:
            ev = q.get(timeout=15)
            events.append(ev)
            if ev["type"] == "job.done":
                break
        assert ev["error"] is None
        meta = eng.library.get(job.meta.id)
        assert meta.status == "done"
        assert [s.text for s in eng.library.segments(meta.id)] == ["hello"]
        stages = [e["job"]["stage"] for e in events if e["type"] == "job.update"]
        assert tr.CPU_FALLBACK_MESSAGE in stages
        builds = [json.loads(line)["device"]
                  for line in log.read_text().splitlines() if line.strip()]
        assert builds == ["auto", "cpu"]
    finally:
        tr._cache.clear()


def test_second_stop_waits_for_the_finisher_the_first_one_started(engine):
    """``stop(wait=True)`` after a ``stop(wait=False)`` must join listn-finish.

    ``/api/live/stop`` stops with ``wait=False`` on purpose, so the app's own
    shutdown finds a session that is already stopping and calls
    ``stop(wait=True)`` on it. That call used to hit
    ``if self._stop.is_set(): return``, return in 0.0000s having joined
    nothing, and let uvicorn exit while the *daemon* ``listn-finish`` was
    still writing transcript.json/meta.json (t_d0ba73dd).

    The final save is slowed down so the finisher is provably still running
    when the second stop arrives — the situation a big transcript on a busy
    disk creates in production.
    """
    session = engine.start_live(Settings(device="BlackHole", save_audio=False, chunk_seconds=2.0))
    wait_for(lambda: session.segments)
    real_save = engine.library.save_meta

    def slow_save(meta, *args, **kwargs):
        time.sleep(1.2)
        return real_save(meta, *args, **kwargs)

    engine.library.save_meta = slow_save
    try:
        engine.stop_live(wait=False)          # what /api/live/stop does
        assert session.state == "finishing"

        t0 = time.monotonic()
        engine.stop_live(wait=True)           # what the shutdown path does
        waited = time.monotonic() - t0
        assert waited >= 1.0, (
            f"stop(wait=True) returned after {waited:.4f}s while "
            f"listn-finish was still saving (state={session.state})")
        assert session.state == "done"
        finisher = getattr(session, "_finisher", None)
        assert finisher is not None, "stop() keeps no handle on the finisher it spawned"
        assert not finisher.is_alive()
        meta = engine.library.get(session.meta.id)
        assert meta.status == "done"
        assert (engine.library.session_path(session.meta.id) / "transcript.json").exists()

        # Idempotent: a third stop neither re-runs the shutdown nor hangs.
        t0 = time.monotonic()
        engine.stop_live(wait=True)
        assert time.monotonic() - t0 < 0.5
        assert session.state == "done"
    finally:
        engine.library.save_meta = real_save
        join_stray_finishers()
