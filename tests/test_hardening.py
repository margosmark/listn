from datetime import datetime
from pathlib import Path

import pytest

from listn.audio import TARGET_RATE
from listn.bundle import assemble, launcher_text
from listn.config import Settings
from listn.engine import Engine
from listn.iconart import render
from listn.ipc import PAUSE, RESUME, START, encode_frame, encode_json
from listn.library import Library, Segment
from listn.power import DiskSpaceError, model_for_power
from listn.retention import apply_retention

from .conftest import MAC_DEVICES, FakeSD
from .test_engine import wait_for
from .test_ingest import FakeClient


def test_low_battery_selects_the_small_model_and_skips_notes():
    settings = Settings(model="large-v3-turbo", summarize_after=True, battery_saver=True)
    assert model_for_power(settings, True, 25) == ("small", False)
    assert model_for_power(settings, True, 80) == ("large-v3-turbo", True)
    assert model_for_power(settings, False, 10) == ("large-v3-turbo", True)
    assert model_for_power(settings, True, None) == ("large-v3-turbo", True)
    settings.battery_saver = False
    assert model_for_power(settings, True, 5) == ("large-v3-turbo", True)


def test_recording_refuses_when_free_space_is_below_the_floor(data_dir, fake_factory):
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    with pytest.raises(DiskSpaceError):
        engine.start_live(Settings(device="Aggregate", min_free_gb=100000))
    assert engine.library.list() == []


def test_low_battery_is_applied_to_the_session_only(data_dir, fake_factory):
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    chosen = Settings(device="Aggregate", model="tiny", summarize_after=True, battery_saver=True, chunk_seconds=2.0)
    session = engine.start_live(chosen, on_battery=True, battery_percent=12)
    try:
        assert session.settings.model == "small"
        assert session.settings.summarize_after is False
        assert chosen.model == "tiny"
        assert chosen.summarize_after is True
    finally:
        engine.stop_live(wait=True)


def test_pause_leaves_a_marked_gap_and_keeps_the_timeline(data_dir, fake_factory):
    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=80.0), transcriber_factory=fake_factory)
    session = engine.start_live(Settings(device="Aggregate", model="tiny", chunk_seconds=2.0, summarize_after=False))
    try:
        wait_for(lambda: session._samples > TARGET_RATE)
        assert session.pause()
        samples = session._samples
        wait_for(lambda: session._samples > samples + TARGET_RATE)
        assert session.resume()
        gaps = [segment for segment in session.segments if segment.text == "[paused]"]
        assert gaps and gaps[0].end > gaps[0].start
    finally:
        engine.stop_live(wait=True)


def test_pause_and_resume_frames(data_dir, fake_factory):
    from listn.ingest import IngestServer

    engine = Engine(sd=FakeSD(MAC_DEVICES, speed=40.0), transcriber_factory=fake_factory)
    server = IngestServer(engine, port=8765)
    server.start()
    client = FakeClient(server.path)
    try:
        client.hello()
        client.send(encode_json(START, {"source": "device", "on_battery": True, "battery_percent": 10}))
        wait_for(lambda: engine.live_active())
        assert engine.live.settings.model == "small"
        client.send(encode_frame(PAUSE))
        wait_for(lambda: engine.live.paused)
        client.send(encode_frame(RESUME))
        wait_for(lambda: any(segment.text == "[paused]" for segment in engine.live.segments))
    finally:
        client.close()
        engine.stop_live(wait=True)
        server.stop()


def test_retention_deletes_old_audio_and_keeps_the_transcript(data_dir):
    library = Library()
    old = library.create("Old", "file", status="done")
    old.created_at = "2020-01-01T00:00:00"
    library.save_meta(old)
    audio = library.session_path(old.id) / "audio.wav"
    audio.write_bytes(b"audio")
    library.update(old.id, audio_file="audio.wav", status="done")
    library.save_segments(old.id, [Segment(0.0, 1.0, "keep this", "Others")])
    library.save_notes(old.id, {"summary": "notes stay"})

    recent = library.create("Recent", "file", status="done")
    (library.session_path(recent.id) / "audio.wav").write_bytes(b"audio")
    library.update(recent.id, audio_file="audio.wav", status="done")

    recording = library.create("Live", "live", status="recording")
    recording.created_at = "2020-01-01T00:00:00"
    library.save_meta(recording)
    (library.session_path(recording.id) / "audio.wav").write_bytes(b"audio")
    library.update(recording.id, audio_file="audio.wav", status="recording")

    removed = apply_retention(library, 30, now=datetime(2026, 6, 1, 12, 0, 0))
    assert old.id in removed
    assert not audio.exists()
    assert library.segments(old.id)[0].text == "keep this"
    assert library.notes(old.id)["summary"] == "notes stay"
    assert library.get(old.id).audio_removed is True
    assert library.get(old.id).audio_file is None
    assert (library.session_path(recent.id) / "audio.wav").exists()
    assert (library.session_path(recording.id) / "audio.wav").exists()
    assert apply_retention(library, 0) == []


def test_bundle_is_relocatable_and_copies_listn(tmp_path):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python3").write_text("#!/bin/sh\n")
    (prefix / "lib" / "python3.12").mkdir(parents=True)
    (prefix / "lib" / "python3.12" / "os.py").write_text("stdlib = True\n")
    (prefix / "lib" / "python3.12" / "test").mkdir()
    (prefix / "lib" / "python3.12" / "test" / "skip.py").write_text("nope\n")
    venv = tmp_path / "venv"
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {prefix}\n")
    site = venv / "lib" / "python3.12" / "site-packages"
    (site / "demo.py").write_text("x = 1\n")
    (site / "listn.pth").write_text(str(tmp_path / "checkout") + "\n")
    project = tmp_path / "project"
    (project / "listn").mkdir(parents=True)
    (project / "listn" / "__init__.py").write_text('__version__ = "test"\n')
    (project / "listn" / "web").mkdir()
    (project / "listn" / "web" / "index.html").write_text("<p>ok</p>")

    dest = tmp_path / "python"
    report = assemble(venv, project, dest)
    launcher = Path(report["launcher"]).read_text()
    assert launcher == launcher_text()
    assert str(project) not in launcher
    assert "PYTHONHOME" in launcher
    assert (dest / "site-packages" / "listn" / "__init__.py").read_text().startswith("__version__")
    assert (dest / "site-packages" / "listn" / "web" / "index.html").exists()
    assert (dest / "site-packages" / "demo.py").exists()
    assert not (dest / "site-packages" / "listn.pth").exists()
    assert (dest / "prefix" / "lib" / "python3.12" / "os.py").exists()
    assert not (dest / "prefix" / "lib" / "python3.12" / "test").exists()
    assert (dest / "prefix" / "bin" / "python3").stat().st_size > 0
    assert not (dest / "prefix" / "bin" / "python3").is_symlink()
    assert "models not included" in (dest / "SIZE.txt").read_text() or "not included" in (dest / "SIZE.txt").read_text()


def test_icon_is_a_png():
    png = render(16)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 32


def test_a_dropped_input_device_is_reopened(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    engine.start_live(Settings(device="Aggregate", model="tiny", chunk_seconds=2.0, summarize_after=False))
    try:
        wait_for(lambda: sd.opened)
        sd.opened[0].active = False
        wait_for(lambda: len(sd.opened) >= 2, timeout=5)
    finally:
        engine.stop_live(wait=True)
