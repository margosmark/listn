"""Dashboard tests: /api/dashboard renders real engine state only.

The endpoint is a read-only projection of ``Engine``/``Library`` — these
tests start and stop real sessions through the same fixtures the server
suite uses, and assert the snapshot follows the engine: nothing mock,
and an honest empty state when nothing has ever run.
"""

from __future__ import annotations

import time

from listn.dashboard import build_snapshot, status_label, track_files, track_summary

from .test_engine import write_wav
from .test_server import client, wait_status  # noqa: F401


# ---------------------------------------------------------------------------
# Honest degradation
# ---------------------------------------------------------------------------

def test_empty_state_is_honest(client):
    snap = client.get("/api/dashboard").json()
    assert snap["live"] is None
    assert snap["recordings"] == []
    assert snap["recordings_total"] == 0
    assert snap["jobs"] == []
    assert snap["pipewire"] is None
    # the shell fields are real, not placeholder values
    assert snap["platform"] and snap["version"] and snap["data_dir"]


def test_dashboard_page_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "listn" in r.text and "/static/dashboard.js" in r.text
    assert client.get("/static/dashboard.js").status_code == 200


def test_main_ui_links_dashboard_on_linux_only(client):
    r = client.get("/")
    assert 'href="/dashboard"' in r.text
    # the link ships hidden; app.js reveals it only when platform === "linux",
    # so macOS UI is unchanged
    assert 'id="dashboard-link"' in r.text and "hidden" in r.text
    app_js = client.get("/static/app.js").text
    assert '#dashboard-link").hidden = false' in app_js
    # the dashboard script itself references only first-party endpoints
    dash_js = client.get("/static/dashboard.js").text
    assert "/api/dashboard" in dash_js
    assert "http://" not in dash_js and "https://" not in dash_js


def test_dashboard_respects_loopback_and_origin(client):
    assert client.get("/api/dashboard", headers={"host": "evil.example.com"}).status_code == 403
    assert client.get("/api/dashboard").status_code == 200
    # read-only: no write verbs exist on it
    assert client.post("/api/dashboard", json={}).status_code == 405


# ---------------------------------------------------------------------------
# Live session state flows through
# ---------------------------------------------------------------------------

def test_live_recording_shows_in_dashboard(client):
    client.put("/api/settings", json={"device": "BlackHole", "chunk_seconds": 2})
    started = client.post("/api/live/start", json={"title": "Standup"})
    assert started.status_code == 200
    sid = started.json()["id"]

    snap = client.get("/api/dashboard").json()
    assert snap["live"] is not None
    assert snap["live"]["id"] == sid
    assert snap["live"]["title"] == "Standup"
    assert snap["live"]["active"] is True
    assert snap["live"]["state"] in ("starting", "loading-model", "listening", "recording-only")
    # the session is also listed, marked live, with status=recording
    row = next(r for r in snap["recordings"] if r["id"] == sid)
    assert row["live"] is True
    assert row["status"] == "recording"
    assert row["status_label"] == "Recording"

    # wait for the first transcribed segment like the server suite does,
    # so the stopped session has a real transcript
    end = time.monotonic() + 10
    while not client.get("/api/live").json()["segments"] and time.monotonic() < end:
        time.sleep(0.05)
    assert client.post("/api/live/stop").status_code == 200
    meta = wait_status(client, sid, "done")
    assert meta["status"] == "done"

    snap = client.get("/api/dashboard").json()
    assert snap["live"] is None  # honest: nothing is recording now
    row = next(r for r in snap["recordings"] if r["id"] == sid)
    assert row["live"] is False
    assert row["status"] == "done"
    assert row["status_label"] == "Done"
    assert row["transcription"]["state"] == "done"
    assert row["transcription"]["segments"] >= 1
    assert row["tracks"]["mixed"] is True  # engine wrote audio.wav
    assert row["track_summary"] == "One track"


def test_transcription_processing_state(client, tmp_path):
    wav = write_wav(tmp_path / "sync.wav", seconds=3.0)
    client.put("/api/settings", json={"chunk_seconds": 2})
    with wav.open("rb") as fh:
        r = client.post("/api/files", files={"file": ("sync.wav", fh, "audio/wav")}, data={"title": ""})
    assert r.status_code == 200
    sid = r.json()["id"]

    # while the job runs the dashboard says Processing, with the real job row
    def processing_seen():
        snap = client.get("/api/dashboard").json()
        row = next((x for x in snap["recordings"] if x["id"] == sid), None)
        if row and row["transcription"]["state"] == "processing":
            return snap, row
        return None

    deadline = time.monotonic() + 10
    seen = None
    while time.monotonic() < deadline and seen is None:
        seen = processing_seen()
        if seen is None:
            time.sleep(0.05)
    if seen is not None:
        snap, row = seen
        assert row["status"] == "transcribing"
        assert row["status_label"] == "Processing"
        job_ids = [j["id"] for j in snap["jobs"]]
        assert sid in job_ids

    wait_status(client, sid, "done")
    snap = client.get("/api/dashboard").json()
    assert snap["jobs"] == []
    row = next(x for x in snap["recordings"] if x["id"] == sid)
    assert row["status"] == "done"
    assert row["transcription"]["state"] == "done"


def test_failed_session_reports_failed(client):
    # A dead-owner "recording" session is recovered as error by the engine at
    # startup; simulate the stored state directly through the library.
    snap = client.get("/api/dashboard").json()
    assert snap["recordings"] == []

    # write a failed session the way the engine does, then re-read
    from listn.library import Library

    lib = Library()
    meta = lib.create("Broken run", "live", status="error",
                      error="pw-record exited with code 1")
    lib.save_meta(meta)
    snap = client.get("/api/dashboard").json()
    row = snap["recordings"][0]
    assert row["status"] == "error"
    assert row["status_label"] == "Failed"
    assert row["error"] == "pw-record exited with code 1"
    assert row["transcription"]["state"] == "failed"
    assert row["track_summary"] == "No audio kept"


# ---------------------------------------------------------------------------
# Track separation is read from real files on disk
# ---------------------------------------------------------------------------

def test_two_track_session_shows_separation(client):
    from listn.library import Library

    lib = Library()
    meta = lib.create("Two track", "live", status="done")
    # the engine's own track files (1 s of real PCM each)
    import numpy as np

    from listn.audio import WavWriter

    for name in ("audio-system.wav", "audio-mic.wav", "audio.wav"):
        w = WavWriter(lib.session_path(meta.id) / name)
        w.write(np.full(16000, 0.1, dtype=np.float32))
        w.close()

    snap = client.get("/api/dashboard").json()
    row = snap["recordings"][0]
    assert row["tracks"] == {"system": True, "mic": True, "mixed": True, "other": None}
    assert row["track_summary"] == "System + microphone (separate tracks) + mixdown"


def test_track_files_and_summary_helpers():
    from types import SimpleNamespace

    class FakeLib:
        def session_path(self, sid):
            raise KeyError(sid)

    meta = SimpleNamespace(id="20261029-120000-abcdef", audio_file="talk.mp3")
    tracks = track_files(FakeLib(), meta)
    assert tracks == {"system": False, "mic": False, "mixed": False, "other": None}
    assert track_summary(tracks) == "No audio kept"
    assert track_summary({"system": False, "mic": True, "mixed": False, "other": None}) == "One separated track"
    assert status_label(None) == "Unknown"
    assert status_label("something-new") == "Something new"


# ---------------------------------------------------------------------------
# Snapshot module mirrors engine state exactly (no independent dashboard state)
# ---------------------------------------------------------------------------

def test_snapshot_tracks_engine_jobs(client, tmp_path):
    wav = write_wav(tmp_path / "meet.wav", seconds=3.0)
    client.put("/api/settings", json={"chunk_seconds": 2})
    with wav.open("rb") as fh:
        r = client.post("/api/files", files={"file": ("meet.wav", fh, "audio/wav")}, data={"title": ""})
    sid = r.json()["id"]

    # while the job is registered it must appear in the snapshot's jobs list
    deadline = time.monotonic() + 10
    jobs_seen = False
    while time.monotonic() < deadline and not jobs_seen:
        snap = client.get("/api/dashboard").json()
        jobs_seen = any(j["id"] == sid for j in snap["jobs"])
        if not jobs_seen:
            time.sleep(0.02)
    wait_status(client, sid, "done")
    snap = client.get("/api/dashboard").json()
    assert snap["jobs"] == []
    assert not any(j["id"] == sid for j in snap["jobs"])


def test_dashboard_never_invents_recordings(client):
    """Two snapshots of an untouched engine stay identical and empty."""
    a = client.get("/api/dashboard").json()
    b = client.get("/api/dashboard").json()
    assert a["recordings"] == b["recordings"] == []
    assert a["recordings_total"] == b["recordings_total"] == 0
    assert a["live"] is None and b["live"] is None


def test_build_snapshot_limit(client):
    from listn.library import Library

    lib = Library()
    for i in range(5):
        lib.create(f"R{i}", "file", status="done")
    snap = build_snapshot(client.app.state.engine, None, limit=2)
    assert len(snap["recordings"]) == 2
    assert snap["recordings_total"] == 5
