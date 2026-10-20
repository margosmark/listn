import time

import pytest
from fastapi.testclient import TestClient

from listn.engine import Engine
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD
from .test_engine import write_wav


@pytest.fixture
def client(data_dir, fake_factory):
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


def test_rejects_non_loopback_host(client):
    r = client.get("/api/status", headers={"host": "evil.example.com"})
    assert r.status_code == 403
    assert client.get("/api/status", headers={"host": "localhost:8765"}).status_code == 200


def test_rejects_cross_origin_writes(client):
    r = client.put("/api/settings", json={"model": "base"}, headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403
    ok = client.put("/api/settings", json={"model": "base"}, headers={"origin": "http://127.0.0.1:8765"})
    assert ok.status_code == 200


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "listn" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_status_and_devices(client):
    st = client.get("/api/status").json()
    assert "small" in st["models"] and st["live"] is None
    devices = client.get("/api/devices").json()
    assert devices["selected"] == "BlackHole 2ch"
    assert devices["default_output"] == "Multi-Output Device"
    assert devices["guidance"][0]["level"] == "ok"


def test_settings_validation_and_secret_hidden(client):
    assert client.put("/api/settings", json={"model": "gigantic"}).status_code == 400
    r = client.put("/api/settings", json={"hf_token": "hf_abc", "language": "DE"}).json()
    assert r["hf_token"] is True and r["language"] == "de"


def test_device_test_endpoint(client):
    r = client.post("/api/devices/test", json={"device": "BlackHole", "seconds": 0.5})
    assert r.status_code == 200 and r.json()["verdict"] == "ok"
    assert client.post("/api/devices/test", json={"device": "Nope"}).status_code == 400


def wait_status(client, sid, want, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        meta = client.get(f"/api/sessions/{sid}").json()["meta"]
        if meta["status"] == want:
            return meta
        time.sleep(0.05)
    raise AssertionError(f"session never reached {want}")


def test_live_start_stop_flow(client):
    client.put("/api/settings", json={"device": "BlackHole", "chunk_seconds": 2})
    started = client.post("/api/live/start", json={"title": "Sync"})
    assert started.status_code == 200
    sid = started.json()["id"]
    assert client.post("/api/live/start", json={}).status_code == 409
    end = time.monotonic() + 10
    while not client.get("/api/live").json()["segments"] and time.monotonic() < end:
        time.sleep(0.05)
    assert client.post("/api/live/stop").status_code == 200
    meta = wait_status(client, sid, "done")
    assert meta["title"] == "Sync"
    assert client.get(f"/api/sessions/{sid}/audio").status_code == 200


def test_upload_transcribe_export_search_delete(client, tmp_path):
    wav = write_wav(tmp_path / "standup.wav")
    with wav.open("rb") as fh:
        r = client.post("/api/files", files={"file": ("standup.wav", fh, "audio/wav")}, data={"title": ""})
    assert r.status_code == 200
    sid = r.json()["id"]
    meta = wait_status(client, sid, "done")
    assert meta["title"] == "standup" and meta["source_name"] == "standup.wav"

    srt = client.get(f"/api/sessions/{sid}/export", params={"format": "srt"})
    assert srt.status_code == 200 and "-->" in srt.text
    assert 'filename="standup.srt"' in srt.headers["content-disposition"]
    assert client.get(f"/api/sessions/{sid}/export", params={"format": "exe"}).status_code == 400

    hits = client.get("/api/sessions", params={"q": "first half"}).json()["results"]
    assert hits[0]["meta"]["id"] == sid

    renamed = client.patch(f"/api/sessions/{sid}", json={"title": "Daily standup"}).json()
    assert renamed["title"] == "Daily standup"

    assert client.post(f"/api/sessions/{sid}/retranscribe", json={}).status_code == 200
    wait_status(client, sid, "done")

    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_upload_rejects_unknown_type(client):
    r = client.post("/api/files", files={"file": ("notes.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 400


def test_bad_session_ids(client):
    assert client.get("/api/sessions/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/sessions/20240101-000000-abcdef").status_code == 404
