"""Wiring tests: ``listn record`` and the local server's two-track path.

No daemon and no hardware — stand-in ``pw-dump``/``pw-record``
executables are placed on PATH speaking the same contract as the real
tools (graph JSON, raw int16 PCM on stdout, exit 1 after SIGINT), with
synthetic audio. The assertions read the produced WAVs with the module's
own analyzer, so a header-only or silent capture fails the test.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from listn import cli
from listn.engine import Engine
from listn.pipewire import analyze_wav
from listn.server import create_app

from .conftest import MAC_DEVICES, FakeSD
from .test_pipewire import FAKE_PW_RECORD, sample_dump

SINK = "alsa_output.pci-0000.analog-stereo"
MIC = "alsa_input.usb-mic.mono-fallback"

FAKE_PW_DUMP = r'''#!/usr/bin/env python3
"""Stand-in pw-dump: full graph with no args, id lookup with one arg.

Contract details the recorder depends on: a vanished id returns an empty
list with exit 0 (never an error), and a present id returns the object
carrying that id (never merely an object whose serial matches).
"""
import json, os, sys

path = os.environ.get("FAKE_PW_DUMP_FILE")
try:
    data = json.load(open(path)) if path else []
except Exception:
    data = []
args = sys.argv[1:]
if not args:
    sys.stdout.write(json.dumps(data))
    sys.exit(0)
wanted = str(args[0])
matched = [o for o in data if isinstance(o, dict) and str(o.get("id")) == wanted]
sys.stdout.write(json.dumps(matched))
sys.exit(0)
'''


@pytest.fixture
def fake_pw_env(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("pw-record", FAKE_PW_RECORD), ("pw-dump", FAKE_PW_DUMP)):
        script = bindir / name
        script.write_text(body)
        script.chmod(0o755)
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(sample_dump()))
    monkeypatch.setenv("FAKE_PW_DUMP_FILE", str(graph_file))
    monkeypatch.setenv("FAKE_PW_ARGV_FILE", str(tmp_path / "argv.log"))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return {"graph_file": graph_file, "bindir": bindir, "tmp_path": tmp_path}


@pytest.fixture
def pw_client(data_dir, fake_factory, fake_pw_env):
    sd = FakeSD(MAC_DEVICES, default=(0, 3), speed=40.0)
    engine = Engine(sd=sd, transcriber_factory=fake_factory)
    app = create_app(engine=engine, sd=sd)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


# ---------------------------------------------------------------------------
# CLI: listn record
# ---------------------------------------------------------------------------

def test_record_lists_discovered_nodes(fake_pw_env, data_dir, capsys):
    assert cli.main(["record", "--list"]) == 0
    out = capsys.readouterr().out
    assert "System outputs (Audio/Sink):" in out
    assert SINK in out and MIC in out
    assert "[session default]" in out
    # the monitor source is not offered as a microphone
    assert "bluez_sink.AA.monitor" not in out


def test_record_captures_two_real_tracks(fake_pw_env, data_dir, tmp_path, capsys):
    out_dir = tmp_path / "rec"
    rc = cli.main(["record", "--backend", "pipewire",
                   "--system-target", SINK, "--mic-target", MIC,
                   "--duration", "0.4", "--out", str(out_dir)])
    assert rc == 0
    assert "selection: explicit" in capsys.readouterr().out
    for name in ("system.wav", "mic.wav"):
        analysis = analyze_wav(out_dir / name)
        assert analysis["error"] is None, analysis
        assert analysis["frames"] > 0 and analysis["duration"] > 0.1, analysis
        assert analysis["verdict"] == "ok", analysis          # synthetic tone, not silence
        assert (out_dir / name).stat().st_size > 44 + 1000     # payload beyond the header
    assert out_dir / "system.wav" != out_dir / "mic.wav"        # independent files


def test_record_json_report_and_default_selection(fake_pw_env, data_dir, tmp_path, capsys):
    rc = cli.main(["record", "--duration", "0.3", "--json",
                   "--out", str(tmp_path / "j")])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0 and report["ok"] is True
    assert report["targets"]["system"]["selection"] == "default"
    assert report["targets"]["mic"]["selection"] == "default"
    for name in ("system", "mic"):
        assert report["tracks"][name]["frames"] > 0
        assert report["tracks"][name]["state"] == "stopped"


def test_record_rejects_unknown_target_without_starting(fake_pw_env, data_dir, tmp_path, capsys):
    out_dir = tmp_path / "never"
    rc = cli.main(["record", "--system-target", "no.such.node",
                   "--mic-target", MIC, "--duration", "0.2", "--out", str(out_dir)])
    assert rc == 3
    assert "No PipeWire node named 'no.such.node'" in capsys.readouterr().err
    assert not out_dir.exists()


def test_record_without_pw_tools_says_so(data_dir, tmp_path, capsys, monkeypatch):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert cli.main(["record", "--list"]) == 3
    assert "pw-dump not found" in capsys.readouterr().err
    assert cli.main(["record", "--system-target", SINK, "--mic-target", MIC]) == 3
    assert "pw-dump not found" in capsys.readouterr().err


def test_record_parser_rejects_unknown_backend(fake_pw_env, data_dir):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["record", "--backend", "wasapi"])


# ---------------------------------------------------------------------------
# Server: two-track capture from the web UI path
# ---------------------------------------------------------------------------

def wait_status(client, sid, want, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        meta = client.get(f"/api/sessions/{sid}").json()["meta"]
        if meta["status"] == want:
            return meta
        time.sleep(0.05)
    raise AssertionError(f"session never reached {want}")


def test_pipewire_devices_endpoint_reports_nodes(pw_client):
    d = pw_client.get("/api/pipewire/devices").json()
    assert d["available"] is True and d["error"] is None
    assert [n["name"] for n in d["sinks"]] == [SINK]
    assert [n["name"] for n in d["sources"]] == [MIC]
    assert [n["name"] for n in d["sources"]] != [n["name"] for n in d["sinks"]]
    assert d["usable"] == {"system": True, "mic": True}
    assert d["defaults"] == {"sink": SINK, "source": MIC}


def test_pipewire_start_requires_explicit_targets(pw_client):
    r = pw_client.post("/api/live/start", json={"capture": "pipewire"})
    assert r.status_code == 400 and "explicitly" in r.json()["detail"]
    r = pw_client.post("/api/live/start",
                       json={"capture": "pipewire", "system_target": SINK})
    assert r.status_code == 400 and "explicitly" in r.json()["detail"]
    assert pw_client.get("/api/live").json()["live"] is None
    assert pw_client.get("/api/sessions").json()["results"] == []


def test_pipewire_start_stop_produces_two_real_tracks(pw_client, data_dir):
    started = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "title": "Pairing",
        "system_target": SINK, "mic_target": MIC})
    assert started.status_code == 200, started.text
    body = started.json()
    sid = body["id"]
    assert body["capture"] == "ingest"
    assert "PipeWire" in body["device"]["name"]
    assert pw_client.post("/api/live/start", json={
        "capture": "pipewire", "system_target": SINK, "mic_target": MIC,
    }).status_code == 409

    # both tracks really produce audio before we claim anything
    pw = {}
    frames = None
    end = time.monotonic() + 10
    while time.monotonic() < end:
        pw = pw_client.get("/api/live").json()["pipewire"]
        frames = {n: pw["tracks"][n]["frames"] for n in ("system", "mic")}
        if all(v > 0 for v in frames.values()):
            break
        time.sleep(0.1)
    if not (frames and all(v > 0 for v in frames.values())):
        pytest.fail(f"no frames arrived on both tracks: {frames}")
    assert pw["state"] == "recording"
    assert pw["targets"]["system"]["id"] != pw["targets"]["mic"]["id"]

    stopped = pw_client.post("/api/live/stop")
    assert stopped.status_code == 200
    report = stopped.json()["pipewire"]
    assert report["ok"] is True, report
    seen = set()
    for name in ("system", "mic"):
        t = report["tracks"][name]
        assert t["state"] == "stopped" and t["error"] is None, t
        assert t["frames"] > 0 and t["duration"] > 0.1, t
        assert t["verdict"] == "ok", t
        path = Path(t["file"])
        assert path.exists() and path.stat().st_size > 44 + 1000
        seen.add(path)
    assert len(seen) == 2, "both tracks must be separate files"

    meta = wait_status(pw_client, sid, "done")
    assert meta["source_name"].startswith("PipeWire ·")
    session_dir = Path(pw_client.app.state.engine.library.root) / sid
    # the engine session kept independent per-track WAVs of its own
    for name in ("audio-system.wav", "audio-mic.wav"):
        track = session_dir / name
        assert track.exists(), f"{name} missing in {session_dir}"
        assert track.stat().st_size > 44 + 1000
    # and the stop status still carries the report
    final = pw_client.get("/api/live").json()["pipewire"]
    assert final["report"]["ok"] is True
    assert final["state"] == "stopped"


def test_pipewire_bad_target_never_starts(pw_client):
    r = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "system_target": "no.such.node", "mic_target": MIC})
    assert r.status_code == 400
    assert "No PipeWire node named 'no.such.node'" in r.json()["detail"]
    assert pw_client.get("/api/live").json()["live"] is None
    assert pw_client.get("/api/sessions").json()["results"] == []


def test_pipewire_unavailable_says_so_and_does_not_start(pw_client, tmp_path, monkeypatch):
    empty = tmp_path / "nopath"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    d = pw_client.get("/api/pipewire/devices").json()
    assert d["available"] is False
    assert "pw-dump" in d["error"] and d["usable"] == {"system": False, "mic": False}
    r = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "system_target": SINK, "mic_target": MIC})
    assert r.status_code == 400
    assert "pw-dump" in r.json()["detail"]
    assert pw_client.get("/api/live").json()["live"] is None
    assert pw_client.get("/api/sessions").json()["results"] == []


def test_both_tracks_dying_fails_the_session_truthfully(pw_client, fake_pw_env):
    """pw-record dying on its own must not keep a 'recording' session alive."""
    payload = sample_dump()
    for obj in payload:
        props = obj.get("info", {}).get("props") if isinstance(obj.get("info"), dict) else None
        props = props or obj.get("props") or {}
        if props.get("node.name") == SINK:
            props["node.name"] = "die.system.out"
        elif props.get("node.name") == MIC:
            props["node.name"] = "die.mic.in"
        elif props.get("metadata.name") == "default":
            for entry in obj.get("metadata", []):
                if entry.get("key") == "default.audio.sink":
                    entry["value"] = {"name": "die.system.out"}
    fake_pw_env["graph_file"].write_text(json.dumps(payload))

    started = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "title": "Short-lived",
        "system_target": "die.system.out", "mic_target": "die.mic.in"})
    assert started.status_code == 200, started.text
    sid = started.json()["id"]

    # the monitor stops the capture and stores the real cause
    meta = wait_status(pw_client, sid, "error", timeout=20)
    assert "PipeWire" in meta["error"] and "system" in meta["error"]

    live = pw_client.get("/api/live").json()
    assert live["pipewire"]["state"] == "failed"
    assert live["pipewire"]["report"] is not None
    report = live["pipewire"]["report"]
    assert report["ok"] is False
    for name in ("system", "mic"):
        t = report["tracks"][name]
        assert t["state"] == "failed", t
        assert t["error"], t
    # nothing is still capturing
    assert live["pipewire"]["active"] is False
    assert live["live"]["state"] in ("done", "error")


def test_index_ships_the_pipewire_picker(pw_client):
    html = pw_client.get("/").text
    for needle in ('id="pw-panel"', 'id="pw-mode"', 'id="pw-system"', 'id="pw-mic"'):
        assert needle in html
    js = pw_client.get("/static/app.js").text
    assert "/api/pipewire/devices" in js and "system_target" in js
