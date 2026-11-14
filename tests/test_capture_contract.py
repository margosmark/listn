"""L0: Linux capture source and per-track failure contract tests.

The contract has two homes that must stay in step:

* ``listn/capture_contract.py`` — the executable form (enums, the ``ok``
  policy, the validators);
* ``docs/CAPTURE_CONTRACT.md`` — the normative prose.

These tests pin the enums to the producer (``listn.pipewire`` /
``listn.ipc``) so the doc cannot drift from the code, pin every canonical
decision fixture against the validator, check real fake-``pw-record``
reports against the validator *and* the ``ok`` policy, and exercise the
API-level pause and partial-failure semantics the contract defines.

No daemon, no hardware, no production audio: stand-in ``pw-dump`` /
``pw-record`` executables with synthetic tones, like the rest of the
PipeWire suite.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest

from listn import capture_contract as cc
from listn import pipewire as pw
from listn.ipc import TRACK_MIC, TRACK_SYSTEM, TRACK_SPEAKER

from .test_pipewire import fake_pw, make_targets, start_recorder, wait_for
from .test_pipewire_wiring import MIC, SINK, fake_pw_env, pw_client

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "capture_contract"
DOC = Path(__file__).parents[1] / "docs" / "CAPTURE_CONTRACT.md"

REPORT_FIXTURES = (
    "report_ok.json",
    "report_mic_silent.json",
    "report_system_silent_alone_ok.json",
    "report_identical_tracks.json",
    "report_clipped_track.json",
    "report_partial_node_loss.json",
)

INVALID_REPORT_FIXTURES = (
    ("report_invalid_track_state.json", "state"),
    ("report_invalid_ok_claim.json", "ok"),
    ("report_invalid_missing_track.json", "tracks"),
)


def _load(name: str) -> dict:
    path = FIXTURE_DIR / name
    assert path.is_file(), f"contract fixture missing: {path}"
    return json.loads(path.read_text())


def _drain(q: queue.Queue, want: str, timeout: float = 8.0) -> dict:
    """Wait for one bus event whose type matches ``want``."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            event = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if event.get("type") == want:
            return event
    raise AssertionError(f"no {want!r} event within {timeout}s")


def _wait_session_terminal(client, session_id: str, timeout: float = 20.0) -> dict:
    """Wait until the engine session reaches done/error, so teardown never
    races the transcription thread against the temp data directory."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        meta = client.get(f"/api/sessions/{session_id}").json()["meta"]
        if meta["status"] in ("done", "error"):
            return meta
        time.sleep(0.05)
    raise AssertionError(f"session {session_id} never reached a terminal state")


# ---------------------------------------------------------------------------
# The contract's constants must equal what the producer actually does
# ---------------------------------------------------------------------------

def test_contract_constants_match_the_producer():
    # Track roles and speaker labels come from the shared IPC frame contract
    # (track 0 = system/Others, track 1 = mic/Me) — the same frames the
    # macOS capture client sends.
    assert cc.TRACK_ROLES == {"system": TRACK_SYSTEM, "mic": TRACK_MIC}
    assert cc.TRACK_ROLES == {"system": 0, "mic": 1}
    assert cc.TRACK_SPEAKER_LABELS == {
        name: TRACK_SPEAKER[track_id]
        for name, track_id in cc.TRACK_ROLES.items()}
    # Per-track states: the recorder's own state machine, default included.
    assert cc.TRACK_STATES == ("pending", "recording", "stopped", "failed")
    assert pw._Track.__dataclass_fields__["state"].default == "pending"
    # Final reports only ever carry the two terminal states.
    assert cc.FINAL_TRACK_STATES == ("stopped", "failed")
    assert set(cc.MID_RUN_TRACK_STATES) | set(cc.FINAL_TRACK_STATES) == set(cc.TRACK_STATES)
    # Selection labels: select_targets() writes exactly these strings.
    assert cc.SELECTION_LABELS == ("explicit", "default", "fallback")
    # Verdicts and the documented signal thresholds (see CAPTURE_CONTRACT.md).
    assert cc.VERDICTS == ("ok", "quiet", "silent", "clipped")
    assert (pw.SILENT_PEAK_DB, pw.QUIET_PEAK_DB) == (-60.0, -35.0)
    assert pw.CLIPPED_FS_FRACTION == 0.02
    assert pw.RAIL_LEVEL == pytest.approx(32766.0 / 32768.0)
    assert pw.CLEAN_STOP_CODES == (0, 1, 130)
    # Top-level capture states the live status may report.
    assert cc.LIVE_STATES == ("recording", "stopping", "stopped", "failed")


def test_source_kind_derivation_including_the_reserved_app_kind():
    assert cc.source_kind({"media_class": "Audio/Sink"}) == cc.KIND_SINK_MONITOR
    assert cc.source_kind({"media_class": "Audio/Source"}) == cc.KIND_MIC
    # Reserved for L1 (per-app capture): an output-producing application
    # stream. L0 never selects it, but the kind is already defined so L1
    # extends the same table instead of inventing a parallel one.
    assert cc.source_kind({"media_class": "Stream/Output/Audio"}) == cc.KIND_APP
    assert cc.KIND_APP in cc.SOURCE_KINDS
    # Unknown or absent media class: no kind, never a guess.
    assert cc.source_kind({"media_class": "Audio/Duplex"}) is None
    assert cc.source_kind({}) is None
    assert cc.source_kind(None) is None
    # The system track accepts sink-monitor (today) and app (L1, reserved);
    # the mic track accepts only a real input.
    assert cc.SYSTEM_KINDS == (cc.KIND_SINK_MONITOR, cc.KIND_APP)
    assert cc.MIC_KINDS == (cc.KIND_MIC,)


def test_stable_identity_is_the_node_name_not_the_id():
    target = {"id": 32, "serial": 117, "name": "alsa_output.pci-0000.analog-stereo",
              "media_class": "Audio/Sink", "description": "Speakers",
              "track": "system", "selection": "explicit"}
    # PipeWire recycles ids (observed: id 56 reused with a new serial), so
    # the stable identity the contract names is the node name; serial/id are
    # the liveness probe's companions, not the identity on their own.
    assert cc.stable_identity(target) == "alsa_output.pci-0000.analog-stereo"
    assert cc.stable_identity({"name": "same", "id": 1}) == cc.stable_identity({"name": "same", "id": 99})
    assert cc.stable_identity({}) == ""
    assert cc.stable_identity({"name": None}) == ""


# ---------------------------------------------------------------------------
# Canonical decision fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", REPORT_FIXTURES)
def test_decision_fixture_matches_the_contract(name):
    fixture = _load(name)
    report = fixture["report"]
    violations = cc.validate_report(report)
    assert violations == [], f"{name}: {violations}"
    assert report["ok"] is fixture["expect"]["ok"]
    assert report["ok"] == cc.expected_ok(report)
    failed = sorted(track for track, t in report["tracks"].items()
                    if t["state"] == "failed")
    assert failed == fixture["expect"]["failed_tracks"], name
    # Fixtures pin the exact current report shape (an additive change to the
    # report must update capture_contract's key tuples in the same commit).
    assert set(report) == set(cc.REPORT_KEYS), name
    for track in report["tracks"].values():
        assert set(track) == set(cc.TRACK_KEYS), name
    for target in report["targets"].values():
        assert set(target) == set(cc.TARGET_KEYS), name


@pytest.mark.parametrize("name,needle", INVALID_REPORT_FIXTURES)
def test_validator_refuses_contract_violations(name, needle):
    fixture = _load(name)
    violations = cc.validate_report(fixture["report"])
    assert violations, f"{name} must not validate"
    assert any(needle in v for v in violations), violations


def test_live_status_fixture_matches_the_contract():
    fixture = _load("live_status_recording.json")
    status = fixture["status"]
    assert cc.validate_live_status(status) == []
    assert status["state"] == fixture["expect"]["state"]
    assert set(status) == set(cc.LIVE_STATUS_KEYS)
    for track in status["tracks"].values():
        assert set(track) == set(cc.LIVE_TRACK_STATUS_KEYS)


def test_pause_is_not_a_capture_source_state():
    """Pause belongs to the engine session, not to the capture source:
    the live status must never claim a paused capture."""
    fixture = _load("live_status_invalid_paused.json")
    violations = cc.validate_live_status(fixture["status"])
    assert violations, "a paused capture-source state must not validate"
    assert any("state" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Real reports (stand-in pw-record, synthetic tones) against the contract
# ---------------------------------------------------------------------------

def _wait_delivered(recorder, timeout: float = 15.0) -> None:
    """Wait until both stand-in tracks have delivered audio before stop().

    The verdict/ok assertions below classify *delivered* audio; a fixed
    sleep can end the capture before the CPU-starved stand-ins have
    written their first 0.1 s chunk, and an empty WAV is honestly
    reported as a failed track with ``verdict: None`` — a race with the
    generator, not the contract under test.
    """
    assert wait_for(lambda: all(track.frames > 0
                                for track in recorder._tracks.values()),
                    timeout=timeout), (
        "no audio delivered before stop: "
        + ", ".join(f"{name}={track.frames}"
                    for name, track in recorder._tracks.items()))


def test_real_ok_report_satisfies_contract_and_ok_policy(fake_pw):
    recorder = start_recorder(fake_pw)
    _wait_delivered(recorder)
    report = recorder.stop()
    assert cc.validate_report(report) == []
    assert report["ok"] is True
    assert report["ok"] == cc.expected_ok(report)
    assert set(report) == set(cc.REPORT_KEYS)
    for name, track in report["tracks"].items():
        assert set(track) == set(cc.TRACK_KEYS)
        assert track["track"] == name
        assert track["track_id"] == cc.TRACK_ROLES[name]
        assert set(track["target"]) == set(cc.TARGET_KEYS)
        assert track["target"] == report["targets"][name]
        assert track["target"]["selection"] in cc.SELECTION_LABELS
    # Source kinds on a real selection: whole-sink monitor + real input.
    assert cc.source_kind(report["targets"]["system"]) == cc.KIND_SINK_MONITOR
    assert cc.source_kind(report["targets"]["mic"]) == cc.KIND_MIC
    for target in report["targets"].values():
        assert isinstance(target["name"], str) and target["name"]
        assert isinstance(target["id"], int)
        assert isinstance(target["serial"], int)


def test_real_silent_mic_report_fails_the_ok_policy(fake_pw):
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.tone",
                                                            mic_name="mic.silent"))
    _wait_delivered(recorder)
    report = recorder.stop()
    assert cc.validate_report(report) == []
    assert report["tracks"]["mic"]["verdict"] == "silent"
    assert report["ok"] is False
    assert report["ok"] == cc.expected_ok(report)


def test_real_system_silent_alone_stays_ok(fake_pw):
    """The asymmetric decision: nobody may be playing anything, so only a
    silent *microphone* fails the run (system silence alone does not)."""
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.silent",
                                                            mic_name="mic.tone"))
    _wait_delivered(recorder)
    report = recorder.stop()
    assert cc.validate_report(report) == []
    assert report["tracks"]["system"]["verdict"] == "silent"
    assert report["ok"] is True
    assert report["ok"] == cc.expected_ok(report)


def test_partial_track_death_is_reported_not_hidden(fake_pw):
    """One track dying mid-run fails the run but keeps the other track's
    audio: partial capture is retained and reported, never claimed ok."""
    recorder = start_recorder(fake_pw, targets=make_targets(sys_name="sys.die",
                                                            mic_name="mic.tone"))
    assert wait_for(lambda: recorder._tracks["system"].state == "failed")
    # Condition wait, not wall time: the mic's stand-in can still be inside
    # its startup/first-chunk window when the sibling dies (worse under CPU
    # saturation), and stopping there honestly reports an empty mic track —
    # a race with the generator, not the partial-death contract under test.
    assert wait_for(lambda: recorder._tracks["mic"].frames > 0, timeout=10.0), (
        "mic never delivered audio before the stop")
    report = recorder.stop()
    assert cc.validate_report(report) == []
    system, mic = report["tracks"]["system"], report["tracks"]["mic"]
    assert system["state"] == "failed" and system["error"]
    assert mic["state"] == "stopped" and mic["frames"] > 0
    assert report["ok"] is False
    assert report["ok"] == cc.expected_ok(report)


# ---------------------------------------------------------------------------
# Mid-run live status (the shape GET /api/live serves under "pipewire")
# ---------------------------------------------------------------------------

class _StubEngine:
    """Just enough engine for PipeWireLive's stop path."""

    def stop_live(self, wait: bool = False):
        class Meta:
            id = "contract-stub-session"
        return Meta()

    def live_active(self) -> bool:
        return True


def test_live_status_mid_run_and_after_stop(fake_pw):
    from listn.pipewire_live import PipeWireLive

    # Same wiring start_live() uses: the capture object exists first so the
    # recorder's on_block lands in it and the frame counters are real.
    capture = PipeWireLive(None, _StubEngine(), fake_pw["out"])
    recorder = pw.TwoTrackRecorder(fake_pw["out"], make_targets(),
                                   pw_record=fake_pw["script"],
                                   probe=lambda n: True,
                                   node_watch_interval=0.05,
                                   on_block=capture.on_block)
    capture.recorder = recorder
    recorder.start()
    try:
        assert wait_for(lambda: all(capture._frames.get(n, 0) > 0
                                    for n in ("system", "mic")))
        status = capture.status()
        assert cc.validate_live_status(status) == []
        assert status["active"] is True and status["state"] == "recording"
        for name in ("system", "mic"):
            track = status["tracks"][name]
            assert track["state"] == "recording"
            assert track["frames"] > 0
        result = capture.stop(wait=True)
        assert result["pipewire"] is not None
        final = capture.status()
        assert cc.validate_live_status(final) == []
        assert final["active"] is False
        assert final["state"] == "stopped"
        assert cc.validate_report(final["report"]) == []
        assert final["report"]["ok"] is True
    finally:
        capture.active = False


# ---------------------------------------------------------------------------
# API-level pause and partial-failure semantics (server, stand-in tools)
# ---------------------------------------------------------------------------

def test_pause_never_stops_or_retargets_capture_streams(pw_client):
    """Pause is an engine-session concern: both pw-record streams keep
    running through pause/resume, nothing is reopened or switched, and the
    stop that follows still returns an honest ok report."""
    started = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "title": "Pause contract",
        "system_target": SINK, "mic_target": MIC})
    assert started.status_code == 200, started.text

    end = time.monotonic() + 10
    frames = {}
    while time.monotonic() < end:
        pw_status = pw_client.get("/api/live").json()["pipewire"]
        frames = {n: pw_status["tracks"][n]["frames"] for n in ("system", "mic")}
        if all(v > 0 for v in frames.values()):
            break
        time.sleep(0.1)
    assert all(v > 0 for v in frames.values()), frames

    assert pw_client.post("/api/live/pause").status_code == 200
    paused_live = pw_client.get("/api/live").json()
    assert paused_live["live"]["state"] == "paused"
    paused_pw = paused_live["pipewire"]
    assert cc.validate_live_status(paused_pw) == []
    # The capture source is untouched by pause: still active, still recording.
    assert paused_pw["active"] is True and paused_pw["state"] == "recording"
    before = {n: paused_pw["tracks"][n]["frames"] for n in ("system", "mic")}
    # Condition wait, not wall time: the claim is "the streams keep
    # delivering audio through pause", so wait for delivery. The stand-in
    # generator can be CPU-starved for longer than any fixed window.
    end = time.monotonic() + 10
    after_pw, after = paused_pw, before
    while time.monotonic() < end:
        after_pw = pw_client.get("/api/live").json()["pipewire"]
        after = {n: after_pw["tracks"][n]["frames"] for n in ("system", "mic")}
        if all(after[n] > before[n] for n in ("system", "mic")):
            break
        time.sleep(0.05)
    for name in ("system", "mic"):
        assert after_pw["tracks"][name]["state"] == "recording", name
        assert after[name] > before[name], (
            f"{name} stopped delivering audio during pause: {before} -> {after}")

    assert pw_client.post("/api/live/resume").status_code == 200
    stopped = pw_client.post("/api/live/stop")
    assert stopped.status_code == 200
    report = stopped.json()["pipewire"]
    assert cc.validate_report(report) == []
    assert report["ok"] is True, report
    assert report["ok"] == cc.expected_ok(report)
    _wait_session_terminal(pw_client, started.json()["id"])


def test_single_track_node_loss_keeps_capture_alive_and_reports_honestly(pw_client, fake_pw_env):
    """Partial failure at the API level: when the system track's node
    vanishes, that track fails with its own cause, the microphone track
    keeps recording, the capture is NOT torn down behind the user's back,
    and the final report keeps the partial audio and says ok=false."""
    q = pw_client.app.state.engine.bus.subscribe()
    started = pw_client.post("/api/live/start", json={
        "capture": "pipewire", "title": "Node loss",
        "system_target": SINK, "mic_target": MIC})
    assert started.status_code == 200, started.text

    pw_status = None
    end = time.monotonic() + 10
    while time.monotonic() < end:
        pw_status = pw_client.get("/api/live").json()["pipewire"]
        if all(pw_status["tracks"][n]["frames"] > 0 for n in ("system", "mic")):
            break
        time.sleep(0.1)
    assert pw_status is not None and all(
        pw_status["tracks"][n]["frames"] > 0 for n in ("system", "mic")), (
        f"no frames arrived on both tracks: {pw_status}")

    # Remove the sink node from the graph the stand-in pw-dump serves: the
    # recorder's node watch must fail the system track instead of letting
    # pw-record silently re-target.
    payload = json.loads(fake_pw_env["graph_file"].read_text())
    payload = [obj for obj in payload
               if not (obj.get("info", {}).get("props") or obj.get("props") or {}
                       ).get("node.name") == SINK]
    fake_pw_env["graph_file"].write_text(json.dumps(payload))

    end = time.monotonic() + 8
    while time.monotonic() < end:
        pw_status = pw_client.get("/api/live").json()["pipewire"]
        if pw_status["tracks"]["system"]["state"] == "failed":
            break
        time.sleep(0.1)
    else:
        pytest.fail("the system track was never failed after its node vanished")

    # The capture keeps running for the surviving track — no silent teardown,
    # no silent retarget — and the cause is published exactly once.
    assert pw_status["active"] is True
    assert pw_status["state"] == "recording"
    assert pw_status["tracks"]["mic"]["state"] == "recording"
    assert "disappeared" in pw_status["tracks"]["system"]["error"]
    event = _drain(q, "live.error")
    assert "system" in event.get("message", "")

    before = pw_status["tracks"]["mic"]["frames"]
    # Condition wait, not wall time: the claim is "the mic keeps
    # delivering audio while the sibling track is dead", so wait for
    # delivery; a fixed window can close before the starved stand-in's
    # next chunk arrives.
    end = time.monotonic() + 10
    after = pw_status
    while time.monotonic() < end:
        after = pw_client.get("/api/live").json()["pipewire"]
        if after["tracks"]["mic"]["frames"] > before:
            break
        time.sleep(0.05)
    assert after["tracks"]["mic"]["frames"] > before, "mic stopped during partial failure"

    stopped = pw_client.post("/api/live/stop")
    assert stopped.status_code == 200
    report = stopped.json()["pipewire"]
    assert cc.validate_report(report) == []
    assert report["ok"] is False
    assert report["ok"] == cc.expected_ok(report)
    system, mic = report["tracks"]["system"], report["tracks"]["mic"]
    assert system["state"] == "failed" and system["error"]
    # The partial capture is retained and usable.
    assert mic["state"] == "stopped" and mic["frames"] > 0
    assert Path(mic["file"]).is_file()
    _wait_session_terminal(pw_client, started.json()["id"])


# ---------------------------------------------------------------------------
# Privacy / local-only invariants
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_PARTS = ("transcript", "title", "content", "window", "app_name",
                        "secret", "token", "text", "pcm", "samples")


def _assert_local_metadata_only(obj, path: str = "report"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            assert not any(part in lowered for part in _SENSITIVE_KEY_PARTS), (
                f"{path}.{key}: the capture report carries metadata only, "
                "never audio content, transcripts or window/app titles")
            _assert_local_metadata_only(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _assert_local_metadata_only(value, f"{path}[{index}]")
    else:
        assert not isinstance(obj, (bytes, bytearray, memoryview)), (
            f"{path}: raw bytes in the report would be captured audio")
        if isinstance(obj, str):
            assert "http://" not in obj and "https://" not in obj, (
                f"{path}: report strings are local paths/labels, not URLs")


def test_real_report_carries_only_local_metadata(fake_pw):
    recorder = start_recorder(fake_pw)
    _wait_delivered(recorder)
    report = recorder.stop()
    _assert_local_metadata_only(report)
    # out_dir/file/log are local filesystem paths; the walker already
    # forbade URLs and raw bytes anywhere in the report.
    assert str(report["out_dir"]).startswith("/")
    assert str(report["tracks"]["mic"]["file"]).startswith("/")


def test_fixtures_carry_only_local_metadata():
    for name in REPORT_FIXTURES + tuple(n for n, _ in INVALID_REPORT_FIXTURES) + (
            "live_status_recording.json", "live_status_invalid_paused.json"):
        fixture = _load(name)
        payload = fixture.get("report") or fixture.get("status")
        _assert_local_metadata_only(payload, name)
        # Fixtures are small hand-authored metadata, never embedded audio.
        assert (FIXTURE_DIR / name).stat().st_size < 20_000, name


# ---------------------------------------------------------------------------
# The normative doc states the same enumerations as the module
# ---------------------------------------------------------------------------

def test_contract_doc_states_the_enumerations():
    assert DOC.is_file(), "docs/CAPTURE_CONTRACT.md is part of the L0 contract"
    doc = DOC.read_text()
    literals = (*cc.SOURCE_KINDS, *cc.TRACK_STATES, *cc.VERDICTS,
                *cc.SELECTION_LABELS, *cc.LIVE_STATES, *cc.FINAL_TRACK_STATES)
    for literal in literals:
        assert literal in doc, f"docs/CAPTURE_CONTRACT.md never mentions {literal!r}"
    # The decisions themselves must be spelled out, not just the enums.
    for phrase in ("partial", "pause", "stable identity", "never guesses",
                   "ok"):
        assert phrase.lower() in doc.lower(), f"doc never discusses {phrase!r}"
