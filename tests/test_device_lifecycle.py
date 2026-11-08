"""L7: device hotplug, node loss and capture lifecycle (injected events).

Every graph/process event here is injected — no real PipeWire daemon, no
real device and no production audio. The recorder-level tests reuse the
stand-in ``pw-record`` from ``tests.test_pipewire`` (synthetic tones only)
so the replacement path is exercised end to end against a real WAV file.
"""

from __future__ import annotations

import time

import pytest

from listn import capture_contract as cc
from listn import device_lifecycle as dl
from listn import pipewire as pw
from listn.device_lifecycle import DeviceLifecycle
from listn.tray import device_note, derive_status

from tests.test_pipewire import (FAKE_PW_RECORD, fake_pw, make_targets,
                                 node, start_recorder, wait_for)

SYS = {"name": "sys.tone", "media_class": "Audio/Sink",
       "description": "Speakers"}
MIC = {"name": "mic.tone", "media_class": "Audio/Source",
       "description": "USB Microphone"}
OTHER_SINK = {"name": "hdmi.out", "media_class": "Audio/Sink",
               "description": "HDMI"}
OTHER_SRC = {"name": "bluez.src", "media_class": "Audio/Source",
             "description": "Headset"}


def graph_payload(*, sinks=(SYS, OTHER_SINK), sources=(MIC, OTHER_SRC),
                  defaults=None) -> dict:
    """A ``discover_devices()``-shaped snapshot with injected contents."""
    return {"sinks": [dict(n) for n in sinks],
            "sources": [dict(n) for n in sources],
            "streams": [],
            "defaults": dict(defaults or {"sink": SYS["name"],
                                          "source": MIC["name"]})}


def armed(**kwargs) -> DeviceLifecycle:
    """A lifecycle whose streams are alive until a test says otherwise."""
    alive = kwargs.pop("alive", {"system": True, "mic": True})
    life = DeviceLifecycle(stream_alive=lambda name: alive.get(name, True),
                           **kwargs)
    life.start({"system": dict(SYS), "mic": dict(MIC)})
    life.observe_graph(graph_payload())
    return life


# ---------------------------------------------------------------------------
# Injected graph / process events
# ---------------------------------------------------------------------------

def test_node_loss_fails_the_track_without_touching_the_target():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST,
                          "target node 'sys.tone' disappeared from the graph")
    status = life.status()
    system = status["tracks"]["system"]
    assert system["state"] == "lost"
    assert system["cause"] == "node_lost"
    assert system["node"] == "sys.tone"          # identity never moves
    assert status["retargets"] == 0              # no silent retarget
    assert status["tracks"]["mic"]["state"] == "recording"
    assert any(e["kind"] == "node_lost" for e in status["events"])


def test_process_exit_is_distinguished_from_node_loss():
    life = armed()
    life.note_stream_dead("mic", dl.CAUSE_PROCESS_EXIT,
                          "pw-record for the mic track exited by itself")
    assert life.state_of("mic") == "lost"
    assert life.status()["tracks"]["mic"]["cause"] == "process_exit"


def test_duplicate_death_is_one_event():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    assert len([e for e in life.events if e["kind"] == "node_lost"]) == 1


def test_daemon_loss_is_unavailable_not_a_dead_device():
    life = armed()
    life.observe_graph(None, error="pw-dump: connection refused")
    assert life.state_of("system") == "unavailable"
    assert life.status()["tracks"]["system"]["cause"] == "daemon_unavailable"
    assert any(e["kind"] == "daemon_unavailable" for e in life.events)
    # A node that is really gone still reads as unavailable while the
    # daemon is down: we cannot tell, and guessing would be a lie.
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    assert life.state_of("system") == "unavailable"
    life.observe_graph(graph_payload())
    assert any(e["kind"] == "daemon_restored" for e in life.events)
    assert life.state_of("system") == "lost"     # the real cause shows again
    assert life.status()["tracks"]["system"]["cause"] == "node_lost"


def test_default_change_is_reported_but_never_applied_mid_capture():
    life = armed()
    changed = graph_payload(defaults={"sink": OTHER_SINK["name"],
                                      "source": MIC["name"]})
    life.observe_graph(changed)
    events = [e for e in life.events if e["kind"] == "default_changed"]
    assert len(events) == 1
    assert events[0]["node"] == OTHER_SINK["name"]
    assert "explicit target" in events[0]["detail"]
    defaults = life.status()["defaults"]
    assert defaults["observed"]["sink"] == OTHER_SINK["name"]
    assert defaults["remembered"]["sink"] == SYS["name"]   # untouched: active
    assert life.status()["tracks"]["system"]["node"] == SYS["name"]


def test_remembered_default_updates_only_while_idle():
    life = armed()
    life.observe_graph(graph_payload(
        defaults={"sink": OTHER_SINK["name"], "source": MIC["name"]}))
    life.stop()
    life.observe_graph(graph_payload(
        defaults={"sink": "another.sink", "source": MIC["name"]}))
    defaults = life.status()["defaults"]
    assert defaults["remembered"]["sink"] == "another.sink"
    assert any("idle" in (e["detail"] or "") for e in life.events
               if e["kind"] == "default_changed")


def test_replugged_device_is_offered_not_adopted():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    # Nothing of that class exists any more: not "lost", "no device".
    life.observe_graph(graph_payload(sinks=()))
    assert life.state_of("system") == "no_device"
    assert life.status()["tracks"]["system"]["replacement_candidates"] == []
    # Another sink appears (hotplug): it is offered, never adopted.
    life.observe_graph(graph_payload(sinks=(OTHER_SINK,)))
    assert life.state_of("system") == "lost"
    names = [c["name"] for c in
             life.status()["tracks"]["system"]["replacement_candidates"]]
    assert names == ["hdmi.out"]                 # the sibling's class only
    # Replug: the original node comes back — still not adopted.
    life.observe_graph(graph_payload())
    system = life.status()["tracks"]["system"]
    assert system["state"] == "lost"
    assert system["reconnectable"] is True
    assert life.status()["retargets"] == 0
    assert any(e["kind"] == "node_reappeared" for e in life.events)


def test_silent_signal_is_its_own_state_while_the_device_is_present():
    life = armed()
    life.observe_signal("mic", True)
    assert life.state_of("mic") == "silent"
    assert life.status()["tracks"]["mic"]["cause"] == "silent_signal"
    life.observe_signal("mic", False)
    assert life.state_of("mic") == "recording"


def test_capture_end_reports_stopped_and_keeps_the_cause():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    life.stop()
    system = life.status()["tracks"]["system"]
    assert system["state"] == "stopped"
    assert system["cause"] == "node_lost"        # partial track keeps its why
    assert life.status()["active"] is False
    assert life.status()["tracks"]["mic"]["state"] == "stopped"


# ---------------------------------------------------------------------------
# Bounded, explicit recovery — the only retarget path
# ---------------------------------------------------------------------------

def test_replacement_is_refused_until_the_user_asks():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    # Nothing automatic ever runs: no replacement, no new target.
    assert life.status()["retargets"] == 0
    assert life.status()["tracks"]["system"]["state"] == "lost"


def test_replacement_refusals_are_explicit_and_leave_state_alone():
    life = armed(alive={"system": False, "mic": True})
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    cases = {
        "hdmi.out": None,                         # accepted below
        "bluez.src": dl.REFUSED_WRONG_KIND,       # a source, not a sink
        "mic.tone": dl.REFUSED_WRONG_KIND,        # the sibling's own node
        "not.in.graph": dl.REFUSED_UNKNOWN_NODE,
    }
    for name, code in cases.items():
        if code is None:
            continue
        decision = life.begin_replacement("system", name)
        assert decision["ok"] is False, name
        assert decision["code"] == code, name
        assert decision["reason"], name
    # A still-recording track is not replaceable at all.
    decision = life.begin_replacement("mic", "bluez.src")
    assert decision["code"] == dl.REFUSED_STILL_RECORDING
    assert life.status()["retargets"] == 0


def test_replacement_is_refused_while_the_sibling_is_dead():
    life = armed(alive={"system": False, "mic": False})
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    life.note_stream_dead("mic", dl.CAUSE_PROCESS_EXIT, "exited")
    decision = life.begin_replacement("system", "hdmi.out")
    assert decision["ok"] is False
    assert decision["code"] == dl.REFUSED_SIBLING_DEAD


def test_recovery_is_bounded_and_then_reported_as_exhausted():
    life = armed(alive={"system": False, "mic": True},
                 max_recovery_attempts=2)
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    for _ in range(2):
        decision = life.begin_replacement("system", "hdmi.out")
        assert decision["ok"] is True
        life.commit_replacement("system", "hdmi.out")
        # The replacement died again (the fake device unplugged twice).
        life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone again")
    decision = life.begin_replacement("system", "hdmi.out")
    assert decision["ok"] is False
    assert decision["code"] == dl.REFUSED_ATTEMPTS_EXHAUSTED
    assert life.status()["tracks"]["system"]["state"] == "exhausted"
    assert life.status()["retargets"] == 2       # only what the user chose


def test_committed_replacement_is_counted_and_visible():
    life = armed(alive={"system": False, "mic": True})
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    decision = life.begin_replacement("system", "hdmi.out")
    assert decision["ok"] is True and decision["node"]["name"] == "hdmi.out"
    life.commit_replacement("system", "hdmi.out")
    status = life.status()
    assert status["retargets"] == 1
    assert status["tracks"]["system"]["state"] == "replaced"
    assert status["tracks"]["system"]["node"] == "hdmi.out"
    assert status["tracks"]["system"]["recovery_attempts"] == 1
    assert any(e["kind"] == "recovery_committed" for e in status["events"])


def test_pause_is_not_a_capture_source_state_in_this_module():
    """L0 §6 + L7: pause never reopens, switches or even appears here."""
    life = armed()
    assert not hasattr(life, "pause") and not hasattr(life, "resume")
    status = life.status()
    assert "paused" not in status["tracks"]["system"]["state"]
    assert "paused" not in [e["kind"] for e in status["events"]]
    # And a paused capture (engine-side) leaves the device state alone.
    before = status["tracks"]["system"]
    life.observe_graph(graph_payload())
    assert life.status()["tracks"]["system"] == before


# ---------------------------------------------------------------------------
# Contract shape and user-facing exposure
# ---------------------------------------------------------------------------

def test_status_shapes_are_the_contract_key_tuples():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    status = life.status()
    assert set(status) == set(cc.LIVE_DEVICES_KEYS)
    assert set(status["defaults"]) == set(cc.LIVE_DEVICE_DEFAULT_KEYS)
    for track in status["tracks"].values():
        assert set(track) == set(cc.LIVE_DEVICE_TRACK_KEYS)
    for event in status["events"]:
        assert set(event) == set(cc.DEVICE_EVENT_KEYS)
    assert set(status["tracks"]["system"]["replacement_candidates"][0]) \
        == set(cc.DEVICE_CANDIDATE_KEYS) if status["tracks"]["system"][
            "replacement_candidates"] else True


def test_devices_block_validates_inside_a_live_status():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    status = {
        "active": True, "state": "recording",
        "out_dir": "/tmp/listn-run", "rate": 48000,
        "targets": {"system": {**SYS, "id": 1, "serial": 1, "track": "system",
                               "selection": "explicit"},
                    "mic": {**MIC, "id": 2, "serial": 2, "track": "mic",
                            "selection": "explicit"}},
        "tracks": {name: {"state": "recording", "frames": 10,
                          "seconds": 0.1, "error": None, "file": None}
                   for name in ("system", "mic")},
        "devices": life.status(),
        "report": None,
    }
    assert cc.validate_live_status(status) == []
    # ...and a lying activity flag is refused.
    status["devices"] = {**status["devices"], "active": False}
    assert any("devices.active" in v for v in cc.validate_live_status(status))


def test_device_state_is_metadata_only():
    life = armed()
    life.note_stream_dead("system", dl.CAUSE_NODE_LOST, "gone")
    life.observe_graph(graph_payload())
    status = life.status()
    import json
    text = json.dumps(status)
    for banned in ("transcript", "title", "content", "window", "app_name",
                   "secret", "token", "http://", "https://", "samples"):
        assert banned not in text, banned


def test_tray_line_names_the_track_that_needs_attention():
    life = armed()
    life.note_stream_dead("mic", dl.CAUSE_NODE_LOST, "gone")
    note = device_note(life.status())
    assert note == "mic: device lost"
    payload = {"live": {"state": "recording", "title": "Standup",
                        "id": "s1", "elapsed": 3.0},
               "pipewire": {"active": True, "state": "recording",
                            "targets": {}, "tracks": {}, "report": None,
                            "devices": life.status()}}
    status = derive_status(payload)
    assert status.device_note == "mic: device lost"
    assert "mic: device lost" in status.summary()
    assert device_note(None) is None


# ---------------------------------------------------------------------------
# Recorder: explicit replacement of a lost track (fake pw-record, tones)
# ---------------------------------------------------------------------------

def _flip_probe(recorder, lost: str):
    recorder.probe = lambda n: n.name != lost


def test_replacement_resumes_the_same_wav_and_pads_the_gap(fake_pw):
    recorder = start_recorder(fake_pw)
    assert wait_for(lambda: recorder._tracks["system"].frames > 0)
    time.sleep(0.2)
    _flip_probe(recorder, "sys.tone")
    assert wait_for(lambda: recorder._tracks["system"].state == "failed")
    assert recorder._tracks["system"].writer_held is True
    before = recorder._tracks["system"].frames

    replacement = pw.Node(id=77, name="hdmi.out", serial=77,
                          media_class="Audio/Sink", description="HDMI")
    recorder.replace_track("system", replacement, gap_frames=4800)
    assert recorder._tracks["system"].state == "recording"
    assert recorder._tracks["system"].target.node.name == "hdmi.out"
    assert recorder._tracks["system"].frames >= before + 4800
    time.sleep(0.35)
    report = recorder.stop()
    system = report["tracks"]["system"]
    assert system["state"] == "stopped", system
    assert system["file"].endswith("system.wav")
    # One file, one timeline: the gap is inside the finalized WAV.
    assert system["frames"] >= before + 4800
    assert system["duration"] * 48000 >= system["frames"] - 48000
    assert report["ok"] is True, report
    assert report["targets"]["system"]["name"] == "hdmi.out"


def test_replacement_never_creates_a_second_stream(fake_pw):
    recorder = start_recorder(fake_pw)
    assert wait_for(lambda: recorder._tracks["system"].frames > 0)
    live_proc = recorder._tracks["system"].proc
    replacement = pw.Node(id=77, name="hdmi.out", serial=77,
                          media_class="Audio/Sink", description="HDMI")
    with pytest.raises(pw.CaptureError, match="not failed"):
        recorder.replace_track("system", replacement)
    # Refused while the track is healthy: the original process is untouched.
    assert recorder._tracks["system"].proc is live_proc
    assert live_proc.poll() is None
    recorder.stop()


def test_replacement_refuses_wrong_kind_and_missing_node(fake_pw):
    recorder = start_recorder(fake_pw)
    assert wait_for(lambda: recorder._tracks["system"].frames > 0)
    _flip_probe(recorder, "sys.tone")
    assert wait_for(lambda: recorder._tracks["system"].state == "failed")
    source = pw.Node(id=78, name="bluez.src", serial=78,
                     media_class="Audio/Source", description="Headset")
    with pytest.raises(pw.CaptureError, match="records"):
        recorder.replace_track("system", source)
    gone = pw.Node(id=79, name="ghost.out", serial=79,
                   media_class="Audio/Sink", description="Gone")
    recorder.probe = lambda n: False      # nothing is in the graph now
    with pytest.raises(pw.CaptureError, match="not in the PipeWire graph"):
        recorder.replace_track("system", gone)
    assert recorder._tracks["system"].state == "failed"
    recorder.stop()


def test_replacement_is_refused_once_the_sibling_track_is_gone(fake_pw):
    recorder = start_recorder(fake_pw)
    assert wait_for(lambda: recorder._tracks["system"].frames > 0)
    # Both probes fail: both tracks die (L0's both-dead abort condition).
    recorder.probe = lambda n: False
    # Wait for BOTH deaths, not just the sibling's: replace_track validates
    # the requested track first ("not failed") and only then the sibling
    # ("not recording"). Each track's reader thread detects the lost node on
    # its own schedule, so calling replace_track the moment the mic is dead
    # races the system track's independent detection loop — under load the
    # system track is still "recording" and the wrong guard fires.
    assert wait_for(lambda: recorder._tracks["mic"].state == "failed"
                    and recorder._tracks["system"].state == "failed")
    replacement = pw.Node(id=77, name="hdmi.out", serial=77,
                          media_class="Audio/Sink", description="HDMI")
    with pytest.raises(pw.CaptureError, match="not recording"):
        recorder.replace_track("system", replacement)
    recorder.stop()
