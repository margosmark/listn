"""L0 contract: Linux capture source abstraction and per-track failure report.

This module is the *executable* half of the contract; the normative prose
lives in ``docs/CAPTURE_CONTRACT.md``. Both are pinned to the producer by
``tests/test_capture_contract.py``, so code, doc and fixtures cannot drift
apart silently.

What the contract fixes
-----------------------
* **Track roles** — the two tracks are ``system`` (track 0, "Others") and
  ``mic`` (track 1, "Me"), the same frame ids the macOS capture client
  sends over the ingest socket.
* **Source kinds** — ``sink_monitor`` (an ``Audio/Sink`` captured through
  its monitor ports: today's whole-sink mode), ``mic`` (an ``Audio/Source``
  input), and ``app`` (an output-producing application stream — reserved
  for L1 per-app capture; L0 never selects it).
* **Stable identity** — a target is identified by its PipeWire *node
  name*; ``id`` is session-scoped and recycled, ``serial`` accompanies the
  name in the liveness probe. Clients must key on the name.
* **States and verdicts** — per-track ``pending``/``recording`` while
  capturing, ``stopped``/``failed`` in a final report; signal verdicts
  ``ok``/``quiet``/``silent``/``clipped``.
* **The ``ok`` policy** — :func:`expected_ok`: the run succeeds only when
  *both* tracks stopped cleanly with no error **and** the microphone
  carried signal. A silent system track alone does not fail the run
  (nobody may be playing); a silent microphone, a clipped track, a dead
  track or two identical tracks always do.
* **Pause** — an engine-session state only. The capture source has no
  paused state: pause never stops, reopens or retargets a stream, and the
  live status must never claim one.
* **Partial failure** — a start is all-or-nothing; a track that dies
  mid-run is failed with its own cause while the other track keeps
  recording; both tracks dead ends the capture. Partial audio is always
  finalized and retained, and ``ok`` is false whenever any track failed.

Extension points (L1 per-app capture, L7 device lifecycle) are additive:
new source kinds go into :data:`MEDIA_CLASS_KINDS`, new report fields into
the key tuples here *and* ``docs/CAPTURE_CONTRACT.md`` in the same
commit. The ``ok`` policy and the refusal-to-guess rules are not
extension points — changing them is a contract change, not a feature.

Local-only: the report is metadata (node names, numeric analysis, local
paths). It never contains audio samples, transcripts, window/app titles,
tokens or URLs, and this module makes no network calls.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Track roles (shared with the ingest frame contract, listn/ipc.py)
# ---------------------------------------------------------------------------

TRACK_ORDER = ("system", "mic")
TRACK_ROLES = {"system": 0, "mic": 1}
TRACK_SPEAKER_LABELS = {"system": "Others", "mic": "Me"}

# ---------------------------------------------------------------------------
# Source kinds
# ---------------------------------------------------------------------------

KIND_SINK_MONITOR = "sink_monitor"
KIND_MIC = "mic"
KIND_APP = "app"           # reserved for L1; never selected by L0
SOURCE_KINDS = (KIND_SINK_MONITOR, KIND_MIC, KIND_APP)
MEDIA_CLASS_KINDS = {
    "Audio/Sink": KIND_SINK_MONITOR,
    "Audio/Source": KIND_MIC,
    "Stream/Output/Audio": KIND_APP,
}
# What each track may point at: the system/Others track is the whole sink
# today and an application stream once L1 lands; the mic/Me track is only
# ever a real input.
SYSTEM_KINDS = (KIND_SINK_MONITOR, KIND_APP)
MIC_KINDS = (KIND_MIC,)

# ---------------------------------------------------------------------------
# Selection labels (select_targets() writes exactly these strings)
# ---------------------------------------------------------------------------

SELECTION_LABELS = ("explicit", "default", "fallback")

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

MID_RUN_TRACK_STATES = ("pending", "recording")
FINAL_TRACK_STATES = ("stopped", "failed")
TRACK_STATES = MID_RUN_TRACK_STATES + FINAL_TRACK_STATES
# The live status also reports "unknown" while no recorder is attached yet.
LIVE_TRACK_STATES = TRACK_STATES + ("unknown",)
# Top-level states of the capture source under GET /api/live -> "pipewire".
# "paused" is deliberately absent: pause is an engine-session state.
LIVE_STATES = ("recording", "stopping", "stopped", "failed")

# ---------------------------------------------------------------------------
# Signal verdicts (analyze_wav(): peak below SILENT_PEAK_DB = silent, below
# QUIET_PEAK_DB = quiet, >= CLIPPED_FS_FRACTION of samples at the format
# maximum = clipped, otherwise ok)
# ---------------------------------------------------------------------------

VERDICTS = ("ok", "quiet", "silent", "clipped")

# ---------------------------------------------------------------------------
# Key shapes: the exact fields each object carries today. Adding a field is
# a contract change: update these tuples and docs/CAPTURE_CONTRACT.md
# together, then refresh the fixtures.
# ---------------------------------------------------------------------------

REPORT_KEYS = ("ok", "out_dir", "rate", "targets", "tracks")
TARGET_KEYS = ("id", "serial", "name", "media_class", "description",
               "track", "selection")
TRACK_KEYS = ("track", "track_id", "target", "state", "file", "duration",
              "samplerate", "channels", "frames", "rms_db", "peak_db",
              "fs_fraction", "verdict", "error", "exit_code", "log")
LIVE_STATUS_KEYS = ("active", "state", "out_dir", "rate", "targets", "tracks",
                    "devices", "report")
LIVE_TRACK_STATUS_KEYS = ("state", "frames", "seconds", "error", "file")
# L7 device lifecycle: the additive ``devices`` block of the live status.
# The state/cause/event *strings* are produced by ``listn.device_lifecycle``
# (validated below); these are the shapes they must have.
LIVE_DEVICES_KEYS = ("active", "tracks", "defaults", "events", "retargets")
LIVE_DEVICE_TRACK_KEYS = ("state", "cause", "node", "media_class", "signal",
                          "reconnectable", "recovery_attempts",
                          "max_recovery_attempts", "replacement_candidates",
                          "detail")
LIVE_DEVICE_DEFAULT_KEYS = ("observed", "remembered")
DEVICE_EVENT_KEYS = ("t", "kind", "track", "node", "detail")
DEVICE_CANDIDATE_KEYS = ("name", "media_class", "description")


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------

def source_kind(target) -> str | None:
    """The source kind of a capture target, from its PipeWire media class.

    ``None`` for an unknown/absent class — a kind is never guessed. L1's
    application streams (``Stream/Output/Audio``) already map to ``app``.
    """
    if not isinstance(target, dict):
        return None
    media_class = target.get("media_class")
    if not isinstance(media_class, str):
        return None
    return MEDIA_CLASS_KINDS.get(media_class)


def stable_identity(target) -> str:
    """The target's stable identity: its PipeWire node name.

    PipeWire recycles node ids (a vanished id can come back as a different
    node with a new serial), so the name is what selection, the liveness
    probe and any persisted choice key on. ``id``/``serial`` are
    companions to the name, never the identity alone.
    """
    if not isinstance(target, dict):
        return ""
    name = target.get("name")
    return name if isinstance(name, str) else ""


def expected_ok(report) -> bool:
    """The normative ``ok`` policy for a final two-track report.

    ``ok`` is true only when **both** tracks reached ``stopped`` with no
    ``error`` **and** the microphone's verdict is not ``silent``. A silent
    system track alone stays ok (nobody may have been playing anything) —
    the asymmetry is the decision; everything else that failed is visible
    in the per-track states and never washed out by ``ok``.
    """
    if not isinstance(report, dict):
        return False
    tracks = report.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != set(TRACK_ORDER):
        return False
    for name in TRACK_ORDER:
        track = tracks.get(name)
        if not isinstance(track, dict):
            return False
        if track.get("state") != "stopped" or track.get("error") is not None:
            return False
    mic = tracks["mic"]
    if not isinstance(mic, dict):
        return False
    return mic.get("verdict") != "silent"


# ---------------------------------------------------------------------------
# Validators: a list of human-readable violations, empty when the object
# honours the contract. Additive unknown keys are tolerated (extensions add
# fields); missing keys, wrong values and policy drift are not.
# ---------------------------------------------------------------------------

def validate_report(report) -> list[str]:
    """Check a final two-track report against the contract."""
    if not isinstance(report, dict):
        return ["report: must be an object"]
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        return [f"report: missing keys {missing}"]
    violations: list[str] = []
    if not isinstance(report["ok"], bool):
        violations.append("report.ok: must be a bool")
    if not isinstance(report["out_dir"], str) or not report["out_dir"]:
        violations.append("report.out_dir: must be a non-empty local path")
    rate = report["rate"]
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        violations.append("report.rate: must be a positive int")
        rate = None
    tracks = report.get("tracks")
    violations += _validate_targets(report.get("targets"), tracks)
    violations += _validate_tracks(tracks, rate)
    if isinstance(report["ok"], bool):
        wanted = expected_ok(report)
        if report["ok"] != wanted:
            violations.append(
                f"report.ok: is {report['ok']!r} but the ok policy says {wanted!r} "
                "(both tracks stopped cleanly and the mic carried signal)")
    return violations


def _validate_targets(targets, tracks) -> list[str]:
    if not isinstance(targets, dict):
        return ["report.targets: must be an object"]
    violations: list[str] = []
    for name in TRACK_ORDER:
        if name not in targets:
            violations.append(f"report.targets: missing {name!r}")
    if violations:
        return violations
    for name in TRACK_ORDER:
        target = targets[name]
        prefix = f"report.targets.{name}"
        if not isinstance(target, dict):
            violations.append(f"{prefix}: must be an object")
            continue
        missing = [key for key in TARGET_KEYS if key not in target]
        if missing:
            violations.append(f"{prefix}: missing keys {missing}")
            continue
        if target["track"] != name:
            violations.append(
                f"{prefix}.track: must be {name!r}, got {target['track']!r}")
        if target["selection"] not in SELECTION_LABELS:
            violations.append(
                f"{prefix}.selection: {target['selection']!r} not in {SELECTION_LABELS}")
        if not stable_identity(target):
            violations.append(
                f"{prefix}.name: stable identity missing (non-empty node name)")
        node_id = target["id"]
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            violations.append(f"{prefix}.id: must be an int")
        serial = target["serial"]
        if serial is not None and (not isinstance(serial, int)
                                   or isinstance(serial, bool)):
            violations.append(f"{prefix}.serial: must be an int or null")
        description = target["description"]
        if description is not None and not isinstance(description, str):
            violations.append(f"{prefix}.description: must be a string or null")
        kind = source_kind(target)
        allowed = SYSTEM_KINDS if name == "system" else MIC_KINDS
        if kind is None:
            violations.append(
                f"{prefix}: media_class {target['media_class']!r} has no source kind")
        elif kind not in allowed:
            violations.append(
                f"{prefix}: source kind {kind!r} is not allowed on the {name} "
                f"track (allowed: {allowed})")
        if isinstance(tracks, dict) and isinstance(tracks.get(name), dict):
            if tracks[name].get("target") != target:
                violations.append(
                    f"report.tracks.{name}.target: must equal report.targets.{name}")
    return violations


def _validate_tracks(tracks, rate) -> list[str]:
    if not isinstance(tracks, dict):
        return ["report.tracks: must be an object"]
    violations: list[str] = []
    for name in TRACK_ORDER:
        if name not in tracks:
            violations.append(f"report.tracks: missing {name!r}")
    for name in TRACK_ORDER:
        track = tracks.get(name)
        if not isinstance(track, dict):
            continue
        prefix = f"report.tracks.{name}"
        missing = [key for key in TRACK_KEYS if key not in track]
        if missing:
            violations.append(f"{prefix}: missing keys {missing}")
            continue
        if track["track"] != name:
            violations.append(
                f"{prefix}.track: must be {name!r}, got {track['track']!r}")
        if track["track_id"] != TRACK_ROLES[name]:
            violations.append(
                f"{prefix}.track_id: must be {TRACK_ROLES[name]} for the "
                f"{name} track, got {track['track_id']!r}")
        state = track["state"]
        if state not in TRACK_STATES:
            violations.append(f"{prefix}.state: {state!r} not in {TRACK_STATES}")
        elif state not in FINAL_TRACK_STATES:
            violations.append(
                f"{prefix}.state: a final report only carries terminal states, "
                f"got {state!r}")
        verdict = track["verdict"]
        if verdict is not None and verdict not in VERDICTS:
            violations.append(f"{prefix}.verdict: {verdict!r} not in {VERDICTS}")
        error = track["error"]
        if error is not None and (not isinstance(error, str) or not error):
            violations.append(
                f"{prefix}.error: must be a non-empty string or null")
        if state == "failed" and not error:
            violations.append(
                f"{prefix}: a failed track must state its cause (error)")
        if verdict is None and state != "failed":
            violations.append(
                f"{prefix}: no verdict means no analysable signal data, "
                "which is never a clean stop")
        frames = track["frames"]
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
            violations.append(f"{prefix}.frames: must be a non-negative int")
            frames = 0
        file_path = track["file"]
        if file_path is not None and not isinstance(file_path, str):
            violations.append(f"{prefix}.file: must be a path string or null")
        log_path = track["log"]
        if log_path is not None and not isinstance(log_path, str):
            violations.append(f"{prefix}.log: must be a path string or null")
        exit_code = track["exit_code"]
        if exit_code is not None and (not isinstance(exit_code, int)
                                      or isinstance(exit_code, bool)):
            violations.append(f"{prefix}.exit_code: must be an int or null")
        if state == "stopped":
            if not isinstance(file_path, str) or not file_path:
                violations.append(
                    f"{prefix}: a stopped track must have a written WAV file")
            if frames <= 0:
                violations.append(
                    f"{prefix}: a stopped track must contain frames "
                    "(an empty capture is a failure, never a clean stop)")
            if verdict is None:
                violations.append(f"{prefix}: a stopped track must carry a verdict")
            if exit_code not in (0, 1, 130):
                violations.append(
                    f"{prefix}.exit_code: a clean stop leaves 0, 1 or 130, "
                    f"got {exit_code!r}")
        samplerate = track["samplerate"]
        if not isinstance(samplerate, int) or isinstance(samplerate, bool) \
                or samplerate < 0:
            violations.append(f"{prefix}.samplerate: must be a non-negative int")
            samplerate = 0
        if frames > 0:
            if rate is not None and samplerate != rate:
                violations.append(
                    f"{prefix}.samplerate: {samplerate} does not match the "
                    f"report rate {rate}")
            duration = track["duration"]
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                violations.append(f"{prefix}.duration: must be a number")
            elif samplerate and abs(duration - frames / samplerate) > 0.01:
                violations.append(
                    f"{prefix}.duration: {duration} does not match "
                    f"{frames}/{samplerate}")
        fs_fraction = track["fs_fraction"]
        if not isinstance(fs_fraction, (int, float)) or isinstance(fs_fraction, bool) \
                or not 0.0 <= fs_fraction <= 1.0:
            violations.append(
                f"{prefix}.fs_fraction: must be a number in [0, 1]")
        for key in ("rms_db", "peak_db"):
            value = track[key]
            if value is not None and (not isinstance(value, (int, float))
                                      or isinstance(value, bool)):
                violations.append(f"{prefix}.{key}: must be a number or null")
        channels = track["channels"]
        if not isinstance(channels, int) or isinstance(channels, bool) \
                or channels < 0:
            violations.append(f"{prefix}.channels: must be a non-negative int")
    return violations


def validate_live_status(status) -> list[str]:
    """Check the mid-run capture status served under GET /api/live."""
    if not isinstance(status, dict):
        return ["status: must be an object"]
    missing = [key for key in LIVE_STATUS_KEYS if key not in status]
    if missing:
        return [f"status: missing keys {missing}"]
    violations: list[str] = []
    active = status["active"]
    if not isinstance(active, bool):
        violations.append("status.active: must be a bool")
    state = status["state"]
    if state not in LIVE_STATES:
        violations.append(
            f"status.state: {state!r} not in {LIVE_STATES} "
            "(pause is an engine-session state, not a capture-source state)")
        return violations
    if not isinstance(status["out_dir"], str):
        violations.append("status.out_dir: must be a path string")
    rate = status["rate"]
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        violations.append("status.rate: must be a positive int")
    report = status["report"]
    if report is not None and not isinstance(report, dict):
        violations.append("status.report: must be an object or null")
    # Coherence with the recorder's lifecycle: the top-level state is
    # derived from (active, report) and nothing else.
    if state == "recording" and (active is not True or report is not None):
        violations.append(
            "status.state: 'recording' requires active=true and no final report")
    if state == "stopping" and (active is not False or report is not None):
        violations.append(
            "status.state: 'stopping' requires active=false and no final report yet")
    if state in ("stopped", "failed"):
        if active is not False or not isinstance(report, dict):
            violations.append(
                f"status.state: {state!r} requires active=false and a final report")
        else:
            wanted = "stopped" if report.get("ok") is True else "failed"
            if state != wanted:
                violations.append(
                    f"status.state: {state!r} disagrees with the report "
                    f"(ok={report.get('ok')!r} means {wanted!r})")
    targets = status["targets"]
    if targets is not None:
        if not isinstance(targets, dict):
            violations.append("status.targets: must be an object or null")
        else:
            for name in TRACK_ORDER:
                if name not in targets:
                    violations.append(f"status.targets: missing {name!r}")
    tracks = status["tracks"]
    if not isinstance(tracks, dict):
        violations.append("status.tracks: must be an object")
        tracks = {}
    for name in TRACK_ORDER:
        if name not in tracks:
            violations.append(f"status.tracks: missing {name!r}")
    for name in TRACK_ORDER:
        track = tracks.get(name)
        if not isinstance(track, dict):
            continue
        prefix = f"status.tracks.{name}"
        missing = [key for key in LIVE_TRACK_STATUS_KEYS if key not in track]
        if missing:
            violations.append(f"{prefix}: missing keys {missing}")
            continue
        track_state = track["state"]
        if track_state not in LIVE_TRACK_STATES:
            violations.append(
                f"{prefix}.state: {track_state!r} not in {LIVE_TRACK_STATES}")
            continue
        if state == "recording" and track_state not in ("pending", "recording",
                                                        "unknown"):
            violations.append(
                f"{prefix}.state: {track_state!r} while the capture is recording")
        if state in ("stopped", "failed") and track_state not in (
                "stopped", "failed", "unknown"):
            violations.append(
                f"{prefix}.state: {track_state!r} after the capture finished")
        frames = track["frames"]
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
            violations.append(f"{prefix}.frames: must be a non-negative int")
        seconds = track["seconds"]
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) \
                or seconds < 0:
            violations.append(f"{prefix}.seconds: must be a non-negative number")
        error = track["error"]
        if error is not None and (not isinstance(error, str) or not error):
            violations.append(f"{prefix}.error: must be a non-empty string or null")
        file_path = track["file"]
        if file_path is not None and not isinstance(file_path, str):
            violations.append(f"{prefix}.file: must be a path string or null")
    violations += _validate_devices(status.get("devices"), state)
    if isinstance(report, dict):
        violations += [f"status.report.{item}" if not item.startswith("report")
                       else item for item in validate_report(report)]
    return violations


def _validate_devices(devices, state: str) -> list[str]:
    """Check the L7 ``devices`` block: shape, vocabulary, coherence.

    The state/cause/event strings come from the producer
    (``listn.device_lifecycle``), imported lazily so the contract module
    stays importable on its own and no import cycle is created.
    """
    from listn import device_lifecycle as dl
    if not isinstance(devices, dict):
        return ["status.devices: must be an object"]
    missing = [key for key in LIVE_DEVICES_KEYS if key not in devices]
    if missing:
        return [f"status.devices: missing keys {missing}"]
    violations: list[str] = []
    active = devices["active"]
    if not isinstance(active, bool):
        violations.append("status.devices.active: must be a bool")
    # Coherence with the capture itself: a finished capture must not keep
    # claiming live device tracking, and a recording one must be tracking.
    if state in ("stopped", "failed") and active is not False:
        violations.append(
            f"status.devices.active: must be false once the capture is {state}")
    if state == "recording" and active is not True:
        violations.append(
            "status.devices.active: must be true while the capture is recording")
    tracks = devices["tracks"]
    if not isinstance(tracks, dict):
        violations.append("status.devices.tracks: must be an object")
        tracks = {}
    for name in TRACK_ORDER:
        if name not in tracks:
            violations.append(f"status.devices.tracks: missing {name!r}")
    for name, track in tracks.items():
        if not isinstance(track, dict):
            violations.append(f"status.devices.tracks.{name}: must be an object")
            continue
        prefix = f"status.devices.tracks.{name}"
        missing = [key for key in LIVE_DEVICE_TRACK_KEYS if key not in track]
        if missing:
            violations.append(f"{prefix}: missing keys {missing}")
            continue
        if track["state"] not in dl.DEVICE_STATES:
            violations.append(
                f"{prefix}.state: {track['state']!r} not in {dl.DEVICE_STATES}")
        if track["cause"] not in dl.DEVICE_CAUSES:
            violations.append(
                f"{prefix}.cause: {track['cause']!r} not in the device causes")
        signal = track["signal"]
        if signal is not None and signal not in VERDICTS:
            violations.append(
                f"{prefix}.signal: {signal!r} not in {VERDICTS} or null")
        if not isinstance(track["reconnectable"], bool):
            violations.append(f"{prefix}.reconnectable: must be a bool")
        attempts = track["recovery_attempts"]
        budget = track["max_recovery_attempts"]
        for key, value in (("recovery_attempts", attempts),
                           ("max_recovery_attempts", budget)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                violations.append(f"{prefix}.{key}: must be a non-negative int")
        if isinstance(attempts, int) and isinstance(budget, int) \
                and isinstance(attempts, bool) is False and attempts > budget:
            violations.append(f"{prefix}.recovery_attempts: exceeds the budget")
        candidates = track["replacement_candidates"]
        if not isinstance(candidates, list):
            violations.append(f"{prefix}.replacement_candidates: must be a list")
        else:
            for candidate in candidates:
                if not isinstance(candidate, dict) or any(
                        key not in candidate for key in DEVICE_CANDIDATE_KEYS):
                    violations.append(
                        f"{prefix}.replacement_candidates: every entry needs "
                        f"{DEVICE_CANDIDATE_KEYS}")
                    break
    defaults = devices["defaults"]
    if not isinstance(defaults, dict):
        violations.append("status.devices.defaults: must be an object")
    else:
        missing = [key for key in LIVE_DEVICE_DEFAULT_KEYS if key not in defaults]
        if missing:
            violations.append(f"status.devices.defaults: missing keys {missing}")
        else:
            for kind, value in defaults.items():
                if value is not None and not isinstance(value, dict):
                    violations.append(
                        f"status.devices.defaults.{kind}: must be an object or null")
    events = devices["events"]
    if not isinstance(events, list):
        violations.append("status.devices.events: must be a list")
    else:
        for event in events:
            if not isinstance(event, dict) or any(
                    key not in event for key in DEVICE_EVENT_KEYS):
                violations.append(
                    f"status.devices.events: every event needs {DEVICE_EVENT_KEYS}")
                break
            if event["kind"] not in dl.DEVICE_EVENT_KINDS:
                violations.append(
                    f"status.devices.events: unknown kind {event['kind']!r}")
                break
            if event["track"] is not None and event["track"] not in TRACK_ORDER:
                violations.append(
                    "status.devices.events: event track must be a track name "
                    "or null")
                break
    retargets = devices["retargets"]
    if not isinstance(retargets, int) or isinstance(retargets, bool) \
            or retargets < 0:
        violations.append("status.devices.retargets: must be a non-negative int")
    return violations
