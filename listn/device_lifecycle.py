"""Device lifecycle policy for Linux (PipeWire) capture — L7.

The recorder already refuses to keep recording a node that vanished
(``listn.pipewire`` stops the affected track instead of letting
``pw-record`` silently re-target — behaviour pinned by L0's contract).
This module is the policy layer on top of that guard: it turns the raw
facts (a probe said the node is gone, ``pw-dump`` cannot connect, the
session default changed, a device came back) into an explicit,
machine-readable device state per track, and it owns the rules for what
may happen next.

Rules enforced here
-------------------
* **No silent retarget.** Nothing in this module ever changes a track's
  target on its own. A target changes only through
  :meth:`DeviceLifecycle.begin_replacement` + :meth:`commit_replacement`,
  which the caller performs because a user asked for it. ``status()
  ["retargets"]`` counts committed replacements so a test (or a reader)
  can assert it stayed ``0`` across every automatic event.
* **Bounded recovery.** Each committed replacement consumes one attempt;
  ``max_recovery_attempts`` per track per capture bounds the churn. When
  the budget is spent the track reports ``exhausted`` and further
  attempts are refused with a reason instead of retried.
* **Never two streams.** A replacement is refused while the track's own
  stream is still alive, and refused while the *other* track is not
  recording (the capture is ending anyway — L0's both-tracks-dead abort).
* **Defaults follow the user, not the desktop.** A session-default change
  is observed and reported; the remembered default selection is updated
  only while no capture is running.
* **Pause is not here.** Pause is an engine-session state (L0 §6): this
  class has no pause operation, so a pause/resume can neither reopen nor
  re-target a stream behind the user's back.

The vocabulary (states, causes, event kinds) is additive contract surface:
``listn.capture_contract`` re-exports it and ``docs/CAPTURE_CONTRACT.md``
documents it. Everything exposed is metadata — node names, descriptions,
states and local paths; never audio, titles, transcripts or URLs.

Local-only by construction: no network, no subprocess; the graph is handed
in by the caller (``pw-dump`` already ran) and this module only reads it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from listn.capture_contract import (DEVICE_EVENT_KEYS, MIC_KINDS,
                                    SYSTEM_KINDS, TRACK_ORDER,
                                    LIVE_DEVICES_KEYS,
                                    LIVE_DEVICE_DEFAULT_KEYS,
                                    LIVE_DEVICE_TRACK_KEYS, source_kind)

# ---------------------------------------------------------------------------
# Vocabulary — the exact strings status()["devices"] may carry
# ---------------------------------------------------------------------------

STATE_RECORDING = "recording"      # stream running, device present
STATE_STOPPED = "stopped"          # no capture is running
STATE_LOST = "lost"                # the target node vanished mid-run
STATE_NO_DEVICE = "no_device"      # graph holds nothing of the right class
STATE_UNAVAILABLE = "unavailable"  # PipeWire daemon / permission unavailable
STATE_RECOVERING = "recovering"    # an explicit replacement is in flight
STATE_EXHAUSTED = "exhausted"      # bounded recovery budget spent
STATE_REPLACED = "replaced"        # an explicit replacement is now recording
STATE_SILENT = "silent"            # device present, no signal
DEVICE_STATES = (STATE_RECORDING, STATE_STOPPED, STATE_LOST, STATE_NO_DEVICE,
                 STATE_UNAVAILABLE, STATE_RECOVERING, STATE_EXHAUSTED,
                 STATE_REPLACED, STATE_SILENT)

CAUSE_NODE_LOST = "node_lost"
CAUSE_PROCESS_EXIT = "process_exit"
CAUSE_DAEMON_UNAVAILABLE = "daemon_unavailable"
CAUSE_NO_DEVICE = "no_device"
CAUSE_SILENT_SIGNAL = "silent_signal"
CAUSE_RECOVERY_EXHAUSTED = "recovery_exhausted"
CAUSE_RECOVERY_IN_PROGRESS = "recovery_in_progress"
DEVICE_CAUSES = (None, CAUSE_NODE_LOST, CAUSE_PROCESS_EXIT,
                 CAUSE_DAEMON_UNAVAILABLE, CAUSE_NO_DEVICE,
                 CAUSE_SILENT_SIGNAL, CAUSE_RECOVERY_EXHAUSTED,
                 CAUSE_RECOVERY_IN_PROGRESS)

EVENT_CAPTURE_STARTED = "capture_started"
EVENT_CAPTURE_STOPPED = "capture_stopped"
EVENT_NODE_LOST = "node_lost"
EVENT_STREAM_STOPPED = "stream_stopped"
EVENT_NODE_REAPPEARED = "node_reappeared"
EVENT_DEFAULT_CHANGED = "default_changed"
EVENT_DAEMON_UNAVAILABLE = "daemon_unavailable"
EVENT_DAEMON_RESTORED = "daemon_restored"
EVENT_SIGNAL_SILENT = "signal_silent"
EVENT_SIGNAL_CLEAR = "signal_clear"
EVENT_RECOVERY_REFUSED = "recovery_refused"
EVENT_RECOVERY_STARTED = "recovery_started"
EVENT_RECOVERY_COMMITTED = "recovery_committed"
EVENT_KINDS = (EVENT_CAPTURE_STARTED, EVENT_CAPTURE_STOPPED, EVENT_NODE_LOST,
               EVENT_STREAM_STOPPED, EVENT_NODE_REAPPEARED,
               EVENT_DEFAULT_CHANGED, EVENT_DAEMON_UNAVAILABLE,
               EVENT_DAEMON_RESTORED, EVENT_SIGNAL_SILENT, EVENT_SIGNAL_CLEAR,
               EVENT_RECOVERY_REFUSED, EVENT_RECOVERY_STARTED,
               EVENT_RECOVERY_COMMITTED)
# Alias the contract validator uses (listn.capture_contract imports this
# name lazily inside validate_live_status).
DEVICE_EVENT_KINDS = EVENT_KINDS

# Machine-readable refusal reasons a client may branch on. The free-text
# ``reason`` that accompanies them is human wording, never a contract.
REFUSED_NOT_RUNNING = "capture_not_running"
REFUSED_STILL_RECORDING = "track_still_recording"
REFUSED_NOT_FAILED = "track_not_failed"
REFUSED_DAEMON_UNAVAILABLE = "daemon_unavailable"
REFUSED_ALREADY_RECOVERING = "recovery_in_progress"
REFUSED_ATTEMPTS_EXHAUSTED = "recovery_exhausted"
REFUSED_SIBLING_DEAD = "other_track_not_recording"
REFUSED_UNKNOWN_NODE = "unknown_node"
REFUSED_WRONG_KIND = "wrong_kind"
REFUSED_SIBLING_TARGET = "other_track_target"

DEFAULT_RECOVERY_ATTEMPTS = 2
DEFAULT_EVENT_LIMIT = 64
MAX_CANDIDATES = 8
# The status shapes are the contract's key tuples (listn.capture_contract),
# so a status built here is set-equal to them by construction.


def candidate_kinds(track: str) -> tuple[str, ...]:
    """Source kinds that may record into ``track`` (L0 §2.1)."""
    return MIC_KINDS if track == "mic" else SYSTEM_KINDS


@dataclass
class _Candidate:
    name: str
    media_class: str | None = None
    description: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "media_class": self.media_class,
                "description": self.description}


@dataclass
class TrackDevice:
    """The facts about one track's device; ``state`` is derived from them."""

    name: str
    node: str | None = None            # stable identity of the selected node
    media_class: str | None = None
    stream_dead: bool = False
    cause_fact: str | None = None      # why the stream died (fact, kept)
    detail: str | None = None          # human wording, not a contract
    daemon_down: bool = False
    graph_seen: bool = False
    graph_has_class: bool = False
    reconnectable: bool = False        # the same node name is back
    attempts: int = 0
    in_flight: bool = False
    replaced_with: str | None = None
    signal: str | None = None          # observed verdict, None while unknown
    started: bool = False

    @property
    def cause(self) -> str | None:
        """The machine cause matching the *current* state.

        Derived so state and cause can never disagree: an unavailable
        daemon explains itself, a lost track keeps the fact recorded when
        its stream died, and a healthy track carries none.
        """
        state = self.state
        if state == STATE_UNAVAILABLE:
            return CAUSE_DAEMON_UNAVAILABLE
        if state == STATE_NO_DEVICE:
            return CAUSE_NO_DEVICE
        if state == STATE_SILENT:
            return CAUSE_SILENT_SIGNAL
        if state == STATE_EXHAUSTED:
            return CAUSE_RECOVERY_EXHAUSTED
        if state == STATE_RECOVERING:
            return CAUSE_RECOVERY_IN_PROGRESS
        if state in (STATE_LOST, STATE_STOPPED):
            return self.cause_fact
        return None

    @property
    def state(self) -> str:
        if not self.started:
            return STATE_STOPPED
        if self.in_flight:
            return STATE_RECOVERING
        if self.stream_dead:
            if self.attempts >= self._max_attempts:
                return STATE_EXHAUSTED
            if self.daemon_down:
                return STATE_UNAVAILABLE
            if self.graph_seen and not self.graph_has_class:
                return STATE_NO_DEVICE
            return STATE_LOST
        if self.daemon_down:
            return STATE_UNAVAILABLE
        if self.signal == "silent":
            return STATE_SILENT
        if self.replaced_with is not None:
            return STATE_REPLACED
        return STATE_RECORDING

    # The derivation above needs the budget; the owning lifecycle injects it
    # before status() is read (kept off the dataclass so the facts stay a
    # plain, comparable record for tests).
    _max_attempts: int = DEFAULT_RECOVERY_ATTEMPTS

    def status(self, candidates: list[_Candidate]) -> dict:
        values = {
            "state": self.state,
            "cause": self.cause,
            "node": self.node,
            "media_class": self.media_class,
            "signal": self.signal,
            "reconnectable": bool(self.reconnectable),
            "recovery_attempts": int(self.attempts),
            "max_recovery_attempts": int(self._max_attempts),
            "replacement_candidates": [c.to_dict() for c in candidates],
            "detail": self.detail,
        }
        return {key: values[key] for key in LIVE_DEVICE_TRACK_KEYS}


class DeviceLifecycle:
    """Observe device events, expose device state, gate explicit recovery.

    Every mutation is driven by an injected observation —
    :meth:`observe_graph` (a ``pw-dump`` snapshot or its failure),
    :meth:`note_stream_dead` (the recorder's own liveness guard),
    :meth:`observe_signal` (the engine's silence verdict) — so tests can
    replay hotplug, default changes, daemon loss and process exits without
    touching real audio or a real desktop.
    """

    def __init__(self, *, tracks: tuple[str, ...] = TRACK_ORDER,
                 max_recovery_attempts: int = DEFAULT_RECOVERY_ATTEMPTS,
                 stream_alive=None, now=None,
                 event_limit: int = DEFAULT_EVENT_LIMIT):
        if int(max_recovery_attempts) < 1:
            raise ValueError("max_recovery_attempts must be >= 1")
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.event_limit = int(event_limit)
        self._now = now or time.monotonic
        self._t0 = self._now()
        self._stream_alive_cb = stream_alive or (lambda name: True)
        self._tracks: dict[str, TrackDevice] = {}
        for name in tracks:
            track = TrackDevice(name)
            track._max_attempts = self.max_recovery_attempts
            self._tracks[name] = track
        self._graph: dict | None = None
        self._daemon_down = False
        self._defaults_seen = False
        self._defaults = {"sink": None, "source": None}
        self._remembered = {"sink": None, "source": None}
        self._active = False
        self._events: list[dict] = []
        self.retargets = 0

    # -- capture lifecycle --------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(self._tracks)

    def start(self, targets: dict) -> None:
        """Arm the lifecycle for a capture about to record ``targets``."""
        self._active = True
        self.retargets = 0
        self._events.clear()
        self._t0 = self._now()
        for name, track in self._tracks.items():
            target = targets.get(name) or {}
            track.node = target.get("name")
            track.media_class = target.get("media_class")
            track.stream_dead = False
            track.cause_fact = None
            track.detail = None
            track.attempts = 0
            track.in_flight = False
            track.replaced_with = None
            track.signal = None
            track.reconnectable = False
            track.started = True
        self._emit(EVENT_CAPTURE_STARTED)

    def stop(self) -> None:
        """The capture ended: every track reports ``stopped`` (idempotent)."""
        if not self._active:
            return
        self._active = False
        for track in self._tracks.values():
            track.started = False
            track.in_flight = False
        self._emit(EVENT_CAPTURE_STOPPED)

    # -- injected observations ---------------------------------------------

    def note_stream_dead(self, track: str, cause: str, detail: str | None = None) -> None:
        """The recorder's guard killed this track's stream (or it exited)."""
        if cause not in DEVICE_CAUSES or cause is None:
            raise ValueError(f"unknown device cause {cause!r}")
        item = self._tracks.get(track)
        if item is None or not self._active:
            return
        if item.stream_dead and item.cause_fact == cause and item.detail == detail:
            return  # the graph watcher and the recorder saw the same death
        item.stream_dead = True
        item.in_flight = False
        item.cause_fact = cause
        item.detail = detail
        self._emit(EVENT_NODE_LOST if cause == CAUSE_NODE_LOST
                   else EVENT_STREAM_STOPPED, track=track, node=item.node,
                   detail=detail or cause)

    def observe_graph(self, graph: dict | None, *, error: str | None = None) -> None:
        """Fold one graph snapshot (or its failure) into the device state.

        ``graph`` is ``discover_devices()``'s shape: ``sinks``/``sources``
        (+ optional ``streams`` once L1 lands) of node dicts and a
        ``defaults`` mapping. ``error`` means ``pw-dump`` itself failed —
        tools missing, daemon down, permission denied — which is reported
        as *unavailable*, never as a dead device.
        """
        if error:
            if not self._daemon_down:
                self._daemon_down = True
                for track in self._tracks.values():
                    track.daemon_down = True
                self._emit(EVENT_DAEMON_UNAVAILABLE, detail=error)
            return
        if graph is None:
            return
        if self._daemon_down:
            self._daemon_down = False
            for track in self._tracks.values():
                track.daemon_down = False
            self._emit(EVENT_DAEMON_RESTORED)
        self._graph = graph
        self._observe_defaults(graph)
        for track in self._tracks.values():
            self._observe_presence(track, graph)

    def _observe_defaults(self, graph: dict) -> None:
        observed = graph.get("defaults") or {}
        for kind in ("sink", "source"):
            new = observed.get(kind)
            if new is None:
                continue
            if not self._defaults_seen:
                # First snapshot: this is the starting point, not a change.
                self._defaults[kind] = new
                self._remembered[kind] = new
                continue
            if new == self._defaults[kind]:
                continue
            old = self._defaults[kind]
            self._defaults[kind] = new
            if self._active:
                detail = (f"session default {kind} changed {old} -> {new}; "
                          "the running capture keeps its explicit target")
            else:
                self._remembered[kind] = new
                detail = f"remembered default {kind} updated to {new} (idle)"
            self._emit(EVENT_DEFAULT_CHANGED, node=new, detail=detail)
        self._defaults_seen = True

    def _observe_presence(self, track: TrackDevice, graph: dict) -> None:
        if not track.started:
            return
        nodes = self._nodes_for(track.name, graph)
        track.graph_seen = True
        track.graph_has_class = bool(nodes)
        if not track.stream_dead:
            track.reconnectable = False
            return
        back = any(n.get("name") == track.node for n in nodes)
        if back and not track.reconnectable:
            track.reconnectable = True
            self._emit(EVENT_NODE_REAPPEARED, track=track.name,
                       node=track.node,
                       detail=(f"the device is back in the graph; choose "
                               f"resume to record it again"))
        elif not back:
            track.reconnectable = False

    def observe_signal(self, track: str, silent: bool) -> None:
        """Fold the engine's silence verdict into the device state."""
        item = self._tracks.get(track)
        if item is None or not self._active:
            return
        value = "silent" if silent else "ok"
        if item.signal == value:
            return
        item.signal = value
        self._emit(EVENT_SIGNAL_SILENT if silent else EVENT_SIGNAL_CLEAR,
                   track=track, node=item.node)

    def observe_report(self, report: dict | None) -> None:
        """Fold a final report's verdicts in (signal facts survive the stop)."""
        if not isinstance(report, dict):
            return
        for name, track in (report.get("tracks") or {}).items():
            item = self._tracks.get(name)
            if item is None:
                continue
            verdict = track.get("verdict")
            if verdict in ("silent", "ok", "quiet", "clipped"):
                item.signal = verdict
            if track.get("state") == "failed" and not item.stream_dead:
                cause = (CAUSE_NODE_LOST
                         if "disappeared" in (track.get("error") or "")
                         else CAUSE_PROCESS_EXIT)
                item.stream_dead = True
                item.cause_fact = item.cause_fact or cause
                item.detail = item.detail or track.get("error")

    # -- explicit replacement (the only path that may retarget) -------------

    def begin_replacement(self, track: str, node_name: str) -> dict:
        """Decide whether the user's requested replacement may proceed.

        Returns ``{"ok": bool, "code": str|None, "reason": str|None,
        "node": dict|None}``. Nothing is mutated on success except the
        in-flight marker; the caller must :meth:`commit_replacement` after
        the new stream really started, or :meth:`note_recovery_failed` if
        it could not.
        """
        item = self._tracks.get(track)
        if item is None:
            return self._refuse(track, REFUSED_NOT_RUNNING, "unknown track.")
        if not self._active:
            return self._refuse(track, REFUSED_NOT_RUNNING,
                                "the capture is not running.")
        if item.in_flight:
            return self._refuse(track, REFUSED_ALREADY_RECOVERING,
                                "a replacement is already in progress.")
        if not item.stream_dead:
            code = (REFUSED_STILL_RECORDING if item.state in
                    (STATE_RECORDING, STATE_REPLACED, STATE_SILENT)
                    else REFUSED_NOT_FAILED)
            return self._refuse(track, code,
                                f"the {track} track is not waiting for a "
                                "replacement.")
        if self._daemon_down:
            return self._refuse(track, REFUSED_DAEMON_UNAVAILABLE,
                                "PipeWire is unavailable; the device cannot "
                                "be probed.")
        if item.attempts >= self.max_recovery_attempts:
            return self._refuse(track, REFUSED_ATTEMPTS_EXHAUSTED,
                                f"the {track} track already used its "
                                f"{self.max_recovery_attempts} recovery "
                                "attempts; stop and start a new recording.")
        other = self._other(track)
        if other is not None and not self._stream_alive(other.name):
            return self._refuse(track, REFUSED_SIBLING_DEAD,
                                f"the {other.name} track is not recording; "
                                "the capture is ending.")
        if self._stream_alive(track):
            return self._refuse(track, REFUSED_STILL_RECORDING,
                                f"the {track} track already has a stream.")
        node = self._find_node(node_name)
        if node is None:
            return self._refuse(track, REFUSED_UNKNOWN_NODE,
                                f"'{node_name}' is not in the PipeWire graph.")
        kind = source_kind(node)
        if kind not in candidate_kinds(track):
            return self._refuse(
                track, REFUSED_WRONG_KIND,
                f"'{node_name}' is a {kind or 'unknown'} node; the {track} "
                f"track records {' or '.join(candidate_kinds(track))}.")
        if other is not None and node.get("name") == other.node:
            return self._refuse(track, REFUSED_SIBLING_TARGET,
                                f"'{node_name}' is already the {other.name} "
                                "track's target.")
        item.in_flight = True
        self._emit(EVENT_RECOVERY_STARTED, track=track,
                   node=node.get("name"),
                   detail=f"replacement requested for the {track} track")
        return {"ok": True, "code": None, "reason": None, "node": dict(node)}

    def commit_replacement(self, track: str, node_name: str) -> None:
        """The replacement stream really started: record the retarget."""
        item = self._tracks.get(track)
        if item is None:
            return
        item.in_flight = False
        item.attempts += 1
        item.stream_dead = False
        item.cause_fact = None
        item.detail = None
        item.signal = None
        item.node = node_name or item.node
        item.replaced_with = item.node
        item.reconnectable = False
        self.retargets += 1
        self._emit(EVENT_RECOVERY_COMMITTED, track=track, node=item.node,
                   detail=f"recording the {track} track from a user-chosen "
                          "replacement device")

    def note_recovery_failed(self, track: str, reason: str) -> None:
        """The replacement could not start; the track stays failed."""
        item = self._tracks.get(track)
        if item is None:
            return
        item.in_flight = False
        self._emit(EVENT_RECOVERY_REFUSED, track=track, node=item.node,
                   detail=reason)

    # -- observations -------------------------------------------------------

    def _refuse(self, track: str, code: str, reason: str,
                node: str | None = None) -> dict:
        self._emit(EVENT_RECOVERY_REFUSED, track=track, node=node,
                   detail=reason)
        return {"ok": False, "code": code, "reason": reason, "node": None}

    def _other(self, track: str) -> TrackDevice | None:
        for name, item in self._tracks.items():
            if name != track:
                return item
        return None

    def _stream_alive(self, track: str) -> bool:
        try:
            return bool(self._stream_alive_cb(track))
        except Exception:
            return False

    def _nodes_for(self, track: str, graph: dict) -> list[dict]:
        kinds = candidate_kinds(track)
        pool: list[dict] = []
        for key in ("sinks", "sources", "streams"):
            value = graph.get(key)
            if isinstance(value, list):
                pool.extend(n for n in value if isinstance(n, dict))
        return [n for n in pool if source_kind(n) in kinds]

    def _find_node(self, node_name: str) -> dict | None:
        if not self._graph or not node_name:
            return None
        for key in ("sinks", "sources", "streams"):
            value = self._graph.get(key)
            if not isinstance(value, list):
                continue
            for node in value:
                if isinstance(node, dict) and node.get("name") == node_name:
                    return node
        return None

    def candidates_for(self, track: str) -> list[_Candidate]:
        """Devices the user may explicitly choose for a dead ``track``."""
        item = self._tracks.get(track)
        if item is None or self._graph is None or not item.started:
            return []
        if item.state not in (STATE_LOST, STATE_NO_DEVICE, STATE_EXHAUSTED,
                              STATE_UNAVAILABLE, STATE_RECOVERING):
            return []
        other = self._other(track)
        out: list[_Candidate] = []
        for node in self._nodes_for(track, self._graph):
            name = node.get("name")
            if not name or (other is not None and name == other.node):
                continue
            if name == item.node and not item.reconnectable:
                continue
            out.append(_Candidate(name=name,
                                  media_class=node.get("media_class"),
                                  description=node.get("description")))
            if len(out) >= MAX_CANDIDATES:
                break
        return out

    def _emit(self, kind: str, *, track: str | None = None,
              node: str | None = None, detail: str | None = None) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown device event {kind!r}")
        event = {"t": round(self._now() - self._t0, 1), "kind": kind,
                 "track": track, "node": node, "detail": detail}
        event = {key: event[key] for key in DEVICE_EVENT_KEYS}
        if self._events and all(self._events[-1][k] == event[k]
                                for k in ("kind", "track", "node")):
            return  # the same observation reported twice is one event
        self._events.append(event)
        if len(self._events) > self.event_limit:
            del self._events[:len(self._events) - self.event_limit]

    # -- exposure -----------------------------------------------------------

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def status(self) -> dict:
        tracks = {}
        for name, item in self._tracks.items():
            tracks[name] = item.status(self.candidates_for(name))
        defaults = {"observed": dict(self._defaults),
                    "remembered": dict(self._remembered)}
        values = {
            "active": self._active,
            "tracks": tracks,
            "defaults": {key: defaults[key] for key in LIVE_DEVICE_DEFAULT_KEYS},
            "events": list(self._events),
            "retargets": int(self.retargets),
        }
        return {key: values[key] for key in LIVE_DEVICES_KEYS}

    def state_of(self, track: str) -> str:
        item = self._tracks.get(track)
        return item.state if item is not None else STATE_STOPPED
