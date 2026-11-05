"""Wire ``listn.pipewire`` into the app's CLI and local web server.

``listn/pipewire.py`` is reused exactly as written — ``discover()``,
``select_targets()`` and ``TwoTrackRecorder`` own discovery, target
validation and capture; nothing here re-implements their logic. This
module is the adaptation layer the CLI and the server share:

* :func:`discover_devices` reports the live node list so the UI can ask
  the user to pick targets explicitly. It never invents a target: when
  the tools or the graph are unusable it says so and the caller must not
  start.
* :func:`start_live` starts the recorder *first* (anything that fails
  there has started nothing else) and then feeds every PCM block into
  the engine's ingest seam — ``LiveSession.push_audio(track, rate,
  samples)`` — the same path the macOS capture client uses, so pause
  semantics, the independent system/mic tracks, per-track WAVs and the
  transcription pipeline apply unchanged.
* :class:`PipeWireLive` owns a running capture: per-track status while
  recording (read from the recorder's own state, not re-derived), a
  monitor that surfaces a track that died on its own, and a stop that
  returns the recorder's honest report instead of a claim.

Local-first: only ``pw-dump``/``pw-record`` subprocesses on this
machine; no network, no telemetry.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

from listn.config import Settings, data_dir
from listn.device_lifecycle import (CAUSE_NODE_LOST, CAUSE_PROCESS_EXIT,
                                    DeviceLifecycle)
from listn.ipc import TRACK_MIC, TRACK_SYSTEM
from listn.pipewire import (DEFAULT_RATE, CaptureError, Graph,
                            TwoTrackRecorder, discover, select_targets)

TRACK_NAMES = {TRACK_SYSTEM: "system", TRACK_MIC: "mic"}
TRACK_ORDER = ("system", "mic")
MONITOR_INTERVAL = 1.0
# How often the graph itself is re-read while capturing: hotplug, a
# replugged device, a changed session default and a daemon that went away
# are all only visible in a fresh ``pw-dump``.
GRAPH_WATCH_INTERVAL = 2.0
PIPEWIRE_DIR_NAME = "pipewire"
STREAM_MEDIA_CLASS = "Stream/Output/Audio"


def tools_missing() -> list[str]:
    """Names of the PipeWire CLI tools not found on PATH."""
    return [name for name in ("pw-dump", "pw-record") if shutil.which(name) is None]


def discover_devices(runner=None) -> dict:
    """Describe the live PipeWire graph for an explicit target picker.

    Always returns a report — ``available`` is false with a readable
    ``error`` when the tools or the graph are unusable, and ``usable``
    says whether an ``Audio/Sink`` / ``Audio/Source`` pair exists at all.
    Nothing here selects a target.
    """
    missing = tools_missing()
    base = {
        "available": not missing,
        "sinks": [],
        "sources": [],
        "defaults": {"sink": None, "source": None},
        "usable": {"system": False, "mic": False},
        "error": None,
    }
    if missing:
        base["error"] = (
            "PipeWire tools not found: " + ", ".join(missing) +
            ". Install the PipeWire tools (for example 'pipewire-bin' or "
            "'pipewire' from your distribution)."
        )
        return base
    try:
        graph = discover(runner=runner)
    except CaptureError as exc:
        base["available"] = False
        base["error"] = str(exc)
        return base
    base["sinks"] = [n.to_dict() for n in graph.sinks]
    base["sources"] = [n.to_dict() for n in graph.sources]
    base["defaults"] = {"sink": graph.default_sink, "source": graph.default_source}
    base["usable"] = {"system": bool(base["sinks"]), "mic": bool(base["sources"])}
    if not base["usable"]["system"] or not base["usable"]["mic"]:
        missing_kinds = [k for k in ("system", "mic") if not base["usable"][k]]
        base["error"] = (
            "PipeWire has no usable node for: " + ", ".join(missing_kinds) +
            ". No recording was started."
        )
    return base


class PipeWireLive:
    """A running two-track capture wired into one engine live session."""

    def __init__(self, recorder: TwoTrackRecorder | None, engine, out_dir: Path,
                 rate: int = DEFAULT_RATE):
        self.recorder = recorder
        self.engine = engine
        self.session = None
        self.out_dir = Path(out_dir)
        self.rate = int(rate)
        self.report: dict | None = None
        self.active = True
        self._stop_lock = threading.Lock()
        self._frames: dict[str, int] = {"system": 0, "mic": 0}
        self._dead_reported: set[str] = set()
        self._device_noted: set[str] = set()
        self._last_graph: Graph | None = None
        self.lifecycle = DeviceLifecycle(stream_alive=self._stream_alive)
        # Armed from the first instant the object exists: the status may be
        # read between construction and the recorder's start, and it must
        # never disagree with the capture's own activity flag.
        self.lifecycle.start({})
        bus = getattr(engine, "bus", None)
        self._events = bus.subscribe() if bus is not None else None
        self._graph_due = 0.0
        self._watcher = threading.Thread(
            target=self._watch, name="listn-pw-watch", daemon=True)
        self._watcher.start()

    @property
    def targets(self):
        return self.recorder.targets if self.recorder is not None else None

    def _stream_alive(self, name: str) -> bool:
        """True only when this track really has a live ``pw-record``."""
        track = getattr(self.recorder, "_tracks", {}).get(name) \
            if self.recorder is not None else None
        if track is None or track.proc is None:
            return False
        return track.state == "recording" and track.proc.poll() is None

    # -- audio seam ---------------------------------------------------------

    def on_block(self, track: int, sample_rate: int, block) -> None:
        """Recorder callback: count the block, then hand it to the session.

        Never raises — the recorder treats a failing callback as a track
        error, and a session that is not attached yet only misses the
        first few milliseconds of audio (the engine pads that gap with
        silence from its own ingest start time).
        """
        name = TRACK_NAMES.get(track)
        if name is None:
            return
        try:
            frames = int(block.shape[0])
        except Exception:
            return
        self._frames[name] = self._frames.get(name, 0) + frames
        session = self.session
        if session is not None:
            try:
                session.push_audio(track, sample_rate, block)
            except Exception:
                pass  # the session is stopping; the recorder still writes its WAV

    def attach_session(self, session) -> None:
        self.session = session

    # -- status -------------------------------------------------------------

    def _track_state(self, name: str) -> tuple[str, str | None]:
        """The recorder's own per-track state and error, or a safe default."""
        track = getattr(self.recorder, "_tracks", {}).get(name) if self.recorder else None
        if track is None:
            return "unknown", None
        return track.state, track.error

    def status(self) -> dict:
        tracks = {}
        for name in TRACK_ORDER:
            state, error = self._track_state(name)
            frames = self._frames.get(name, 0)
            track = getattr(self.recorder, "_tracks", {}).get(name) if self.recorder else None
            tracks[name] = {
                "state": state,
                "frames": frames,
                "seconds": round(frames / self.rate, 3) if self.rate else 0.0,
                "error": error,
                "file": str(track.path) if track is not None else None,
            }
        if self.report is not None:
            state = "stopped" if self.report.get("ok") else "failed"
        elif self.active:
            state = "recording"
        else:
            state = "stopping"
        return {
            "active": self.active,
            "state": state,
            "out_dir": str(self.out_dir),
            "rate": self.rate,
            "targets": self.targets.to_dict() if self.targets is not None else None,
            "tracks": tracks,
            "devices": self.lifecycle.status(),
            "report": self.report,
        }

    # -- lifecycle ----------------------------------------------------------

    def stop(self, wait: bool = False) -> dict:
        """Stop the recorder, then the session; return the honest report.

        Serialized: a user Stop and the failure monitor can arrive at the
        same time and only one of them performs the stop.
        """
        with self._stop_lock:
            self.active = False
            report = None
            if self.recorder is not None and self.recorder.state != "idle" \
                    and self.report is None:
                report = self.recorder.stop()
            # The final report is the last word on each track: fold its
            # verdicts/causes in and retire the lifecycle *before* the
            # report is published, so a status can never claim "stopped"
            # while the device block still says the capture is running.
            # A partial (node-lost) track keeps its cause after the stop.
            self.lifecycle.observe_report(report if report is not None
                                          else self.report)
            self.lifecycle.stop()
            if report is not None:
                self.report = report
            meta = self.engine.stop_live(wait=wait)
        if self._events is not None:
            try:
                self.engine.bus.unsubscribe(self._events)
            except Exception:
                pass
            self._events = None
        return {"id": meta.id if meta is not None else None,
                "pipewire": self.report}

    # -- monitor ------------------------------------------------------------

    def _publish(self, kind: str, message: str) -> None:
        if self.session is None:
            return
        try:
            self.engine.bus.publish(
                {"type": kind, "session": self.session.meta.id, "message": message})
        except Exception:
            pass  # a status event must never break the capture

    def _drain_bus(self) -> None:
        """Fold engine events (silence verdicts) into the device state."""
        events = self._events
        if events is None:
            return
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                return
            except Exception:
                self._events = None
                return
            if isinstance(event, dict) and event.get("type") == "live.silent":
                track = event.get("track")
                if track in TRACK_ORDER:
                    self.lifecycle.observe_signal(track, True)

    def _observe_devices(self) -> None:
        """Read the graph (or its failure) for hotplug/default/daemon state."""
        try:
            graph = discover()
        except Exception as exc:  # missing tool, daemon down, bad payload
            self.lifecycle.observe_graph(None, error=str(exc))
            self._last_graph = None
            return
        self._last_graph = graph
        self.lifecycle.observe_graph({
            "sinks": [n.to_dict() for n in graph.sinks],
            "sources": [n.to_dict() for n in graph.sources],
            "streams": [n.to_dict() for n in graph.nodes
                        if n.media_class == STREAM_MEDIA_CLASS],
            "defaults": {"sink": graph.default_sink,
                         "source": graph.default_source},
        })

    def _note_dead_tracks(self) -> None:
        """Tell the lifecycle why a track's stream is gone (once each)."""
        for name in TRACK_ORDER:
            if name in self._device_noted:
                continue
            state, error = self._track_state(name)
            if state != "failed":
                continue
            self._device_noted.add(name)
            track = getattr(self.recorder, "_tracks", {}).get(name)
            cause = CAUSE_NODE_LOST if getattr(track, "target_lost", False) \
                else CAUSE_PROCESS_EXIT
            self.lifecycle.note_stream_dead(name, cause, error)

    def replace_track(self, track: str, node_name: str) -> dict:
        """Resume a dead track on the node the user explicitly chose.

        L7's only retarget path: the graph is re-read, the lifecycle
        validates the request (right kind, not the sibling's target,
        recovery budget, no second stream, sibling still recording) and
        only then does the recorder spawn the replacement. Any refusal
        raises :class:`CaptureError` with the reason — nothing is
        retried or substituted automatically.
        """
        if track not in TRACK_ORDER:
            raise CaptureError(f"Unknown track '{track}'.")
        self._observe_devices()
        decision = self.lifecycle.begin_replacement(track, node_name)
        if not decision["ok"]:
            raise CaptureError(decision["reason"])
        node = self._last_graph.by_name(node_name) if self._last_graph else None
        if node is None:
            self.lifecycle.note_recovery_failed(
                track, f"'{node_name}' is not in the PipeWire graph.")
            raise CaptureError(f"'{node_name}' is not in the PipeWire graph.")
        if self.recorder is None:
            self.lifecycle.note_recovery_failed(track, "no recorder is attached.")
            raise CaptureError("No recorder is attached to this capture.")
        try:
            self.recorder.replace_track(track, node)
        except CaptureError as exc:
            self.lifecycle.note_recovery_failed(track, str(exc))
            raise
        self.lifecycle.commit_replacement(track, node.name)
        # It may die again: allow the next death to be noted and reported.
        self._device_noted.discard(track)
        self._dead_reported.discard(track)
        return self.status()

    def _watch(self) -> None:
        """Surface capture failures truthfully instead of recording on.

        * A track that dies on its own (``pw-record`` exited, target node
          vanished) is reported once via ``live.error`` while the other
          track keeps going — the recorder already failed that track.
        * The graph itself is re-read while capturing so a hotplugged or
          replugged device, a changed session default and a daemon that
          went away show up as explicit device state (never as a silent
          re-target).
        * When every track is dead, or the engine session ended first,
          the capture is stopped so the UI never shows "recording" for a
          dead pipeline, and the stored session carries the real cause.
        """
        while self.active:
            time.sleep(MONITOR_INTERVAL)
            if not self.active:
                return
            self._drain_bus()
            now = time.monotonic()
            if now >= self._graph_due:
                self._graph_due = now + GRAPH_WATCH_INTERVAL
                self._observe_devices()
            self._note_dead_tracks()
            if self.session is None:
                continue
            if not self.engine.live_active():
                # The session ended on its own (disk policy, engine error):
                # finalize the capture instead of leaving pw-record running.
                self.stop(wait=False)
                return
            states = {name: self._track_state(name)[0] for name in TRACK_ORDER}
            dead = [n for n, s in states.items()
                    if s not in ("recording", "pending", "unknown")]
            for name in dead:
                if name in self._dead_reported:
                    continue
                self._dead_reported.add(name)
                error = self._track_state(name)[1] or f"the {name} track stopped"
                self._publish("live.error",
                              f"PipeWire {name} track failed: {error}")
            if dead and len(dead) == len(states):
                self._abort()
                return

    def _abort(self) -> None:
        causes = []
        for name in TRACK_ORDER:
            _, error = self._track_state(name)
            if error:
                causes.append(f"{name}: {error}")
        message = ("PipeWire recording failed — " + "; ".join(causes)
                   if causes else "PipeWire recording failed — both tracks stopped.")
        result = self.stop(wait=True)
        if self.session is not None:
            # The engine's own finish already ran; make the live state and the
            # stored metadata say what actually happened instead of "Saved".
            self.session.state = "error"
        session_id = result.get("id")
        if session_id:
            try:
                self.engine.library.update(
                    session_id, status="error", error=message)
            except Exception:
                pass  # the report and the live.error still carry the cause
        self._publish("live.error", message)


def start_live(engine, settings: Settings, system: str | None, mic: str | None,
               title: str | None = None) -> PipeWireLive:
    """Start a two-track recording that feeds the engine's live session.

    ``system`` and ``mic`` must be explicit node names or ids — the UI
    collected them from :func:`discover_devices`. Discovery, target
    validation and capture belong to ``listn.pipewire``; its
    ``CaptureError`` propagates before anything has started.
    """
    if engine.live_active():
        raise RuntimeError("A recording is already running.")
    graph = discover()
    targets = select_targets(graph, system, mic)

    root = data_dir() / PIPEWIRE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S") + "-",
                                    dir=root))
    capture = PipeWireLive(None, engine, out_dir, rate=DEFAULT_RATE)
    recorder = TwoTrackRecorder(out_dir, targets, on_block=capture.on_block)
    capture.recorder = recorder
    try:
        recorder.start()
    except CaptureError:
        # Nothing else was started; do not leave an empty run directory behind.
        capture.lifecycle.stop()
        capture.active = False
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    capture.lifecycle.start(targets.to_dict())
    try:
        session = engine.start_ingest(
            settings, {"source": "system", "mic": True}, title)
    except Exception:
        recorder.stop()  # finalize whatever was captured before failing
        capture.lifecycle.stop()
        capture.active = False
        raise
    source_name = (f"PipeWire · {targets.system.node.name} "
                   f"+ {targets.mic.node.name}")
    session.meta.source_name = source_name
    info = dict(session.device_info or {})
    info["name"] = source_name
    info["kind"] = "pipewire"
    session.device_info = info
    session.library.save_meta(session.meta)
    capture.attach_session(session)
    return capture
