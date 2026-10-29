"""Prototype two-track capture for Linux desktops through PipeWire.

Roadmap step 2 (docs/DECISIONS.md): record the microphone and the system
output as independent tracks with the desktop's own audio stack, without
touching the macOS capture path or the shared engine.

How it fits together
--------------------
* ``discover()`` reads the PipeWire graph with ``pw-dump`` and
  ``select_targets()`` picks an ``Audio/Sink`` (system track, recorded
  through the sink's monitor ports) and an ``Audio/Source`` (mic track).
* ``TwoTrackRecorder`` runs one ``pw-record`` process per track with raw
  PCM on stdout. Each track is written to its own WAV file and every block
  can be handed to an ``on_block`` callback — the same shape as the
  engine's ``LiveSession.push_audio(track, sample_rate, samples)`` used by
  the macOS capture client, so the existing ingest pipeline, pause
  semantics and silence detection apply unchanged.
* While recording, the target node of each track is watched with
  ``pw-dump <id>``. PipeWire 1.4.2 ``pw-record`` does not stop when its
  target disappears: it silently switches to another source (verified on a
  headless PipeWire daemon, see docs/LINUX_CAPTURE.md). A vanished target
  therefore fails that track and stops its process instead of quietly
  recording the wrong audio.

Honesty rules
-------------
* ``report["ok"]`` is true only when both tracks stopped cleanly, wrote a
  finalized WAV with more than zero frames, kept their target node, and the
  mic captured actual signal. Silence is never reported as success: each
  track carries its own ``verdict`` (``ok``/``quiet``/``silent``).
* Nothing here was verified on real audio hardware or a KDE session; the
  automated tests use a fake ``pw-record`` and synthetic audio. The runtime
  probe (``python -m listn.pipewire``) is the tool for real-hardware checks
  and says so in its output.

Limitations of this increment: no KDE tray, no web-UI wiring, no per-app
capture, no pause of its own (pause belongs to the engine session that
consumes ``on_block``), and no GPU or model changes.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from listn.audio import WavWriter, to_mono
from listn.capture import dbfs
from listn.ipc import TRACK_MIC, TRACK_SYSTEM

DEFAULT_RATE = 48000
SYSTEM_CHANNELS = 2
MIC_CHANNELS = 1
STOP_GRACE_SECONDS = 5.0
NODE_WATCH_INTERVAL = 1.0
# Exit codes a pw-record we signalled ourselves is allowed to leave with
# (observed 1 on SIGINT with PipeWire 1.4.2; 0 and 130 are common variants).
CLEAN_STOP_CODES = (0, 1, 130)
SILENT_PEAK_DB = -60.0
QUIET_PEAK_DB = -35.0
TRACK_NAMES = {TRACK_SYSTEM: "system", TRACK_MIC: "mic"}


class CaptureError(RuntimeError):
    """The capture cannot start or continue; message says what to do."""


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    serial: int | None = None
    media_class: str | None = None
    description: str | None = None
    priority: int | None = None
    monitor_of: int | None = None  # set for *.monitor source nodes

    @property
    def is_sink(self) -> bool:
        return self.media_class == "Audio/Sink"

    @property
    def is_source(self) -> bool:
        return self.media_class == "Audio/Source"

    @property
    def is_monitor(self) -> bool:
        return self.monitor_of is not None or self.name.endswith(".monitor")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "serial": self.serial,
            "name": self.name,
            "media_class": self.media_class,
            "description": self.description,
        }


@dataclass
class Graph:
    nodes: list[Node]
    default_sink: str | None = None
    default_source: str | None = None

    @property
    def sinks(self) -> list[Node]:
        return [n for n in self.nodes if n.is_sink and not n.is_monitor]

    @property
    def sources(self) -> list[Node]:
        return [n for n in self.nodes if n.is_source and not n.is_monitor]

    def by_name(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def by_id(self, node_id: int) -> Node | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


@dataclass(frozen=True)
class TrackTarget:
    track: str
    track_id: int
    node: Node
    selection: str  # "explicit" | "default" | "fallback"

    def to_dict(self) -> dict:
        return {**self.node.to_dict(), "track": self.track, "selection": self.selection}


@dataclass(frozen=True)
class Targets:
    system: TrackTarget
    mic: TrackTarget

    def to_dict(self) -> dict:
        return {"system": self.system.to_dict(), "mic": self.mic.to_dict()}


class CommandRunner:
    """The only side-effectful process layer; tests substitute this."""

    def run(self, argv: list[str], timeout: float | None = 10.0) -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def spawn(self, argv: list[str], **kwargs) -> subprocess.Popen:
        return subprocess.Popen(argv, **kwargs)


def parse_graph(payload) -> Graph:
    """Parse ``pw-dump`` JSON into nodes and default node names.

    Handles both layouts seen in the wild: ``props`` on the object (the
    ``Metadata`` interface) and ``props`` nested under ``info`` (``Node``).
    """
    if not isinstance(payload, list):
        raise CaptureError("pw-dump returned unexpected JSON (expected a list).")
    nodes: list[Node] = []
    defaults: dict[str, str | None] = {"sink": None, "source": None}
    for obj in payload:
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type", "")
        props = obj.get("props") or (obj.get("info") or {}).get("props") or {}
        if kind == "PipeWire:Interface:Node":
            media_class = props.get("media.class")
            name = props.get("node.name")
            if not name:
                continue
            serial = props.get("object.serial")
            priority = props.get("priority.session")
            try:
                node = Node(
                    id=int(obj.get("id", -1)),
                    name=str(name),
                    serial=int(serial) if serial is not None else None,
                    media_class=str(media_class) if media_class else None,
                    description=(str(props["node.description"]) if props.get("node.description") else None),
                    priority=int(priority) if priority is not None else None,
                    monitor_of=int(props["monitor.sink"]) if props.get("monitor.sink") is not None else None,
                )
            except (TypeError, ValueError):
                continue
            nodes.append(node)
        elif kind == "PipeWire:Interface:Metadata":
            entries = obj.get("metadata") or []
            if (props.get("metadata.name") or "") != "default":
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key")
                value = entry.get("value")
                if key not in ("default.audio.sink", "default.audio.source"):
                    continue
                name = None
                if isinstance(value, dict):
                    name = value.get("name")
                    if name is None and value.get("id") is not None:
                        by_id = _node_name_by_id(nodes, value.get("id"))
                        name = by_id
                elif isinstance(value, str):
                    name = value
                if name:
                    defaults["sink" if key.endswith("sink") else "source"] = str(name)
    if not nodes:
        raise CaptureError("pw-dump listed no PipeWire nodes; the audio graph looks empty.")
    return Graph(nodes=nodes, default_sink=defaults["sink"], default_source=defaults["source"])


def _node_name_by_id(nodes: list[Node], node_id) -> str | None:
    try:
        wanted = int(node_id)
    except (TypeError, ValueError):
        return None
    for node in nodes:
        if node.id == wanted:
            return node.name
    return None


def discover(runner=None, timeout: float = 10.0) -> Graph:
    """Read the live PipeWire graph. Raises ``CaptureError`` when unavailable."""
    if runner is None:
        if shutil.which("pw-dump") is None:
            raise CaptureError("pw-dump not found. Install the PipeWire tools (for example "
                               "'pipewire-bin' or 'pipewire' from your distribution).")
        runner = CommandRunner()
    try:
        proc = runner.run(["pw-dump"], timeout=timeout)
    except FileNotFoundError as exc:
        raise CaptureError(f"pw-dump could not be executed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CaptureError("pw-dump timed out; is the PipeWire session daemon running?") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise CaptureError("pw-dump failed (exit %s): %s"
                           % (proc.returncode, detail[-1] if detail else "no error output"))
    try:
        payload = json.loads(proc.stdout or "")
    except json.JSONDecodeError as exc:
        raise CaptureError(f"pw-dump did not return JSON: {exc}") from exc
    return parse_graph(payload)


def _resolve(selector: str | None, graph: Graph) -> Node | None:
    if selector is None:
        return None
    if isinstance(selector, int) or str(selector).isdigit():
        node = graph.by_id(int(selector))
        if node is not None:
            return node
    node = graph.by_name(str(selector))
    if node is None:
        available = ", ".join(sorted(n.name for n in graph.nodes if n.media_class)) or "none"
        raise CaptureError(f"No PipeWire node named '{selector}'. Nodes seen: {available}.")
    return node


def _best(nodes: list[Node]) -> Node | None:
    if not nodes:
        return None
    return sorted(nodes, key=lambda n: n.priority or 0, reverse=True)[0]


def select_targets(graph: Graph, system: str | None = None, mic: str | None = None) -> Targets:
    """Choose the two capture targets, refusing to guess silently.

    The default source recorded by the session manager can point at a sink
    (observed with WirePlumber 0.5.8 on a machine without microphones), so
    every default is re-checked against the node's media class before use.
    """
    sinks = graph.sinks
    sources = graph.sources

    chosen = _resolve(system, graph)
    if chosen is not None:
        if not chosen.is_sink:
            raise CaptureError(f"System target '{chosen.name}' is a {chosen.media_class or 'unknown'} "
                               "node, not an Audio/Sink.")
        system_target = TrackTarget("system", TRACK_SYSTEM, chosen, "explicit")
    else:
        by_default = graph.by_name(graph.default_sink) if graph.default_sink else None
        if by_default is not None and by_default.is_sink and not by_default.is_monitor:
            system_target = TrackTarget("system", TRACK_SYSTEM, by_default, "default")
        else:
            fallback = _best(sinks)
            if fallback is None:
                raise CaptureError("No Audio/Sink node found, so system audio cannot be captured. "
                                   "Check that PipeWire sees an output device.")
            system_target = TrackTarget("system", TRACK_SYSTEM, fallback, "fallback")

    chosen = _resolve(mic, graph)
    if chosen is not None:
        if not chosen.is_source or chosen.is_monitor:
            raise CaptureError(f"Microphone target '{chosen.name}' is a {chosen.media_class or 'unknown'} "
                               "node, not an Audio/Source input.")
        mic_target = TrackTarget("mic", TRACK_MIC, chosen, "explicit")
    else:
        by_default = graph.by_name(graph.default_source) if graph.default_source else None
        if by_default is not None and by_default.is_source and not by_default.is_monitor:
            mic_target = TrackTarget("mic", TRACK_MIC, by_default, "default")
        else:
            fallback = _best(sources)
            if fallback is None:
                raise CaptureError(
                    "No Audio/Source node found, so the microphone cannot be captured. "
                    "Connect or enable an input device; the session default "
                    f"({graph.default_source or 'unset'}) does not point at an input.")
            mic_target = TrackTarget("mic", TRACK_MIC, fallback, "fallback")
    return Targets(system=system_target, mic=mic_target)


def analyze_wav(path: Path | str) -> dict:
    """Inspect a captured WAV without trusting its header.

    A file whose writer was killed mid-recording keeps placeholder sizes
    (``data`` = 0, ``RIFF`` = 8 with PipeWire 1.4.2) and the stdlib ``wave``
    module refuses to open it, so the chunks are walked manually and an
    unfinalized file is reported as such instead of raising.
    """
    result = {
        "file": None, "duration": 0.0, "samplerate": 0, "channels": 0,
        "frames": 0, "rms_db": None, "peak_db": None, "verdict": None, "error": None,
    }
    p = Path(path)
    if not p.is_file():
        result["error"] = "no file was written"
        return result
    result["file"] = str(p)
    size = p.stat().st_size
    with open(p, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            result["error"] = "not a WAV file"
            return result
        fmt = None
        data_offset = None
        declared = None
        offset = 12
        while offset + 8 <= size:
            fh.seek(offset)
            raw = fh.read(8)
            if len(raw) < 8:
                break
            cid, csize = raw[:4], int.from_bytes(raw[4:], "little")
            if cid == b"fmt " and csize >= 16:
                body = fh.read(16)
                if len(body) < 16:
                    break
                fmt = (int.from_bytes(body[2:4], "little"),
                       int.from_bytes(body[4:8], "little"),
                       int.from_bytes(body[12:14], "little"))
            elif cid == b"data":
                data_offset = offset + 8
                declared = csize
                break
            offset += 8 + csize + (csize & 1)
    if fmt is None or data_offset is None:
        result["error"] = "WAV file has no fmt/data chunk"
        return result
    channels, rate, width = fmt
    if channels < 1 or rate < 1 or width != 2:
        result["error"] = f"unsupported WAV format ({channels}ch {rate}Hz {width * 8}-bit)"
        return result
    available = max(0, size - data_offset)
    payload = None
    if declared is None or declared == 0:
        if available:
            result["error"] = "recording was not finalized (header sizes were never written)"
            payload = available
        else:
            result["error"] = "recording was not finalized and contains no audio"
    elif declared > available:
        result["error"] = "recording is truncated (header claims more audio than the file holds)"
        payload = available
    else:
        payload = declared
    if not payload:
        return result
    frame_bytes = channels * 2
    frames = payload // frame_bytes
    with open(p, "rb") as fh:
        fh.seek(data_offset)
        raw = fh.read(frames * frame_bytes)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    result.update({
        "samplerate": rate, "channels": channels, "frames": int(frames),
        "duration": round(frames / rate, 3) if rate else 0.0,
    })
    if samples.size:
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        result["rms_db"] = round(dbfs(rms), 1)
        result["peak_db"] = round(dbfs(peak), 1)
        if result["peak_db"] < SILENT_PEAK_DB:
            result["verdict"] = "silent"
        elif result["peak_db"] < QUIET_PEAK_DB:
            result["verdict"] = "quiet"
        else:
            result["verdict"] = "ok"
    return result


def iter_blocks(path: Path | str, block_seconds: float = 0.2):
    """Yield ``(sample_rate, block)`` float32 blocks from a captured WAV.

    ``block`` is shaped ``(frames, channels)`` for stereo and ``(frames,)``
    for mono — exactly what ``LiveSession.push_audio`` accepts.
    """
    import wave

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise CaptureError(f"{Path(path).name} is not 16-bit PCM.")
        block_frames = max(1, int(rate * block_seconds))
        while True:
            raw = wf.readframes(block_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels)
            yield rate, samples


@dataclass
class _Track:
    target: TrackTarget
    path: Path
    channels: int
    log_path: Path
    proc: subprocess.Popen | None = None
    writer: WavWriter | None = None
    thread: threading.Thread | None = None
    state: str = "pending"  # pending | recording | stopped | failed
    error: str | None = None
    exit_code: int | None = None
    stop_requested: bool = False
    target_lost: bool = False
    killed: bool = False
    frames: int = 0
    pending: bytearray = field(default_factory=bytearray)


class TwoTrackRecorder:
    """Record the system track and the microphone track as separate files.

    ``runner`` supplies ``run``/``spawn`` (defaults to :class:`CommandRunner`),
    ``probe(node) -> bool`` reports whether the target node still exists
    (defaults to a ``pw-dump <id>`` identity check), and ``on_block`` receives
    every PCM block as ``(track_id, sample_rate, block)``.
    """

    def __init__(self, out_dir: Path | str, targets: Targets, *,
                 rate: int = DEFAULT_RATE,
                 system_channels: int = SYSTEM_CHANNELS,
                 mic_channels: int = MIC_CHANNELS,
                 runner=None, probe=None, on_block=None,
                 pw_record: str = "pw-record",
                 node_watch_interval: float = NODE_WATCH_INTERVAL,
                 stop_grace: float = STOP_GRACE_SECONDS):
        if int(rate) <= 0:
            raise CaptureError(f"Bad sample rate {rate}.")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.out_dir, 0o700)
        except OSError:
            pass
        self.targets = targets
        self.rate = int(rate)
        self.runner = runner or CommandRunner()
        self.probe = probe or (lambda node: _node_present(self.runner, node))
        self.on_block = on_block
        self.pw_record = pw_record
        self.node_watch_interval = float(node_watch_interval)
        self.stop_grace = float(stop_grace)
        self.state = "idle"
        self._tracks: dict[str, _Track] = {
            "system": _Track(targets.system, self.out_dir / "system.wav", system_channels,
                             self.out_dir / "system.pw-record.log"),
            "mic": _Track(targets.mic, self.out_dir / "mic.wav", mic_channels,
                          self.out_dir / "mic.pw-record.log"),
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.state != "idle":
            raise CaptureError(f"Capture already {self.state}.")
        started: list[_Track] = []
        try:
            for track in self._tracks.values():
                if not self.probe(track.target.node):
                    raise CaptureError(
                        f"The {track.target.track} target '{track.target.node.name}' is already "
                        "gone from the PipeWire graph. Re-run discovery.")
                argv = [
                    self.pw_record, "--raw",
                    "--target", track.target.node.name,
                    "--rate", str(self.rate),
                    "--channels", str(track.channels),
                    "--format", "s16", "-",
                ]
                log = open(track.log_path, "wb")
                try:
                    track.proc = self.runner.spawn(
                        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=log, bufsize=0)
                finally:
                    log.close()
                track.writer = WavWriter(track.path, rate=self.rate)
                track.state = "recording"
                started.append(track)
        except FileNotFoundError as exc:
            self._rollback(started)
            raise CaptureError(
                "pw-record not found. Install the PipeWire tools (for example 'pipewire-bin') "
                "and make sure pw-record is on PATH.") from exc
        except CaptureError:
            self._rollback(started)
            raise
        except OSError as exc:
            self._rollback(started)
            raise CaptureError(f"Could not start pw-record: {exc}") from exc
        for track in started:
            track.thread = threading.Thread(
                target=self._read_track, name=f"listn-pw-{track.target.track}",
                args=(track,), daemon=True)
            track.thread.start()
        self.state = "recording"

    def stop(self, grace: float | None = None) -> dict:
        """Stop both tracks and return the honest per-track report."""
        if self.state == "idle":
            raise CaptureError("Capture was never started.")
        if self.state == "stopped":
            return self.report()
        grace = self.stop_grace if grace is None else float(grace)
        for track in self._tracks.values():
            track.stop_requested = True
        for track in self._tracks.values():
            proc = track.proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
        deadline = time.monotonic() + grace
        for track in self._tracks.values():
            thread = track.thread
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for track in self._tracks.values():
            thread = track.thread
            proc = track.proc
            if thread is not None and thread.is_alive():
                # pw-record ignored SIGINT: kill it, the reader sees EOF.
                track.killed = True
                track.error = ("pw-record did not stop on SIGINT and was force-killed; "
                               "the track is not trustworthy")
                if proc is not None and proc.poll() is None:
                    try:
                        proc.kill()
                    except (ProcessLookupError, OSError):
                        pass
                thread.join(timeout=5.0)
            if proc is not None and proc.returncode is None:
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            if track.writer is not None:
                track.writer.close()
        self.state = "stopped"
        return self.report()

    def report(self) -> dict:
        tracks = {}
        for name, track in self._tracks.items():
            analysis = analyze_wav(track.path)
            if track.state == "stopped" and analysis["error"]:
                track.state = "failed"
                track.error = track.error or analysis["error"]
            if track.state == "stopped" and not analysis["frames"]:
                track.state = "failed"
                track.error = "no audio was captured"
            if track.state == "stopped" and track.exit_code not in (None,) + CLEAN_STOP_CODES:
                track.state = "failed"
                track.error = track.error or f"pw-record exited with code {track.exit_code}"
            if track.state == "stopped" and track.exit_code is None:
                track.state = "failed"
                track.error = track.error or "pw-record exit status could not be determined"
            tracks[name] = {
                "track": name,
                "track_id": track.target.track_id,
                "target": track.target.to_dict(),
                "state": track.state,
                "file": analysis["file"],
                "duration": analysis["duration"],
                "samplerate": analysis["samplerate"],
                "channels": analysis["channels"],
                "frames": analysis["frames"],
                "rms_db": analysis["rms_db"],
                "peak_db": analysis["peak_db"],
                "verdict": analysis["verdict"],
                "error": track.error,
                "exit_code": track.exit_code,
                "log": str(track.log_path) if track.log_path.exists() else None,
            }
        mic = tracks["mic"]
        ok = (all(t["state"] == "stopped" and t["error"] is None for t in tracks.values())
              and mic["verdict"] != "silent")
        return {
            "ok": ok,
            "out_dir": str(self.out_dir),
            "rate": self.rate,
            "targets": self.targets.to_dict(),
            "tracks": tracks,
        }

    # -- internals ---------------------------------------------------------

    def _rollback(self, started: list[_Track]) -> None:
        for track in started:
            proc = track.proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if track.writer is not None:
                track.writer.close()
        self.state = "idle"

    def _read_track(self, track: _Track) -> None:
        """Drain stdout into the WAV (and ``on_block``) until EOF.

        Also watches the target node every ``node_watch_interval`` seconds:
        PipeWire keeps pw-record running — silently re-targeted — when the
        node it was recording disappears, so the process is stopped here.
        """
        proc = track.proc
        assert proc is not None, "reader started without a process"
        stdout = proc.stdout
        assert stdout is not None, "reader started without stdout"
        last_probe = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if (not track.stop_requested and not track.target_lost
                        and self.node_watch_interval > 0
                        and now - last_probe >= self.node_watch_interval):
                    last_probe = now
                    if not self.probe(track.target.node):
                        track.target_lost = True
                        track.error = (f"target node '{track.target.node.name}' disappeared from "
                                       "the PipeWire graph while recording; stopped this track "
                                       "instead of recording another source")
                        self._terminate(proc)
                ready, _, _ = select.select([stdout], [], [], 0.2)
                if not ready:
                    continue
                try:
                    data = os.read(stdout.fileno(), 65536)
                except OSError:
                    break
                if not data:
                    break
                self._consume(track, data)
        except Exception as exc:  # a reader bug must fail the track, not vanish
            track.error = track.error or f"reader failed: {exc}"
        finally:
            try:
                stdout.close()
            except OSError:
                pass
        if proc.returncode is None:
            try:
                track.exit_code = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                track.exit_code = None
        else:
            track.exit_code = proc.returncode
        if track.writer is not None:
            track.writer.close()
        self._finish_state(track)

    def _consume(self, track: _Track, data: bytes) -> None:
        track.pending.extend(data)
        sample_bytes = (len(track.pending) // 2) * 2
        if track.channels > 1:
            sample_bytes -= sample_bytes % (track.channels * 2)
        if sample_bytes == 0:
            return
        chunk = bytes(track.pending[:sample_bytes])
        del track.pending[:sample_bytes]
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        block = samples.reshape(-1, track.channels) if track.channels > 1 else samples
        if track.writer is not None:
            track.writer.write(to_mono(block) if track.channels > 1 else block)
        track.frames += int(block.shape[0])
        if self.on_block is not None:
            try:
                self.on_block(track.target.track_id, self.rate, block)
            except Exception as exc:  # a dead consumer must not kill the file capture
                if track.error is None:
                    track.error = f"on_block callback failed: {exc}"

    def _finish_state(self, track: _Track) -> None:
        if track.target_lost or track.killed:
            track.state = "failed"
            return
        if not track.stop_requested:
            track.state = "failed"
            tail = _log_tail(track.log_path)
            code = f" (exit code {track.exit_code})" if track.exit_code is not None else ""
            track.error = f"pw-record for the {track.target.track} track exited by itself{code}"
            if tail:
                track.error += f": {tail}"
            return
        track.state = "stopped"

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass


def _node_present(runner, node: Node) -> bool:
    """True when the node still exists with the same identity.

    ``pw-dump <id>`` prints nothing and exits 0 for a vanished id
    (PipeWire 1.4.2), and ids are recycled, so the serial and name are
    re-checked to catch a reused id.
    """
    try:
        proc = runner.run(["pw-dump", str(node.id)], timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return True  # uncertainty must not kill a healthy recording
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        # rc 0 + empty output = the id is gone. A connect error (daemon
        # down) exits non-zero, and the pw-record process fails on its own.
        return proc.returncode != 0
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, list) or not payload:
        return False
    props = payload[0].get("props") or (payload[0].get("info") or {}).get("props") or {}
    serial = props.get("object.serial")
    if serial is not None and node.serial is not None and int(serial) != int(node.serial):
        return False
    name = props.get("node.name")
    if name is not None and node.name and str(name) != node.name:
        return False
    return True


def _log_tail(path: Path, limit: int = 300) -> str:
    try:
        text = path.read_bytes()[-limit:].decode("utf-8", "replace").strip()
    except OSError:
        return ""
    return " ".join(text.split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m listn.pipewire",
        description="Prototype: record the system output and the microphone as separate "
                    "PipeWire tracks. Local files only; nothing is uploaded.")
    parser.add_argument("--duration", type=float, default=5.0, metavar="SEC",
                        help="Seconds to record (default 5)")
    parser.add_argument("--out", metavar="DIR",
                        help="Directory for system.wav/mic.wav (default: a fresh temp directory)")
    parser.add_argument("--system", metavar="NODE",
                        help="System track target: PipeWire node name or id (default: session default sink)")
    parser.add_argument("--mic", metavar="NODE",
                        help="Microphone target: PipeWire node name or id (default: session default source)")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE, metavar="HZ",
                        help=f"Sample rate (default {DEFAULT_RATE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only discover and print the selected targets")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = discover()
        targets = select_targets(graph, args.system, args.mic)
    except CaptureError as exc:
        print(f"PipeWire capture unavailable: {exc}", file=sys.stderr)
        return 3
    if args.dry_run:
        if args.json:
            print(json.dumps(targets.to_dict(), indent=2))
        else:
            for name, target in targets.to_dict().items():
                print(f"{name}: {target['name']} ({target['media_class']}, id {target['id']}, "
                      f"selection: {target['selection']})")
        return 0
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="listn-pipewire-"))
    recorder = TwoTrackRecorder(out_dir, targets, rate=args.rate)
    # Status chatter goes to stderr so --json stdout stays machine-readable.
    print(f"Recording system + mic for up to {args.duration:g}s into {out_dir}", file=sys.stderr)
    print("Prototype capture: verified against a headless PipeWire daemon with synthetic "
          "audio only — not yet validated on real hardware or a KDE session.", file=sys.stderr)
    try:
        recorder.start()
    except CaptureError as exc:
        print(f"Capture unavailable: {exc}", file=sys.stderr)
        return 3
    try:
        end = time.monotonic() + max(0.0, args.duration)
        while time.monotonic() < end:
            time.sleep(min(0.2, max(0.0, end - time.monotonic())))
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    report = recorder.stop()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, track in report["tracks"].items():
            verdict = (track["verdict"] or "no signal data").upper()
            line = (f"{name}: {track['state']} {track['duration']:.2f}s -> {track['file'] or 'no file'} "
                    f"[{verdict}")
            if track["peak_db"] is not None:
                line += f" peak {track['peak_db']} dBFS"
            line += "]"
            if track["error"]:
                line += f"  error: {track['error']}"
            print(line)
        if report["ok"]:
            print("Capture OK: both tracks recorded with signal. "
                  "This validates the pipeline on this machine only.")
        else:
            reasons = [f"{n}: {t['error'] or t['verdict'] or t['state']}"
                       for n, t in report["tracks"].items()
                       if t["state"] != "stopped" or t["verdict"] == "silent"]
            print("Capture NOT OK — " + "; ".join(reasons))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
