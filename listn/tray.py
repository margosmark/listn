"""KDE Plasma tray controls for a real two-track recording.

The tray is a *client* of the local server — the same loopback JSON API the
web UI uses (``/api/live``, ``/api/live/start`` …). It never re-implements
capture: start/stop/pause reach the recorder the parent increment wired up
(``listn/pipewire_live.py`` on the server), and every state the tray shows is
derived from the server's own report, never from the success of an HTTP call.

Three layers, so the interesting logic needs neither Qt nor a session bus:

* :class:`ServerClient` — stdlib HTTP against ``127.0.0.1`` only.
* :func:`derive_status` / :func:`build_menu` / :class:`TrayController` — a
  pure state machine: server JSON in, truthful status + minimal menu out.
  Failure is a first-class state with the recorder's own cause attached
  (dead track, refused start, engine gone) — silence or a stale "recording"
  is never shown for a dead pipeline.
* ``listn/tray_sni.py`` — StatusNotifierItem + com.canonical.dbusmenu over
  D-Bus (Plasma's tray protocol), a renderer of this model.

Scope honesty: the menu offers exactly start/stop, pause/resume, the target
pickers, open privacy settings, quit. The system-output picker covers both
capture modes of the same track — a whole sink ("all applications") or one
application stream — from the lists the local API returns, in one radio
group, so what is checked is what is recorded. The macOS-only or
unimplemented controls (``offer-meetings``, ``auto-record``,
``launch-at-login``) do not exist here — not disabled, absent. Privacy
settings on Linux is listn's own retention/audio settings page: PipeWire
has no macOS-style per-app permission list, and the tray does not pretend
otherwise.

Local-first: HTTP stays on loopback, D-Bus stays on the session bus; nothing
here opens a socket to anywhere else.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from listn.config import data_dir

LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 8765
POLL_SECONDS = 1.0

# Status names are the tray's vocabulary; they are derived, never guessed.
UNREACHABLE = "unreachable"   # the engine (server) is not answering
IDLE = "idle"
STARTING = "starting"
RECORDING = "recording"
PAUSED = "paused"
STOPPING = "stopping"
FAILED = "failed"

# Fixed menu item ids — the D-Bus layer dispatches on these.
ITEM_STATUS = 1
ITEM_START = 3
ITEM_STOP = 4
ITEM_PAUSE = 5
ITEM_RESUME = 6
ITEM_SYSTEM_BASE = 100   # system output entries: 100, 101, ...
ITEM_MIC_BASE = 200      # microphone entries: 200, 201, ...
ITEM_APP_BASE = 300      # application-stream entries: 300, 301, ...
# Disabled section header *inside* the System output submenu. Not a
# control: clicking it does nothing (no action branch owns this id).
ITEM_SYSTEM_APPS_SECTION = 12
ITEM_PRIVACY = 20
ITEM_QUIT = 30


class ServerUnreachable(ConnectionError):
    """The local engine did not answer; the tray says so instead of faking state."""


class RequestRefused(RuntimeError):
    """The server refused an action and returned a readable reason."""

    def __init__(self, detail: str, status: int | None = None):
        super().__init__(detail)
        self.detail = detail
        self.status = status


class ServerClient:
    """Minimal JSON client for the loopback API. No third-party HTTP dependency."""

    def __init__(self, port: int = DEFAULT_PORT, timeout: float = 5.0,
                 transport=None):
        self.base = f"http://{LOOPBACK}:{port}"
        self.timeout = timeout
        # Tests inject a transport with .get(path) / .post(path, body).
        self._transport = transport

    # -- plumbing ----------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if self._transport is not None:
            status, payload = (self._transport.get(path) if method == "GET"
                               else self._transport.post(path, body))
            if status >= 400:
                raise RequestRefused(_detail_of(payload), status)
            return payload
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read() or b"{}")
            except Exception:
                payload = {}
            raise RequestRefused(_detail_of(payload) or str(exc), exc.code) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ServerUnreachable(
                f"listn is not answering on {self.base} — start it with `listn ui` "
                f"or `listn tray` ({exc}).") from None

    # -- API ---------------------------------------------------------------

    def live(self) -> dict:
        return self._request("GET", "/api/live")

    def devices(self) -> dict:
        return self._request("GET", "/api/pipewire/devices")

    def start(self, system: str, mic: str, title: str | None = None) -> dict:
        return self._request("POST", "/api/live/start", {
            "capture": "pipewire", "system_target": system, "mic_target": mic,
            **({"title": title} if title else {})})

    def pause(self) -> dict:
        return self._request("POST", "/api/live/pause", {})

    def resume(self) -> dict:
        return self._request("POST", "/api/live/resume", {})

    def stop(self) -> dict:
        return self._request("POST", "/api/live/stop", {})


def _detail_of(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)
    return str(payload)


# ---------------------------------------------------------------------------
# Status: server JSON -> truthful tray state
# ---------------------------------------------------------------------------

@dataclass
class TrayStatus:
    state: str = UNREACHABLE
    detail: str | None = None          # real cause for failed/unreachable
    title: str | None = None
    elapsed: float | None = None
    system_target: str | None = None
    mic_target: str | None = None
    system_frames: int | None = None
    mic_frames: int | None = None
    session_id: str | None = None

    @property
    def active(self) -> bool:
        return self.state in (STARTING, RECORDING, PAUSED, STOPPING)

    def summary(self) -> str:
        """One truthful line for the disabled status item and the tooltip."""
        if self.state == UNREACHABLE:
            return self.detail or "listn is not running."
        if self.state == FAILED:
            return f"Failed — {self.detail}" if self.detail else "Failed."
        if self.state == IDLE:
            return "Idle. Nothing is recording."
        label = {STARTING: "Starting…", RECORDING: "Recording", PAUSED: "Paused",
                 STOPPING: "Stopping…"}[self.state]
        bits = [label]
        if self.title:
            bits.append(self.title)
        if self.elapsed is not None:
            minutes, seconds = divmod(int(self.elapsed), 60)
            bits.append(f"{minutes:02d}:{seconds:02d}")
        if self.state in (RECORDING, PAUSED):
            tracks = []
            if self.system_frames is not None:
                tracks.append(f"system {self.system_frames}f")
            if self.mic_frames is not None:
                tracks.append(f"mic {self.mic_frames}f")
            if tracks:
                bits.append("· ".join(tracks))
        return " — ".join(bits)


def derive_status(payload: dict | None, error: str | None = None) -> TrayStatus:
    """Map ``GET /api/live`` (plus a transport error, if any) to tray status.

    The recorder's own words win: a ``pipewire`` report with ``ok: false``
    becomes :data:`FAILED` with the per-track causes; a report with ``ok:
    true`` is :data:`IDLE` — the recording genuinely ended.
    """
    if error is not None:
        return TrayStatus(state=UNREACHABLE, detail=error)
    if payload is None:
        return TrayStatus(state=UNREACHABLE,
                          detail="listn is not answering.")
    live = payload.get("live")
    pw = payload.get("pipewire")
    status = TrayStatus()
    if isinstance(pw, dict):
        targets = pw.get("targets") or {}
        status.system_target = (targets.get("system") or {}).get("name")
        status.mic_target = (targets.get("mic") or {}).get("name")
        tracks = pw.get("tracks") or {}
        status.system_frames = (tracks.get("system") or {}).get("frames")
        status.mic_frames = (tracks.get("mic") or {}).get("frames")
        report = pw.get("report")
        if isinstance(report, dict):
            if report.get("ok"):
                status.state = IDLE
                return status
            causes = []
            for name in ("system", "mic"):
                track = (report.get("tracks") or {}).get(name) or {}
                reason = track.get("error") or track.get("verdict") or track.get("state")
                if track.get("state") != "stopped" or track.get("verdict") == "silent":
                    causes.append(f"{name}: {reason}")
            status.state = FAILED
            status.detail = "; ".join(causes) or "the capture ended without a report"
            return status
        if pw.get("state") == "failed":
            status.state = FAILED
            for name in ("system", "mic"):
                track = (tracks.get(name) or {})
                if track.get("error"):
                    status.detail = f"{name}: {track['error']}"
                    break
            if status.detail is None:
                status.detail = "both PipeWire tracks stopped."
            return status
    if not live:
        if isinstance(pw, dict) and pw.get("active"):
            # The recorder is already capturing but the engine session is
            # not attached yet (start_live starts the recorder first):
            # that is "starting", never a false "idle".
            status.state = STARTING
            return status
        status.state = IDLE
        return status
    status.title = live.get("title")
    status.session_id = live.get("id")
    if live.get("state") == "error":
        status.state = FAILED
        status.detail = live.get("error") or "the recording session failed."
        return status
    if live.get("state") == "done":
        # The server keeps the last session in /api/live after it ended
        # (engine.live_active() treats done/error as not running): the
        # recording genuinely ended — never show it as still recording.
        status.state = IDLE
        return status
    status.elapsed = live.get("elapsed")
    if live.get("paused") or live.get("state") == "paused":
        status.state = PAUSED
    elif live.get("state") == "finishing":
        status.state = STOPPING
    else:
        status.state = RECORDING
    return status


# ---------------------------------------------------------------------------
# Menu: exactly the supported controls, honestly enabled
# ---------------------------------------------------------------------------

@dataclass
class MenuItem:
    id: int
    label: str
    enabled: bool = True
    separator: bool = False
    children: list["MenuItem"] = field(default_factory=list)
    toggle: str | None = None       # "radio" for target picks
    toggled: bool = False


def build_menu(status: TrayStatus, sinks: list[dict], sources: list[dict],
               apps: list[dict], selected_system: str | None,
               selected_mic: str | None,
               discovery_error: str | None = None) -> list[MenuItem]:
    """The whole tray menu. Unimplemented macOS controls are absent, not greyed.

    ``sinks`` and ``apps`` are two halves of *one* system-track picker:
    whole-sink capture and a single application stream are alternatives,
    not layers, so they share the System output submenu and its radio
    group — exactly one entry in it can be checked, and whichever one is
    checked is exactly the selection ``start()`` sends. The section
    header is a disabled label, never a target.
    """
    items = [MenuItem(ITEM_STATUS, status.summary(), enabled=False)]
    if status.state == FAILED or status.state == UNREACHABLE:
        pass  # the cause is already in the summary line
    items.append(MenuItem(0, "", separator=True))

    can_start = (status.state in (IDLE, FAILED) and not status.active
                 and selected_system is not None and selected_mic is not None)
    items.append(MenuItem(ITEM_START, "Start recording", enabled=can_start))
    items.append(MenuItem(ITEM_STOP, "Stop",
                          enabled=status.state in (RECORDING, PAUSED)))
    items.append(MenuItem(ITEM_PAUSE, "Pause",
                          enabled=status.state == RECORDING))
    items.append(MenuItem(ITEM_RESUME, "Resume",
                          enabled=status.state == PAUSED))
    items.append(MenuItem(0, "", separator=True))

    system = MenuItem(10, "System output")
    if discovery_error:
        system.enabled = False
        system.label = f"System output — {discovery_error}"
    for index, node in enumerate(sinks):
        name = node.get("name") or "?"
        system.children.append(MenuItem(
            ITEM_SYSTEM_BASE + index, node.get("description") or name,
            toggle="radio", toggled=(name == selected_system)))
    if apps:
        # One picker, one radio group: an application stream is an
        # alternative to the whole sink for this same track, so it lives
        # in this submenu rather than in a group of its own — the host
        # and the controller can then never show both modes checked.
        # The label is the application's own name (parse_graph keeps
        # window/media titles out of it); identity stays the node name.
        if system.children:
            system.children.append(MenuItem(0, "", separator=True))
        system.children.append(MenuItem(
            ITEM_SYSTEM_APPS_SECTION, "Application streams", enabled=False))
        for index, node in enumerate(apps):
            name = node.get("name") or "?"
            system.children.append(MenuItem(
                ITEM_APP_BASE + index, node.get("description") or name,
                toggle="radio", toggled=(name == selected_system)))
    if not system.children and not discovery_error:
        system.enabled = False
        system.label = "System output — none found"
    mic = MenuItem(11, "Microphone")
    if discovery_error:
        mic.enabled = False
        mic.label = f"Microphone — {discovery_error}"
    for index, node in enumerate(sources):
        name = node.get("name") or "?"
        mic.children.append(MenuItem(
            ITEM_MIC_BASE + index, node.get("description") or name,
            toggle="radio", toggled=(name == selected_mic)))
    if not mic.children and not discovery_error:
        mic.enabled = False
        mic.label = "Microphone — none found"
    items.extend([system, mic, MenuItem(0, "", separator=True)])

    items.append(MenuItem(ITEM_PRIVACY, "Open privacy settings",
                          enabled=status.state != UNREACHABLE))
    items.append(MenuItem(0, "", separator=True))
    quit_label = ("Quit listn — stops the recording"
                  if status.active else "Quit listn")
    items.append(MenuItem(ITEM_QUIT, quit_label))
    return items


# ---------------------------------------------------------------------------
# Controller: menu clicks -> server calls -> refreshed status
# ---------------------------------------------------------------------------

class TrayController:
    """Owns the target selection and translates menu ids into real actions.

    Start requires both targets — the selection is explicit (session defaults
    are adopted only as a labelled suggestion the user can change, mirroring
    ``listn record``), and a refused start stays visibly failed/idle with the
    server's reason instead of pretending to record.

    The system selection is a single field holding either a sink's node
    name (whole-sink mode) or an application stream's node name (one
    application only). It is passed to the server unchanged: the tray
    never translates one mode into the other, and the report's
    ``targets.system.media_class`` is what tells them apart.
    """

    def __init__(self, client: ServerClient, on_quit=None,
                 targets_file: Path | None = None):
        self.client = client
        self.on_quit = on_quit
        self.status = TrayStatus(state=UNREACHABLE)
        self.sinks: list[dict] = []
        self.sources: list[dict] = []
        self.apps: list[dict] = []
        self.discovery_error: str | None = None
        self.selected_system: str | None = None
        self.selected_mic: str | None = None
        self.notice: str | None = None      # last refused action, shown once
        self.quit_requested = False
        self._targets_file = targets_file or (data_dir() / "tray-targets.json")
        self._load_targets()

    # -- persistence (selection only; never audio, never telemetry) --------

    def _load_targets(self) -> None:
        try:
            saved = json.loads(Path(self._targets_file).read_text())
            self.selected_system = saved.get("system")
            self.selected_mic = saved.get("mic")
        except (OSError, ValueError):
            pass

    def _save_targets(self) -> None:
        try:
            path = Path(self._targets_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"system": self.selected_system,
                                        "mic": self.selected_mic}))
        except OSError:
            pass  # a selection that cannot be remembered is not an error

    # -- polling -----------------------------------------------------------

    def refresh(self) -> TrayStatus:
        """Poll the server once. The status always reflects its answer."""
        error = None
        try:
            payload = self.client.live()
        except ServerUnreachable as exc:
            payload, error = None, str(exc)
        except RequestRefused as exc:
            payload, error = None, exc.detail
        self.status = derive_status(payload, error)
        if self.status.state != UNREACHABLE and not self.sinks and not self.discovery_error:
            self._load_devices()
        return self.status

    def _load_devices(self) -> None:
        """Adopt the session defaults as labelled suggestions, once."""
        try:
            devices = self.client.devices()
        except (ServerUnreachable, RequestRefused) as exc:
            self.discovery_error = exc.detail if isinstance(exc, RequestRefused) else None
            return
        self.sinks = devices.get("sinks") or []
        self.sources = devices.get("sources") or []
        self.apps = devices.get("apps") or []
        defaults = devices.get("defaults") or {}
        sink_names = {n.get("name") for n in self.sinks}
        source_names = {n.get("name") for n in self.sources}
        # The system track can point at a whole sink *or* at one
        # application stream; both name spaces come from one graph, so a
        # remembered selection is valid if it is in either list. A pick
        # that is still there is kept as the user made it — an
        # application selection is never quietly swapped back to a sink.
        system_names = sink_names | {n.get("name") for n in self.apps}
        # A session manager can point a default at the wrong media class —
        # observed with WirePlumber 0.5.8 on a machine without microphones,
        # where default.audio.source named an Audio/Sink. Every default is
        # re-checked against the node's own class before it is adopted, the
        # same rule `listn.pipewire.select_targets` applies.
        default_sink = defaults.get("sink") if defaults.get("sink") in sink_names else None
        default_mic = defaults.get("source") if defaults.get("source") in source_names else None
        if self.selected_system is None and default_sink:
            self.selected_system = default_sink
        if self.selected_mic is None and default_mic:
            self.selected_mic = default_mic
        if self.selected_system not in system_names:
            # The remembered node is gone — a sink or an application
            # stream that left the graph. Re-adopt only the graph's own
            # session default (the same label `listn record` prints); if
            # the graph offers none, selection becomes empty and Start
            # disables — the tray never falls back to "the first node it
            # saw", and it never invents a node the graph does not list.
            self.selected_system = default_sink
        if self.selected_mic not in source_names:
            self.selected_mic = default_mic

    def menu(self) -> list[MenuItem]:
        items = build_menu(self.status, self.sinks, self.sources, self.apps,
                           self.selected_system, self.selected_mic,
                           self.discovery_error)
        if self.notice and self.status.state in (IDLE, FAILED):
            # The last action was refused (or failed) — show the server's
            # own reason as the status line instead of a bare "Idle".
            items[0] = MenuItem(ITEM_STATUS,
                                f"{self.status.summary()} {self.notice}".strip(),
                                enabled=False)
        return items

    # -- actions -----------------------------------------------------------

    def action(self, item_id: int) -> None:
        if item_id == ITEM_START:
            self.start()
        elif item_id == ITEM_STOP:
            self.stop()
        elif item_id == ITEM_PAUSE:
            self.pause()
        elif item_id == ITEM_RESUME:
            self.resume()
        elif item_id == ITEM_PRIVACY:
            self.open_privacy()
        elif item_id == ITEM_QUIT:
            self.quit()
        elif ITEM_SYSTEM_BASE <= item_id < ITEM_MIC_BASE:
            index = item_id - ITEM_SYSTEM_BASE
            if 0 <= index < len(self.sinks):
                self.selected_system = self.sinks[index].get("name")
                self._save_targets()
        elif ITEM_APP_BASE <= item_id < ITEM_APP_BASE + 100:
            index = item_id - ITEM_APP_BASE
            if 0 <= index < len(self.apps):
                # Same field, other mode: the node name of the stream the
                # user picked. start() sends it verbatim, so the server
                # reports this selection — never a translated one.
                self.selected_system = self.apps[index].get("name")
                self._save_targets()
        elif ITEM_MIC_BASE <= item_id < ITEM_MIC_BASE + 100:
            index = item_id - ITEM_MIC_BASE
            if 0 <= index < len(self.sources):
                self.selected_mic = self.sources[index].get("name")
                self._save_targets()

    def _refused(self, exc: RequestRefused) -> None:
        # Keep the server's own reason visible; do not flip to a fake state.
        self.notice = exc.detail
        self.refresh()

    def start(self) -> None:
        if self.selected_system is None or self.selected_mic is None:
            self.notice = ("Pick a system output and a microphone first — "
                           "listn never guesses a capture target.")
            return
        self.status = TrayStatus(state=STARTING,
                                 system_target=self.selected_system,
                                 mic_target=self.selected_mic)
        try:
            self.client.start(self.selected_system, self.selected_mic)
        except RequestRefused as exc:
            self._refused(exc)
            return
        except ServerUnreachable as exc:
            self.status = derive_status(None, str(exc))
            return
        self._save_targets()
        self.notice = None
        self.refresh()

    def pause(self) -> None:
        self._simple(self.client.pause)

    def resume(self) -> None:
        self._simple(self.client.resume)

    def stop(self) -> None:
        self._simple(self.client.stop)

    def _simple(self, call) -> None:
        try:
            call()
        except RequestRefused as exc:
            self._refused(exc)
            return
        except ServerUnreachable as exc:
            self.status = derive_status(None, str(exc))
            return
        self.notice = None
        self.refresh()

    def open_privacy(self) -> str | None:
        """Open listn's own privacy/retention settings in the default browser.

        Linux has no macOS-style per-app permission list for PipeWire; the
        privacy surface that exists is listn's settings page, and that is
        what opens. Returns the URL for tests; None when it could not open.
        """
        import webbrowser
        url = f"{self.base_url()}/#settings"
        try:
            return url if webbrowser.open(url) else None
        except Exception:
            return None

    def base_url(self) -> str:
        return getattr(self.client, "base", f"http://{LOOPBACK}:{DEFAULT_PORT}")

    def quit(self) -> None:
        """Stop any active recording first — quitting never leaves a silent
        recorder running, and the quit item's label says so."""
        if self.status.active:
            try:
                self.client.stop()
            except (RequestRefused, ServerUnreachable):
                pass
        self.quit_requested = True
        if self.on_quit is not None:
            self.on_quit()


def port_in_use(port: int, host: str = LOOPBACK) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) == 0


def start_embedded_server(port: int = DEFAULT_PORT):
    """Run the engine's web server in a daemon thread (``listn tray`` owns it).

    Returns ``(server, thread)``; shutting the tray down also ends the server,
    whose lifespan finalizes any two-track recording with its honest report.
    """
    import uvicorn

    from listn.server import create_app

    app = create_app()
    app.state.http_port = port
    config = uvicorn.Config(app, host=LOOPBACK, port=port, log_level="warning",
                            timeout_graceful_shutdown=3)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="listn-tray-server",
                              daemon=True)
    thread.start()
    return server, thread
