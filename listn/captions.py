"""Local-only caption window for a running recording (Linux desktop).

Captions are the most sensitive surface listn has: the text on screen is a
live transcript. This module therefore has three hard rules, each of them
tested:

* **Local only.** The window is drawn by this process; the text comes from
  the engine's own event stream over the loopback API (``GET /api/events``)
  — the same events the web UI already renders. Nothing is sent anywhere,
  and the client subscribes *only while captions are switched on*, so a
  recording made with captions off never reaches this code path.
* **Never logged, never copied.** No caption text is passed to ``log``
  (only counts and states), and there is no clipboard call in this file.
* **Closed on stop, cleared on pause.** ``live.paused`` empties the window
  and hides it, ``live.stopped`` empties it and destroys it, and events from
  any session other than the one captions were opened for are dropped.

Three layers, so the interesting rules need neither Tk nor a socket:

* :class:`CaptionModel` — pure state machine: events in, lines + window
  action out.
* :class:`EventStreamClient` — blocking reader for the loopback SSE stream.
* :class:`CaptionWindow` / :class:`CaptionSupervisor` — the Tk window and
  the thread that owns it. Both are optional: without tkinter or a display
  the supervisor reports why and does nothing, instead of failing silently.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("listn.captions")

LOOPBACK = "127.0.0.1"
MAX_LINES = 8
PAUSE_GAP = "[paused]"
PUMP_MS = 100
SETTINGS_POLL_SECONDS = 1.5

# What a caption window should do after an event. The window implements
# these; the model only decides.
OPEN = "open"
UPDATE = "update"
HIDE = "hide"
SHOW = "show"
CLOSE = "close"


@dataclass
class CaptionModel:
    """Events -> visible lines + the window action they imply.

    ``session`` pins the window to one recording: segments from any other
    session (a stale event, a second recording started elsewhere) are
    dropped, so captions can only ever show the active session.
    """

    max_lines: int = MAX_LINES
    lines: list[str] = field(default_factory=list)
    session: str | None = None
    running: bool = False
    paused: bool = False

    def clear(self) -> None:
        self.lines = []

    def handle(self, event: dict) -> str | None:
        """Apply one event; return the window action, or None."""
        if not isinstance(event, dict):
            return None
        kind = event.get("type")
        if kind == "live.started":
            self.session = str(event.get("session") or "") or None
            self.running = True
            self.paused = False
            self.clear()
            return OPEN
        if not self.running or self.session is None:
            return None
        if event.get("session") not in (None, self.session):
            return None
        if kind == "live.segments":
            added = False
            for seg in event.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                text = (seg.get("text") or "").strip()
                # The pause marker is a transcript gap, not speech: it must
                # never be rendered as something somebody said.
                if not text or text == PAUSE_GAP:
                    continue
                who = (seg.get("speaker") or "").strip()
                line = f"{who}: {text}" if who else text
                self.lines.append(line)
                added = True
            if len(self.lines) > self.max_lines:
                del self.lines[:len(self.lines) - self.max_lines]
            return UPDATE if added else None
        if kind == "live.paused":
            self.paused = True
            self.clear()
            return HIDE
        if kind == "live.resumed":
            self.paused = False
            self.clear()
            return SHOW
        if kind == "live.stopped":
            self.session = None
            self.running = False
            self.paused = False
            self.clear()
            return CLOSE
        return None


class EventStreamClient:
    """Blocking reader for ``GET /api/events`` on loopback.

    ``events()`` reconnects until ``stop()``. Payloads are never logged:
    a dropped connection reports the URL and the error, never the last
    caption line that happened to be in flight.
    """

    def __init__(self, port: int, host: str = LOOPBACK, timeout: float = 10.0,
                 opener=None):
        self.url = f"http://{host}:{port}/api/events"
        self.timeout = timeout
        self._stop = threading.Event()
        self._opener = opener or urllib.request.urlopen

    def stop(self) -> None:
        self._stop.set()

    def events(self):
        """Yield events forever until :meth:`stop` (reconnecting on failure)."""
        while not self._stop.is_set():
            try:
                with self._opener(self.url, timeout=self.timeout) as resp:
                    for line in resp:
                        if self._stop.is_set():
                            return
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            event = json.loads(payload)
                        except ValueError:
                            continue
                        if isinstance(event, dict):
                            yield event
            except Exception as exc:  # reconnect; the cause is not user text
                log.debug("caption event stream reconnect (%s)", type(exc).__name__)
                if self._stop.wait(1.0):
                    return


class CaptionWindow:
    """The Tk window. Created lazily so headless hosts can import this."""

    def __init__(self, root, width: int = 680, height: int = 170):
        import tkinter as tk

        self._tk = tk
        self.root = root
        self.visible = False
        self.destroyed = False
        self.text = tk.Text(root, wrap="word", font=("sans-serif", 15),
                            padx=12, pady=10, relief="flat", bd=0,
                            bg="rgba(0,0,0,0)" if False else "#101418",
                            fg="#f2f4f8", insertbackground="#f2f4f8",
                            state="disabled", height=5)
        self.text.pack(fill="both", expand=True)
        root.title("listn captions")
        root.geometry(f"{width}x{height}")
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass  # a compositor that refuses topmost still shows the window
        root.withdraw()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """(usable, reason when not). tkinter or a display may be missing."""
        try:
            import tkinter  # noqa: F401
        except Exception as exc:
            return False, f"tkinter is not available in this Python ({type(exc).__name__})"
        import os

        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False, "no display is available to this process"
        return True, ""

    @classmethod
    def create(cls) -> "CaptionWindow":
        import tkinter as tk

        return cls(tk.Tk())

    def apply(self, action: str | None, lines: list[str]) -> None:
        if self.destroyed or action is None:
            return
        if action == CLOSE:
            self.close()
            return
        self._write(lines)
        if action == OPEN or action == SHOW:
            self.root.deiconify()
            self.root.lift()
            self.visible = True
        elif action == HIDE:
            self.root.withdraw()
            self.visible = False
        elif action == UPDATE and not self.visible:
            # A segment can arrive before the started event is processed
            # (or while the window is hidden): never pop it up unprompted.
            pass

    def _write(self, lines: list[str]) -> None:
        body = "\n".join(lines)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if body:
            self.text.insert("1.0", body)
        self.text.configure(state="disabled")

    def close(self) -> None:
        if self.destroyed:
            return
        self._write([])
        self.destroyed = True
        self.visible = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


class CaptionSupervisor:
    """Owns the window thread and the (opt-in) event subscription.

    ``get_settings`` is called on a slow tick and must return the settings
    dict; captions render only while ``captions_enabled`` is truthy. The
    event stream is connected while enabled and disconnected when it is
    switched off, so "captions off" means the text never leaves the server.
    """

    def __init__(self, port: int, get_settings, window_factory=None,
                 stream_factory=None, poll_seconds: float = SETTINGS_POLL_SECONDS,
                 live_getter=None, available=None):
        self.port = port
        self.get_settings = get_settings
        self.poll_seconds = poll_seconds
        self._window_factory = window_factory or CaptionWindow.create
        self._stream_factory = stream_factory or (lambda: EventStreamClient(port))
        self._live_getter = live_getter or self._fetch_live
        # Tests inject a fake display check; a real run asks tkinter/X11.
        self._available = available or CaptionWindow.available
        self.model = CaptionModel()
        self.disabled_reason: str | None = None
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader: EventStreamClient | None = None
        self._reader_thread: threading.Thread | None = None
        self._window = None
        self._enabled = False
        self._ticks = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="listn-captions",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._unsubscribe()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -- the Tk thread ----------------------------------------------------

    def _run(self) -> None:
        usable, reason = self._available()
        if not usable:
            # Honest, one line, and it names the fix; captions stay off.
            self.disabled_reason = reason
            log.info("captions unavailable: %s", reason)
            return
        try:
            window = self._window_factory()
        except Exception as exc:
            self.disabled_reason = f"the caption window could not open ({type(exc).__name__})"
            log.info("captions unavailable: %s", self.disabled_reason)
            return
        self._window = window
        self._loop(window)

    def _loop(self, window) -> None:
        import time

        last_poll = 0.0
        while not self._stop.is_set():
            if time.monotonic() - last_poll >= self.poll_seconds:
                last_poll = time.monotonic()
                self._tick_settings()
            self._drain(window)
            self._stop.wait(PUMP_MS / 1000.0)
        # Shutdown: close the window and drop every line it held.
        self.model.handle({"type": "live.stopped", "session": self.model.session})
        window.close()
        self._unsubscribe()

    def _tick_settings(self) -> None:
        try:
            settings = self.get_settings() or {}
        except Exception as exc:
            log.debug("settings poll failed (%s)", type(exc).__name__)
            return
        enabled = bool(settings.get("captions_enabled"))
        if enabled and not self._enabled:
            self._enabled = True
            self.disabled_reason = None
            self._snapshot()
            self._subscribe()
        elif not enabled and self._enabled:
            self._enabled = False
            self._unsubscribe()
            self.model.handle({"type": "live.stopped", "session": self.model.session})
            if self._window is not None:
                self._window.apply(CLOSE, [])

    def _drain(self, window) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return
            action = self.model.handle(event)
            if action is not None:
                window.apply(action, list(self.model.lines))

    # -- subscription -----------------------------------------------------

    def _subscribe(self) -> None:
        if self._reader is not None:
            return
        reader = self._stream_factory()
        self._reader = reader

        def pump():
            for event in reader.events():
                if self._stop.is_set():
                    break
                self._queue.put(event)

        self._reader_thread = threading.Thread(target=pump, name="listn-caption-events",
                                               daemon=True)
        self._reader_thread.start()

    def _unsubscribe(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.stop()
        thread, self._reader_thread = self._reader_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _fetch_live(self) -> dict:
        with urllib.request.urlopen(
                f"http://{LOOPBACK}:{self.port}/api/live", timeout=5.0) as resp:
            return json.loads(resp.read() or b"{}")

    def _snapshot(self) -> None:
        """Catch up with a recording that is already running when captions
        are switched on: same loopback API, one GET, no new event source."""
        try:
            payload = self._live_getter()
        except Exception as exc:
            log.debug("live snapshot failed (%s)", type(exc).__name__)
            return
        live = payload.get("live")
        if not isinstance(live, dict) or live.get("state") in (None, "done", "error"):
            return
        session = live.get("id")
        if not session:
            return
        self._queue.put({"type": "live.started", "session": session})
        segments = payload.get("segments") or []
        if segments:
            self._queue.put({"type": "live.segments", "session": session,
                             "segments": segments})
        if live.get("paused") or live.get("state") == "paused":
            self._queue.put({"type": "live.paused", "session": session})
