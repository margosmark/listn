"""Localhost-only web UI and JSON API.

The server binds to 127.0.0.1 only. Requests whose Host header isn't a
loopback name are rejected (blocks DNS-rebinding), and state-changing
requests must come from the UI's own origin (blocks cross-site requests from
other pages open in your browser).
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("listn")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from listn import __version__, autostart, diarize
from listn import devices as devmod
from listn.capture import DeviceError, measure_levels
from listn.config import BACKENDS, MODELS, Settings, SettingsError, data_dir
from listn.engine import Engine
from listn.library import AUDIO_EXTS, EXPORT_FORMATS, export
from listn.transcriber import available_backends, resolve_backend

WEB_DIR = Path(__file__).parent / "web"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
MIME = {"txt": "text/plain", "md": "text/markdown", "srt": "application/x-subrip",
        "vtt": "text/vtt", "json": "application/json"}


class SettingsIn(BaseModel):
    device: str | None = None
    model: str | None = None
    language: str | None = None
    backend: str | None = None
    chunk_seconds: float | None = None
    save_audio: bool | None = None
    diarize_after: bool | None = None
    hf_token: str | None = None
    llm_model: str | None = None
    summarize_after: bool | None = None
    offer_meetings: bool | None = None
    auto_record_meetings: bool | None = None
    launch_at_login: bool | None = None
    vocabulary: list[str] | None = None
    vad_chunks: bool | None = None
    match_speakers: bool | None = None
    speaker_threshold: float | None = None
    audio_retention_days: int | None = None
    min_free_gb: float | None = None
    battery_saver: bool | None = None
    # Linux desktop extras (listn tray). Inert on macOS, whose caption
    # window and hotkeys are the native menu bar app's own.
    captions_enabled: bool | None = None
    hotkey_start_stop: str | None = None
    hotkey_bookmark: str | None = None


class DeviceTestIn(BaseModel):
    device: str | None = None
    seconds: float = 2.0


class LiveStartIn(BaseModel):
    title: str | None = None
    capture: str | None = None          # None = input device, "pipewire" = two tracks
    system_target: str | None = None    # explicit PipeWire node (name or id)
    mic_target: str | None = None       # explicit PipeWire node (name or id)


class LiveReplaceIn(BaseModel):
    """An explicit device replacement request (L7) — user-initiated only."""
    track: str                          # "system" | "mic"
    node: str                           # PipeWire node name from the graph


class SessionPatch(BaseModel):
    title: str | None = None
    speakers: dict[str, str] | None = None
    attendees: list[str] | None = None


class RetranscribeIn(BaseModel):
    diarize: bool = False


class NotesIn(BaseModel):
    summary: str = ""
    decisions: list[str] = []
    actions: list[dict] = []
    key_points: list[str] | None = None


class PrivacyIn(BaseModel):
    pane: str = "microphone"


# Provenance for a bookmark. The tray and the global shortcut both write
# through the same route; the source is stored with the bookmark so the
# transcript shows where a marker came from.
BOOKMARK_SOURCES = {"api", "tray", "hotkey", "web"}


class BookmarkIn(BaseModel):
    note: str = ""
    source: str = "api"


class AskIn(BaseModel):
    question: str = ""


class SpeakerCorrectIn(BaseModel):
    session_id: str
    start: float
    name: str


class SpeakerNameIn(BaseModel):
    label: str
    name: str


def _host_ok(host_header: str) -> bool:
    host = host_header.strip().lower()
    if host.startswith("["):
        name = host.split("]")[0] + "]"
    else:
        name = host.split(":")[0]
    return name in LOOPBACK_HOSTS


def create_app(engine: Engine | None = None, sd=None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        from listn.ingest import IngestServer

        try:
            ingest = IngestServer(app.state.engine, port=app.state.http_port)
            ingest.start()
            app.state.ingest = ingest
        except OSError as exc:
            log.warning("Capture socket unavailable: %s", exc)
            app.state.ingest = None
        yield
        app.state.shutting_down.set()
        pwlive = getattr(app.state, "pw", None)
        if pwlive is not None and pwlive.active:
            # Finalize the two tracks first; its stop also ends the session.
            pwlive.stop(wait=True)
        if app.state.engine.live_active():
            app.state.engine.stop_live(wait=True)
        if app.state.ingest is not None:
            app.state.ingest.stop()

    app = FastAPI(title="listn", version=__version__, docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.engine = engine or Engine(sd=sd)
    app.state.sd = sd
    app.state.settings = Settings.load()
    app.state.shutting_down = threading.Event()
    app.state.http_port = None
    app.state.ingest = None
    app.state.pw = None   # listn.pipewire_live.PipeWireLive while a two-track run is active

    def eng() -> Engine:
        return app.state.engine

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not _host_ok(request.headers.get("host", "")):
            return PlainTextResponse("listn only answers on localhost.", status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin:
                origin_host = origin.split("://", 1)[-1]
                if not _host_ok(origin_host):
                    return PlainTextResponse("Cross-origin request blocked.", status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def _sd():
        if app.state.sd is not None:
            return app.state.sd
        import sounddevice

        return sounddevice

    def device_report() -> dict:
        try:
            sd = _sd()
            all_devices = devmod.list_all(sd)
            output = devmod.default_output_name(sd)
        except Exception as exc:
            return {"inputs": [], "default_output": None, "selected": None,
                    "guidance": [{"level": "error", "title": "Audio system unavailable", "detail": str(exc)}]}
        inputs = [d for d in all_devices if d.inputs > 0]
        selected = None
        try:
            selected = devmod.resolve(app.state.settings.device, all_devices).name
        except LookupError:
            pass
        return {"inputs": [d.to_dict() for d in inputs], "default_output": output, "selected": selected,
                "guidance": devmod.setup_guidance(all_devices, output)}

    @app.get("/api/status")
    def status():
        e = eng()
        return {
            "version": __version__,
            "platform": sys.platform,
            "settings": app.state.settings.public(),
            "models": MODELS,
            "backends": BACKENDS,
            "backend_available": available_backends(),
            "resolved_backend": resolve_backend(app.state.settings.backend),
            "diarization_available": diarize.installed(),
            "diarization": diarize.readiness(app.state.settings.resolved_hf_token()),
            "data_dir": str(data_dir()),
            "live": e.live.status() if e.live else None,
            "live_active": e.live_active(),
            "jobs": [j.to_dict() for j in e.jobs.values()],
            "capture_sock": str(app.state.ingest.path) if getattr(app.state, "ingest", None) and app.state.ingest.path else None,
        }

    @app.get("/api/dashboard")
    def dashboard():
        """Read-only snapshot of real engine state for the local dashboard."""
        from listn.dashboard import build_snapshot

        pwlive = getattr(app.state, "pw", None)
        pipewire = pwlive.status() if pwlive is not None else None
        return build_snapshot(eng(), pipewire)

    @app.get("/api/devices")
    def get_devices():
        return device_report()

    @app.get("/api/pipewire/devices")
    def get_pipewire_devices():
        """Discovered PipeWire nodes for explicit target picking. Never selects."""
        from listn.pipewire_live import discover_devices

        return discover_devices()

    @app.post("/api/devices/test")
    def test_device(body: DeviceTestIn):
        if eng().live_active():
            raise HTTPException(409, "Stop the current recording before testing a device.")
        try:
            return measure_levels(body.device or app.state.settings.device,
                                  seconds=min(5.0, max(0.5, body.seconds)), sd=_sd())
        except DeviceError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/settings")
    def get_settings():
        return app.state.settings.public()

    @app.put("/api/settings")
    def put_settings(body: SettingsIn):
        current = app.state.settings
        changes = body.model_dump(exclude_unset=True)
        if "model" in changes and changes["model"] not in MODELS:
            raise HTTPException(400, f"Unknown model. Choose one of {', '.join(MODELS)}.")
        if "backend" in changes and changes["backend"] not in BACKENDS:
            raise HTTPException(400, f"Unknown backend. Choose one of {', '.join(BACKENDS)}.")
        before = {key: getattr(current, key) for key in changes}
        if "launch_at_login" in changes:
            # On Linux the autostart entry *is* the switch: apply it first so
            # a refusal (foreign file, HOME unset, path outside $HOME) leaves
            # settings.json untouched instead of claiming it worked. macOS is
            # a no-op here — the native app owns that toggle.
            try:
                autostart.set_enabled(bool(changes["launch_at_login"]))
            except autostart.AutostartError as exc:
                raise HTTPException(400, str(exc)) from exc
        for key, value in changes.items():
            setattr(current, key, value)
        try:
            current.validated().save()
        except SettingsError as exc:
            # An unfixable value (a malformed shortcut, say) is a 400 the
            # user can act on, and the previous settings stay in effect.
            for key, value in before.items():
                setattr(current, key, value)
            raise HTTPException(400, str(exc)) from None
        return current.public()

    @app.post("/api/live/start")
    def live_start(body: LiveStartIn):
        if body.capture not in (None, "", "pipewire"):
            raise HTTPException(400, f"Unknown capture mode '{body.capture}'. "
                                     "Use 'pipewire' for two tracks, or leave it out for the input device.")
        wants_pipewire = body.capture == "pipewire" or bool(body.system_target or body.mic_target)
        if wants_pipewire:
            from listn.pipewire import CaptureError
            from listn.pipewire_live import start_live

            if not (body.system_target and body.mic_target):
                raise HTTPException(400, "Choose both targets explicitly: a system output "
                                         "and a microphone from GET /api/pipewire/devices. "
                                         "listn never guesses a capture target.")
            if eng().live_active():
                raise HTTPException(409, "A recording is already running.")
            try:
                capture = start_live(eng(), Settings(**{**app.state.settings.__dict__}),
                                     body.system_target, body.mic_target, body.title)
            except CaptureError as exc:
                raise HTTPException(400, str(exc))
            except RuntimeError as exc:
                raise HTTPException(409 if "already" in str(exc) else 400, str(exc))
            app.state.pw = capture
            if capture.session is None:  # start_live attaches before returning
                raise HTTPException(500, "The live session did not start.")
            return capture.session.status()
        try:
            session = eng().start_live(Settings(**{**app.state.settings.__dict__}), body.title)
        except RuntimeError as exc:
            raise HTTPException(409 if "already" in str(exc) else 400, str(exc))
        return session.status()

    @app.post("/api/privacy/open")
    def open_privacy(body: PrivacyIn):
        from listn.privacy import PANES, open_pane

        if body.pane not in PANES:
            raise HTTPException(400, f"Unknown pane. Choose one of {', '.join(PANES)}.")
        return {"opened": open_pane(body.pane), "url": PANES[body.pane]}

    @app.post("/api/live/pause")
    def live_pause():
        if not eng().pause_live():
            raise HTTPException(409, "Nothing is recording.")
        return eng().live.status()

    @app.post("/api/live/resume")
    def live_resume():
        if not eng().resume_live():
            raise HTTPException(409, "Nothing is paused.")
        return eng().live.status()

    @app.post("/api/live/replace")
    def live_replace(body: LiveReplaceIn):
        """Resume a dead track on a device the user chose (L7).

        The one path that may change the target of a running capture. It
        never happens on its own: every refusal (unknown node, wrong kind,
        the sibling's target, a second stream, an exhausted recovery
        budget, a paused session) comes back as 400/409 with the reason
        and the capture keeps its original targets.
        """
        from listn.pipewire import CaptureError
        pwlive = getattr(app.state, "pw", None)
        if pwlive is None or not pwlive.active:
            raise HTTPException(409, "Nothing is recording.")
        live = eng().live
        if live is not None and getattr(live, "paused", False):
            raise HTTPException(409, "Resume the recording before replacing "
                                     "a device.")
        try:
            return pwlive.replace_track(body.track, body.node)
        except CaptureError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/live/stop")
    def live_stop():
        pwlive = getattr(app.state, "pw", None)
        if pwlive is not None and pwlive.active:
            # Stops the recorder first (finalizing both WAVs with its own
            # honest report), then the session that consumed the blocks.
            result = pwlive.stop(wait=False)
            if result["id"] is None and result["pipewire"] is None:
                raise HTTPException(409, "Nothing is recording.")
            return result
        meta = eng().stop_live(wait=False)
        if meta is None:
            raise HTTPException(409, "Nothing is recording.")
        return {"id": meta.id}

    @app.get("/api/live")
    def live_status():
        e = eng()
        pwlive = getattr(app.state, "pw", None)
        pw_status = pwlive.status() if pwlive is not None else None
        if not e.live:
            return {"live": None, "pipewire": pw_status}
        return {"live": e.live.status(), "segments": [s.to_dict() for s in e.live.segments],
                "pipewire": pw_status}

    @app.post("/api/live/bookmark")
    def live_bookmark(body: BookmarkIn = BookmarkIn()):
        source = (body.source or "api").strip().lower()
        if source not in BOOKMARK_SOURCES:
            raise HTTPException(400, f"Unknown bookmark source. Choose one of "
                                     f"{', '.join(sorted(BOOKMARK_SOURCES))}.")
        item = eng().add_bookmark(body.note, source=source)
        if item is None:
            raise HTTPException(409, "Nothing is recording.")
        return item

    @app.post("/api/files")
    async def upload(file: UploadFile = File(...), title: str = Form(""), diarize_speakers: bool = Form(False)):
        name = Path(file.filename or "audio").name
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext or name}'. Use one of: {', '.join(sorted(AUDIO_EXTS))}.")
        tmp_dir = data_dir() / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
            tmp_path = Path(tmp.name)
        settings = Settings(**{**app.state.settings.__dict__})
        job = eng().enqueue_file(tmp_path, title.strip() or Path(name).stem, settings, diarize_speakers, move=True)
        eng().library.update(job.meta.id, source_name=name)
        return job.to_dict()

    @app.post("/api/jobs/{session_id}/cancel")
    def cancel_job(session_id: str):
        if not eng().cancel_job(session_id):
            raise HTTPException(404, "No running job for that session.")
        return {"ok": True}

    def _meta(session_id: str):
        try:
            return eng().library.get(session_id)
        except KeyError:
            raise HTTPException(404, "Session not found.")

    @app.get("/api/sessions")
    def sessions(q: str = ""):
        from listn.index import SearchIndex

        if q.strip():
            return {"results": SearchIndex(eng().library).sessions_for(q)}
        return {"results": eng().library.search(q)}

    @app.get("/api/search")
    def search(q: str = ""):
        from listn.index import SearchIndex

        index = SearchIndex(eng().library)
        return {"results": index.sessions_for(q), "hits": index.search(q)}

    def voice_book():
        """A SpeakerBook on WeSpeaker when its weights are cached, else the spectrum fallback."""
        from listn.speakers import SpeakerBook, voice_embedder

        return SpeakerBook(embedder=voice_embedder(allow_download=False))

    @app.get("/api/speakers")
    def speakers():
        from listn.speakers import loaded_embedder

        book = voice_book()
        settings = app.state.settings
        neural = book.embedder.neural
        return {"speakers": book.describe(), "method": book.embedder.key, "method_label": book.embedder.label,
                "neural": neural, "loaded": loaded_embedder() is not None,
                "threshold": settings.speaker_threshold if neural else settings.spectrum_threshold,
                "margin": book.embedder.margin}

    @app.post("/api/speakers/reenroll")
    def reenroll_speakers():
        book = voice_book()
        if not book.embedder.neural:
            raise HTTPException(400, "The voice model isn't on this machine yet. Label speakers in one session first; "
                                     "that downloads it once.")
        return book.reenroll(eng().library)

    @app.delete("/api/speakers/{name}")
    def delete_speaker(name: str):
        from listn.speakers import SpeakerBook

        if not SpeakerBook().delete(name):
            raise HTTPException(404, "No voice with that name.")
        return {"ok": True}

    @app.get("/api/speakers/export")
    def export_speakers():
        """The user's own voice prints as one JSON file, on request.

        Portability counterpart to delete: the numbers behind each profile, so
        the data can be taken out or backed up without handing over the audio.
        No absolute paths, no audio bytes, nothing leaves the machine — this is
        a download the user asked for.
        """
        book = voice_book()
        payload = {
            "listn_voice_export": 1,
            "method": book.embedder.key,
            "profiles": [
                {"name": p["name"], "vectors": p["vectors"], "clips": p["clips"],
                 "needs_reenroll": p["needs_reenroll"]}
                for p in book.profiles()
            ],
            "note": ("Voice prints only: the audio clips stay in the data directory. "
                     "Forgetting a voice in Settings removes the profile and its clips."),
        }
        return Response(json.dumps(payload, indent=2), media_type="application/json; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="listn-voices.json"'})

    @app.post("/api/speakers/correct")
    def correct_speaker(body: SpeakerCorrectIn):
        from listn.speakers import correct_segment

        name = " ".join(body.name.split())
        if not name:
            raise HTTPException(400, "Type the person's name.")
        try:
            return correct_segment(eng().library, voice_book(), body.session_id, body.start, name)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except KeyError:
            raise HTTPException(404, "Session not found.")

    @app.post("/api/speakers/enroll")
    async def enroll_speaker(name: str = Form(""), file: UploadFile = File(...)):
        label = " ".join(name.split())
        if not label:
            raise HTTPException(400, "Type the person's name.")
        raw = await file.read()
        if not raw or len(raw) > 20_000_000:
            raise HTTPException(400, "Choose a short WAV clip.")
        tmp = Path(tempfile.mkdtemp(prefix="lt")) / "clip.wav"
        try:
            tmp.write_bytes(raw)
            from listn.audio import read_wav

            audio = read_wav(tmp)
        except Exception as exc:
            raise HTTPException(400, f"Could not read that clip ({exc}). Use a WAV file.")
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        try:
            return voice_book().enroll(label, audio)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/ask")
    def ask_library(body: AskIn):
        from listn.ask import ask
        from listn.index import SearchIndex

        return ask(SearchIndex(eng().library), body.question, app.state.settings)

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        meta = _meta(session_id)
        lib = eng().library
        return {"meta": meta.to_dict(), "segments": [s.to_dict() for s in lib.segments(session_id)],
                "has_audio": lib.audio_path(meta) is not None, "notes": lib.notes(session_id),
                "bookmarks": lib.bookmarks(session_id)}

    @app.patch("/api/sessions/{session_id}")
    def patch_session(session_id: str, body: SessionPatch):
        _meta(session_id)
        changes = body.model_dump(exclude_none=True)
        if "title" in changes:
            changes["title"] = changes["title"].strip()[:200] or "Untitled"
        if "attendees" in changes:
            names = []
            for item in changes["attendees"]:
                text = str(item).strip()[:80]
                if text and text not in names:
                    names.append(text)
            changes["attendees"] = names[:40]
        return eng().library.update(session_id, **changes).to_dict()

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        e = eng()
        if e.live_active() and e.live.meta.id == session_id:
            raise HTTPException(409, "Stop the recording before deleting it.")
        if session_id in e.jobs:
            raise HTTPException(409, "Cancel the transcription before deleting it.")
        _meta(session_id)
        e.library.delete(session_id)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/summary")
    def summarize(session_id: str):
        _meta(session_id)
        from listn.summary import write_notes

        return write_notes(eng().library, session_id, app.state.settings)

    @app.put("/api/sessions/{session_id}/summary")
    def edit_summary(session_id: str, body: NotesIn):
        _meta(session_id)
        from listn.summary import edited_notes

        notes = edited_notes(body.summary, body.decisions, body.actions, eng().library.notes(session_id),
                             key_points=body.key_points)
        eng().library.save_notes(session_id, notes)
        return notes

    @app.post("/api/sessions/{session_id}/retranscribe")
    def retranscribe(session_id: str, body: RetranscribeIn):
        e = eng()
        meta = _meta(session_id)
        audio = e.library.audio_path(meta)
        if audio is None:
            raise HTTPException(400, "This session has no saved audio to transcribe.")
        if session_id in e.jobs or (e.live_active() and e.live.meta.id == session_id):
            raise HTTPException(409, "This session is busy.")
        settings = Settings(**{**app.state.settings.__dict__})
        return e.enqueue_file(audio, meta.title, settings, body.diarize, existing=meta).to_dict()

    @app.post("/api/sessions/{session_id}/diarize")
    def label_speakers(session_id: str):
        e = eng()
        _meta(session_id)
        if not diarize.installed():
            raise HTTPException(400, diarize.INSTALL_HINT)
        settings = Settings(**{**app.state.settings.__dict__})
        try:
            return e.enqueue_speakers(session_id, settings).to_dict()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/sessions/{session_id}/speakers/name")
    def name_session_speaker(session_id: str, body: SpeakerNameIn):
        from listn.speakers import name_speaker

        _meta(session_id)
        try:
            return name_speaker(eng().library, voice_book(), session_id, body.label, body.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str, format: str = "txt"):
        meta = _meta(session_id)
        if format not in EXPORT_FORMATS:
            raise HTTPException(400, f"Format must be one of {', '.join(EXPORT_FORMATS)}.")
        body = export(meta, eng().library.segments(session_id), format)
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in meta.title).strip() or meta.id
        return Response(body, media_type=f"{MIME[format]}; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{safe}.{format}"'})

    @app.get("/api/sessions/{session_id}/audio")
    def session_audio(session_id: str):
        meta = _meta(session_id)
        path = eng().library.audio_path(meta)
        if path is None:
            raise HTTPException(404, "No audio saved for this session.")
        return FileResponse(path)

    @app.post("/api/sessions/{session_id}/reveal")
    def reveal(session_id: str):
        meta = _meta(session_id)
        target = eng().library.audio_path(meta) or (eng().library.root / meta.id)
        if sys.platform != "darwin":
            return {"ok": False, "path": str(target)}
        subprocess.run(["open", "-R", str(target)], check=False)
        return {"ok": True, "path": str(target)}

    @app.get("/api/events")
    def events():
        bus = eng().bus
        q = bus.subscribe()

        stopping = app.state.shutting_down

        def stream():
            try:
                yield "retry: 2000\n\n"
                idle = 0.0
                while not stopping.is_set():
                    try:
                        event = q.get(timeout=0.5)
                    except queue.Empty:
                        idle += 0.5
                        if idle >= 15:
                            idle = 0.0
                            yield ": keepalive\n\n"
                        continue
                    idle = 0.0
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/dashboard")
    def dashboard_page():
        return FileResponse(WEB_DIR / "dashboard.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse({"detail": f"Unexpected error: {exc}"}, status_code=500)

    return app


class _Server:
    """uvicorn.Server whose shutdown ends open event streams first.

    The browser's EventSource (or any /api/events client) otherwise outlives
    the 3-second grace period, so uvicorn cancels it with an ASGI traceback
    after logging "Cancel N running task(s), timeout graceful shutdown
    exceeded". Signalled exits (Ctrl+C) and programmatic ones (the tray
    setting ``should_exit``) both go through ``shutdown()``.
    """

    def __new__(cls, app: FastAPI, port: int):
        import uvicorn

        class Server(uvicorn.Server):
            async def shutdown(self, sockets=None):
                app.state.shutting_down.set()
                await super().shutdown(sockets)

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                timeout_graceful_shutdown=3)
        return Server(config)


def serve(port: int = 8765, open_browser: bool = True, app: FastAPI | None = None) -> None:
    import webbrowser

    import socket

    url = f"http://127.0.0.1:{port}"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Port {port} is already in use. If listn is already running, open {url}; "
                  f"otherwise pick another port with `listn ui --port <n>`.")
            if open_browser:
                webbrowser.open(url)
            return
    app = app or create_app()
    app.state.http_port = port
    server = _Server(app, port)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"listn is running at {url}  (local only; Ctrl+C to quit)")
    print(f"Data directory: {data_dir()}")
    server.run()
