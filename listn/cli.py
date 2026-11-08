"""Command-line entry point: ``listn <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from listn import __version__
from listn.config import BACKENDS, MODELS, Settings, SettingsError, data_dir, disable_network_telemetry
from listn.library import EXPORT_FORMATS, Library, export, fmt_clock


def _settings_from(args) -> Settings:
    s = Settings.load()
    for key in ("device", "model", "language", "backend"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(s, key, value)
    if getattr(args, "chunk", None) is not None:
        s.chunk_seconds = args.chunk
    if getattr(args, "no_audio", False):
        s.save_audio = False
    if getattr(args, "speakers", False):
        s.diarize_after = True
    if getattr(args, "hf_token", None):
        s.hf_token = args.hf_token
    return s.validated()


def _perf_epilog() -> str:
    """Show the transcription performance knobs as they are right now.

    Reads the effective ``Settings`` (settings.json + ``LISTN_*`` env), so
    ``--help`` reports what a run would actually use instead of a hard-coded
    default that may no longer be true.
    """
    try:
        s = Settings.load()
    except SettingsError as exc:
        return (f"performance knobs: INVALID — {exc}\n"
                "  Fix settings.json or the LISTN_COMPUTE_TYPE / "
                "LISTN_CPU_THREADS / LISTN_BEAM_SIZE environment variable.")
    threads = "auto (backend chooses)" if s.cpu_threads is None else str(s.cpu_threads)
    return ("performance knobs (effective for this run):\n"
            f"  compute type: {s.compute_type}\n"
            f"  cpu threads:  {threads}\n"
            f"  beam size:    {s.beam_size}\n"
            "  override with LISTN_COMPUTE_TYPE, LISTN_CPU_THREADS, LISTN_BEAM_SIZE "
            "(env wins over settings.json)")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=MODELS,
                   help="Whisper model (default: saved setting; first run is large-v3-turbo on Apple Silicon with MLX, otherwise small)")
    p.add_argument("--language", metavar="CODE", help="Language code such as en, es, de (default: auto-detect)")
    p.add_argument("--backend", choices=BACKENDS, help="Transcription backend (default: auto)")
    p.add_argument("--speakers", action="store_true", help="Label speakers with pyannote (needs the diarize extra)")
    p.add_argument("--hf-token", help="Hugging Face token for the first pyannote download (or set HF_TOKEN)")


def cmd_ui(args) -> int:
    from listn.server import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_tray(args) -> int:
    """KDE Plasma tray controls for the real two-track recorder.

    The tray is a client of the loopback server (same API the web UI uses).
    If no server is answering on --port, one is started here and the tray
    owns its lifetime; otherwise the tray attaches to the running server and
    only ever acts on it.
    """
    if sys.platform == "darwin":
        print("The KDE tray is Linux-only; on macOS use the listn menu bar app.",
              file=sys.stderr)
        return 2
    import asyncio

    try:
        import dbus_next  # noqa: F401
    except ImportError:
        print("The tray needs dbus-next. Install it with `uv sync --extra tray` "
              "or `pip install 'listn[tray]'`.", file=sys.stderr)
        return 3

    from listn.meeting_watch import MeetingWatcher
    from listn.tray import ServerClient, TrayController, port_in_use
    from listn.tray_sni import StatusNotifierItem

    port = args.port
    owned = None
    thread = None
    if port_in_use(port):
        print(f"listn tray: attached to the engine on 127.0.0.1:{port}")
    elif args.no_server:
        print(f"listn is not answering on 127.0.0.1:{port}. Start it with "
              "`listn ui`, or drop --no-server to let the tray start it.",
              file=sys.stderr)
        return 1
    else:
        from listn.tray import start_embedded_server

        owned, thread = start_embedded_server(port)
        end = time.monotonic() + 30
        while time.monotonic() < end and not port_in_use(port):
            time.sleep(0.2)
        if not port_in_use(port):
            print("The tray's embedded engine did not come up.", file=sys.stderr)
            return 1
        print(f"listn tray: engine started on 127.0.0.1:{port}")

    controller = TrayController(ServerClient(port=port),
                                meeting_watcher=MeetingWatcher())
    tray = StatusNotifierItem(controller)
    try:
        asyncio.run(_run_desktop(controller, tray, port))
    finally:
        # Quit must never leave a recording running behind us.
        if controller.status.active:
            try:
                controller.client.stop()
            except Exception:
                logging.getLogger("listn.tray").exception(
                    "stop on quit failed; check the engine at 127.0.0.1:%s", port)
        if owned is not None and thread is not None:
            owned.should_exit = True
            thread.join(timeout=10)
    return 0


async def _run_desktop(controller, tray, port: int) -> None:
    """The tray's D-Bus loop plus the two desktop extras.

    * global shortcuts (start/stop, bookmark) — registered with the
      desktop's own portal; whatever it grants, or the reason it granted
      nothing, is shown in the menu.
    * captions — an opt-in window over the local transcript events; it
      only subscribes while the setting is on.

    Neither is a hard dependency: no portal or no display leaves the tray
    fully working and says so on the menu instead of failing to start.
    """
    import asyncio

    from listn.captions import CaptionSupervisor
    from listn.shortcuts import (DEFAULT_BOOKMARK, DEFAULT_START_STOP,
                                 HotkeyActions, ShortcutManager)

    controller.refresh()  # settings first: chords and captions come from it
    settings = controller.settings or {}
    chords = {
        "start_stop": settings.get("hotkey_start_stop") or DEFAULT_START_STOP,
        "bookmark": settings.get("hotkey_bookmark") or DEFAULT_BOOKMARK,
    }
    shortcuts = ShortcutManager(HotkeyActions(controller), chords)
    controller.shortcut_status = shortcuts.status

    captions = CaptionSupervisor(port=port,
                                 get_settings=lambda: controller.settings)
    controller.captions = captions
    captions.start()

    loop = asyncio.get_running_loop()

    def rebind(new_chords: dict) -> None:
        # refresh() runs in the tray's executor thread: hand the coroutine
        # back to the loop that owns the D-Bus connection.
        async def apply():
            await shortcuts.rebind(new_chords)
            controller.shortcut_status = shortcuts.status

        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(apply()))

    controller.on_chords_changed = rebind
    try:
        await shortcuts.start()
        controller.shortcut_status = shortcuts.status
        await tray.run()
    finally:
        await shortcuts.close()
        captions.stop()


def cmd_devices(args) -> int:
    from listn import devices as devmod
    from listn.capture import DeviceError, measure_levels

    try:
        all_devices = devmod.list_all()
    except Exception as exc:
        print(f"Could not query audio devices: {exc}", file=sys.stderr)
        return 1
    output = devmod.default_output_name()
    try:
        auto = devmod.resolve(Settings.load().device, all_devices)
    except LookupError:
        auto = None
    print(f"{'IDX':>3}  {'NAME':<42} {'IN':>3}  {'RATE':>6}  KIND")
    for d in all_devices:
        if d.inputs <= 0:
            continue
        mark = "*" if auto and d.index == auto.index else " "
        print(f"{d.index:>3}{mark} {d.name[:42]:<42} {d.inputs:>3}  {int(d.default_samplerate):>6}  {d.kind}")
    print(f"\nSystem output: {output or 'unknown'}   (* = device listn will use)")
    for note in devmod.setup_guidance(all_devices, output):
        print(f"\n[{note['level'].upper()}] {note['title']}\n  {note['detail']}")
    if args.test is not None:
        target = args.test or None
        print(f"\nListening for 2 seconds on {target or (auto.name if auto else 'default')}...")
        try:
            report = measure_levels(target)
        except DeviceError as exc:
            print(exc, file=sys.stderr)
            return 1
        for ch in report["per_channel"]:
            print(f"  ch{ch['channel']}: rms {ch['rms_db']} dBFS, peak {ch['peak_db']} dBFS")
        print(f"  {report['advice']}")
    return 0


def _print_pipewire_targets(targets, stream) -> None:
    for name, target in targets.to_dict().items():
        print(f"{name}: {target['name']} ({target['media_class']}, id {target['id']}, "
              f"selection: {target['selection']})", file=stream)


def cmd_record(args) -> int:
    """Two-track capture through PipeWire: system output and microphone."""
    from listn import pipewire as pw

    try:
        graph = pw.discover()
    except pw.CaptureError as exc:
        print(f"PipeWire capture unavailable: {exc}", file=sys.stderr)
        return 3
    if args.list_targets:
        print("System outputs (Audio/Sink):")
        default = graph.default_sink or ""
        for node in graph.sinks:
            mark = "  [session default]" if node.name == default else ""
            desc = f" — {node.description}" if node.description else ""
            print(f"  {node.name}  (id {node.id}){desc}{mark}")
        print("Application output streams (Stream/Output/Audio):")
        apps = graph.app_streams
        if not apps:
            print("  (none)")
        for node in apps:
            # node.description already carries the application's own name
            # only — never a window or media title (listn/pipewire.py).
            desc = f" — {node.description}" if node.description else ""
            print(f"  {node.name}  (id {node.id}){desc}")
        print("Microphones (Audio/Source):")
        default = graph.default_source or ""
        for node in graph.sources:
            mark = "  [session default]" if node.name == default else ""
            desc = f" — {node.description}" if node.description else ""
            print(f"  {node.name}  (id {node.id}){desc}{mark}")
        print("\nPick both explicitly: listn record --system-target <node> --mic-target <node>")
        print("  --system-target takes a system output (all applications mixed) or one "
              "application stream (that app only).")
        return 0
    if args.backend != "pipewire":
        print(f"Unknown capture backend '{args.backend}'. Use pipewire.", file=sys.stderr)
        return 2
    chatter = sys.stderr if args.json else sys.stdout
    try:
        targets = pw.select_targets(graph, args.system_target, args.mic_target)
    except pw.CaptureError as exc:
        print(f"PipeWire targets unavailable: {exc}", file=sys.stderr)
        return 3
    _print_pipewire_targets(targets, chatter)
    if args.out:
        out_dir = Path(args.out)
    else:
        from listn.config import data_dir

        root = data_dir() / "pipewire"
        root.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S") + "-",
                                        dir=root))
    print(f"Recording two tracks into {out_dir}", file=chatter)
    recorder = pw.TwoTrackRecorder(out_dir, targets, rate=args.rate)
    try:
        recorder.start()
    except pw.CaptureError as exc:
        print(f"Capture unavailable: {exc}", file=sys.stderr)
        return 3
    duration = max(0.0, args.duration)
    try:
        end = time.monotonic() + duration
        while duration > 0 and time.monotonic() < end:
            time.sleep(min(0.2, max(0.0, end - time.monotonic())))
        if duration <= 0:
            print("Recording until Ctrl+C...", file=chatter)
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", file=chatter)
    report = recorder.stop()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    for name, track in report["tracks"].items():
        verdict = (track["verdict"] or "no signal data").upper()
        line = (f"{name}: {track['state']} {track['duration']:.2f}s -> "
                f"{track['file'] or 'no file'} [{verdict}")
        if track["peak_db"] is not None:
            line += f" peak {track['peak_db']} dBFS"
        line += "]"
        if track["error"]:
            line += f"  error: {track['error']}"
        print(line)
    if report["ok"]:
        print(f"Capture OK: both tracks recorded with signal in {report['out_dir']}.")
    else:
        reasons = [f"{n}: {t['error'] or t['verdict'] or t['state']}"
                   for n, t in report["tracks"].items()
                   if t["state"] != "stopped" or t["verdict"] == "silent"]
        print("Capture NOT OK — " + "; ".join(reasons), file=sys.stderr)
    return 0 if report["ok"] else 1


def cmd_live(args) -> int:
    from listn.engine import Engine

    settings = _settings_from(args)
    engine = Engine()
    q = engine.bus.subscribe()
    done = threading.Event()
    try:
        session = engine.start_live(settings, args.title)
    except Exception as exc:
        print(f"Could not start: {exc}", file=sys.stderr)
        return 1
    dev = session.device_info or {}
    print(f"Recording from {dev.get('name')} ({dev.get('channels')}ch @ {dev.get('samplerate')} Hz). "
          f"Model: {settings.model}. Press Ctrl+C to stop.\n")

    def on_sigint(signum, frame):
        if not done.is_set():
            done.set()
            print("\nStopping, finishing the last chunk...", file=sys.stderr)
            threading.Thread(target=engine.stop_live, kwargs={"wait": True}, daemon=True).start()

    signal.signal(signal.SIGINT, on_sigint)
    while True:
        try:
            event = q.get(timeout=0.5)
        except Exception:
            continue
        kind = event.get("type")
        if kind == "live.segments":
            for seg in event["segments"]:
                who = f"{seg['speaker']}: " if seg.get("speaker") else ""
                print(f"[{fmt_clock(seg['start'])}] {who}{seg['text']}", flush=True)
        elif kind == "live.status" and event.get("message") not in ("Listening",):
            print(f"  ({event['message']})", file=sys.stderr)
        elif kind == "live.error":
            print(f"  ! {event['message']}", file=sys.stderr)
            if session.state == "error":
                return 1
        elif kind == "live.stopped":
            break
    if session.meta.id in engine.jobs:
        _follow_speaker_job(engine, q, session.meta.id)
    meta = engine.library.get(session.meta.id)
    print(f"\nSaved session {meta.id} ({fmt_clock(meta.duration)}) in {engine.library.root / meta.id}")
    if args.output:
        Path(args.output).write_text(export(meta, engine.library.segments(meta.id), args.format))
        print(f"Wrote {args.output}")
    return 0


def cmd_transcribe(args) -> int:
    from listn.engine import Engine

    settings = _settings_from(args)
    engine = Engine()
    q = engine.bus.subscribe()
    status = 0
    for file in args.files:
        path = Path(file).expanduser()
        if not path.is_file():
            print(f"Not found: {path}", file=sys.stderr)
            status = 1
            continue
        job = engine.enqueue_file(path, args.title or path.stem, settings, settings.diarize_after)
        last = -1
        while True:
            event = q.get()
            if event.get("job", {}).get("id") != job.meta.id:
                continue
            if event["type"] == "job.update":
                pct = int(event["job"]["progress"] * 100)
                stage = event["job"]["stage"]
                if pct != last or stage not in ("transcribing",):
                    last = pct
                    print(f"\r{path.name}: {stage} {pct}%   ", end="", file=sys.stderr, flush=True)
            elif event["type"] == "job.done":
                print(file=sys.stderr)
                if event["error"]:
                    print(f"{path.name}: failed: {event['error']}", file=sys.stderr)
                    status = 1
                break
        meta = engine.library.get(job.meta.id)
        if meta.status != "done":
            continue
        text = export(meta, engine.library.segments(meta.id), args.format)
        if args.output:
            out = Path(args.output)
            if len(args.files) > 1 or out.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                out = out / f"{path.stem}.{args.format}"
            out.write_text(text)
            print(f"Wrote {out}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        print(f"Saved to library as {meta.id}", file=sys.stderr)
    return status


def _follow_speaker_job(engine, q, session_id: str) -> str | None:
    """Print speaker-labelling progress until the job ends. Returns its error, if any."""
    last = None
    while True:
        try:
            event = q.get(timeout=1.0)
        except Exception:
            if session_id not in engine.jobs:
                return None
            continue
        if event.get("job", {}).get("id") != session_id:
            continue
        if event["type"] == "job.update":
            job = event["job"]
            line = f"Speakers: {job.get('detail') or job['stage']} {int(job['progress'] * 100)}%"
            if line != last:
                last = line
                print(f"\r{line}   ", end="", file=sys.stderr, flush=True)
        elif event["type"] == "job.done":
            print(file=sys.stderr)
            return event["error"]


def cmd_diarize(args) -> int:
    from listn import diarize
    from listn.engine import Engine

    settings = _settings_from(args)
    if not diarize.installed():
        print(diarize.INSTALL_HINT, file=sys.stderr)
        return 3
    engine = Engine()
    q = engine.bus.subscribe()
    try:
        engine.enqueue_speakers(args.id, settings)
    except KeyError:
        print(f"No session '{args.id}'. Run `listn list`.", file=sys.stderr)
        return 1
    except (RuntimeError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1
    error = _follow_speaker_job(engine, q, args.id)
    meta = engine.library.get(args.id)
    if error:
        print(meta.speaker_note or error, file=sys.stderr)
        return 1 if meta.speaker_state == "error" else 3
    print(meta.speaker_note or "Speakers labelled.")
    for label, shown in (meta.speakers or {}).items():
        if label in ("Me", "Others"):
            continue
        found = (meta.speaker_matches or {}).get(label) or {}
        score = f"  (voice match {found['score']:.2f})" if found.get("name") and found.get("score") is not None else ""
        print(f"  {shown}{score}")
    return 0


def cmd_speakers(args) -> int:
    import json

    from listn import diarize
    from listn.speakers import SpeakerBook, voice_embedder

    settings = Settings.load()
    if args.check:
        report = diarize.readiness(settings.resolved_hf_token())
        if report["installed"]:
            try:
                diarize.telemetry_off()
                import pyannote.audio
                import torch

                report["pyannote"] = pyannote.audio.__version__
                report["torch"] = torch.__version__
                report["device"] = diarize.pick_device().type
            except Exception as exc:
                report["import_error"] = str(exc)
        report.pop("terms", None)
        print(json.dumps(report, indent=2))
        if not report["installed"]:
            return 3
        return 1 if "import_error" in report else 0
    book = SpeakerBook(embedder=voice_embedder(settings.resolved_hf_token(), allow_download=args.reenroll))
    if args.reenroll:
        if not book.embedder.neural:
            print("The voice model could not be loaded, so profiles stay on the spectrum fallback.", file=sys.stderr)
            return 3
        result = book.reenroll(Library())
        print(f"Re-learned: {', '.join(result['reenrolled']) or 'none'}")
        if result["needs_reenroll"]:
            print(f"Name these again in a meeting: {', '.join(result['needs_reenroll'])}")
    profiles = book.describe()
    print(f"Voice method: {book.embedder.label}")
    if not profiles:
        print("No known voices yet. Name a speaker in the library to add one.")
    for p in profiles:
        flag = "  needs re-enrolment" if p["needs_reenroll"] else ""
        print(f"  {p['name']:<30} {p['clips']} sample{'s' if p['clips'] != 1 else ''}{flag}")
    return 0


def cmd_list(args) -> int:
    lib = Library()
    if args.query:
        from listn.index import SearchIndex

        results = SearchIndex(lib).sessions_for(args.query)
    else:
        results = lib.search("")
    if not results:
        print("No sessions yet." if not args.query else f"Nothing matches '{args.query}'.")
        return 0
    for r in results:
        m = r["meta"]
        extra = f"  {r['hit_count']} matches" if args.query else ""
        print(f"{m['id']}  {m['created_at'][:16].replace('T', ' ')}  {fmt_clock(m['duration']):>8}  "
              f"{m['status']:<12} {m['title']}{extra}")
        for hit in r["hits"][:3]:
            print(f"    [{fmt_clock(hit['start'])}] {hit['text']}")
    return 0


def cmd_export(args) -> int:
    lib = Library()
    try:
        meta = lib.get(args.id)
    except KeyError:
        print(f"No session '{args.id}'. Run `listn list`.", file=sys.stderr)
        return 1
    text = export(meta, lib.segments(meta.id), args.format)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_path(args) -> int:
    print(data_dir())
    return 0


def cmd_summarize(args) -> int:
    from listn.summary import write_notes

    lib = Library()
    try:
        lib.get(args.id)
    except KeyError:
        print(f"No session '{args.id}'.", file=sys.stderr)
        return 1
    notes = write_notes(lib, args.id, Settings.load(), force=bool(getattr(args, "force", False)))
    if notes.get("edited") and not getattr(args, "force", False):
        print("(These notes were edited by hand; pass --force to regenerate.)", file=sys.stderr)
    print(notes["summary"])
    for point in notes.get("key_points") or []:
        print(f"Point: {point['text']}")
    for decision in notes["decisions"]:
        print(f"Decision: {decision}")
    for action in notes["actions"]:
        due = f" (due {action['due']})" if action.get("due") else ""
        print(f"Action ({action['owner']}): {action['text']}{due}")
    for question in notes.get("open_questions") or []:
        print(f"Open: {question['text']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="listn", description="Local-only audio transcription.")
    p.add_argument("--version", action="version", version=f"listn {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = p.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="Open the local web UI (default)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    ui.set_defaults(func=cmd_ui)

    dv = sub.add_parser("devices", help="List input devices and setup guidance")
    dv.add_argument("--test", nargs="?", const="", metavar="DEVICE", help="Measure input level for 2 seconds")
    dv.set_defaults(func=cmd_devices)

    lv = sub.add_parser("live", help="Record and transcribe live in the terminal")
    lv.add_argument("--device", help="Device name, part of a name, or index (default: automatic)")
    lv.add_argument("--title")
    lv.add_argument("--chunk", type=float, metavar="SEC", help="Seconds per transcription chunk (2-30)")
    lv.add_argument("--no-audio", action="store_true", help="Don't keep the WAV recording")
    lv.add_argument("--output", help="Also write the transcript to this file")
    lv.add_argument("--format", choices=EXPORT_FORMATS, default="txt")
    _add_model_args(lv)
    lv.set_defaults(func=cmd_live)

    rc = sub.add_parser(
        "record",
        help="Record system output and microphone as two separate tracks (Linux/PipeWire)")
    rc.add_argument("--backend", choices=["pipewire"], default="pipewire",
                    help="Capture backend (default: pipewire)")
    rc.add_argument("--system-target", metavar="NODE",
                    help="PipeWire node name, id, or application name for the system output "
                         "track: a system output (all applications) or one application's "
                         "output stream (see --list)")
    rc.add_argument("--mic-target", metavar="NODE",
                    help="PipeWire node name or id for the microphone track (see --list)")
    rc.add_argument("--duration", type=float, default=5.0, metavar="SEC",
                    help="Seconds to record; 0 runs until Ctrl+C (default: 5)")
    rc.add_argument("--out", metavar="DIR",
                    help="Directory for system.wav/mic.wav (default: a fresh folder in the data directory)")
    rc.add_argument("--rate", type=int, default=48000, metavar="HZ",
                    help="Sample rate (default: 48000)")
    rc.add_argument("--list", action="store_true", dest="list_targets",
                    help="List the discovered PipeWire nodes and exit")
    rc.add_argument("--json", action="store_true", help="Print the capture report as JSON")
    rc.set_defaults(func=cmd_record)

    ty = sub.add_parser(
        "tray",
        help="KDE Plasma tray: start/stop/pause the two-track recorder (Linux)")
    ty.add_argument("--port", type=int, default=8765,
                    help="Engine port to control (default: 8765)")
    ty.add_argument("--no-server", action="store_true",
                    help="Only attach to an engine that is already running")
    ty.set_defaults(func=cmd_tray)

    tr = sub.add_parser("transcribe", help="Transcribe audio files (wav, mp3, m4a, ...)",
                        epilog=_perf_epilog(),
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    tr.add_argument("files", nargs="+")
    tr.add_argument("--title")
    tr.add_argument("--output", help="File (or directory for several inputs) to write")
    tr.add_argument("--format", choices=EXPORT_FORMATS, default="txt")
    _add_model_args(tr)
    tr.set_defaults(func=cmd_transcribe)

    ls = sub.add_parser("list", help="List or search saved sessions")
    ls.add_argument("query", nargs="?")
    ls.set_defaults(func=cmd_list)

    ex = sub.add_parser("export", help="Export a saved session")
    ex.add_argument("id")
    ex.add_argument("--format", choices=EXPORT_FORMATS, default="txt")
    ex.add_argument("--output")
    ex.set_defaults(func=cmd_export)

    pa = sub.add_parser("path", help="Print the data directory")
    pa.set_defaults(func=cmd_path)

    sm = sub.add_parser("summarize", help="Write a local summary, decisions, and action items")
    sm.add_argument("id")
    sm.add_argument("--force", action="store_true", help="Regenerate even if the notes were edited by hand")
    sm.set_defaults(func=cmd_summarize)

    di = sub.add_parser("diarize", help="Label speakers on a saved session again (Others track only)")
    di.add_argument("id")
    di.add_argument("--hf-token", help="Hugging Face token for the one-time model download (or set HF_TOKEN)")
    di.set_defaults(func=cmd_diarize)

    sp = sub.add_parser("speakers", help="List known voices")
    sp.add_argument("--reenroll", action="store_true", help="Re-learn profiles with the current voice model")
    sp.add_argument("--check", action="store_true", help="Check the speaker model install without downloading")
    sp.set_defaults(func=cmd_speakers)
    return p


def main(argv: list[str] | None = None) -> int:
    disable_network_telemetry()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s")
    if not getattr(args, "func", None):
        args = parser.parse_args([*(sys.argv[1:] if argv is None else argv), "ui"])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except SettingsError as exc:
        # Only the settings validation (settings.json / LISTN_* knobs) is a
        # user-facing error, so it alone gets the one-line treatment. Any other
        # ValueError is a bug: let it propagate with a traceback instead of
        # dressing it up as a usage problem.
        print(f"listn: {exc}", file=sys.stderr)
        return 2
