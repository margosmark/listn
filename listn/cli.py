"""Command-line entry point: ``listn <command>``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from listn import __version__
from listn.config import BACKENDS, MODELS, Settings, data_dir, disable_network_telemetry
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


def cmd_list(args) -> int:
    lib = Library()
    results = lib.search(args.query or "")
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
    notes = write_notes(lib, args.id, Settings.load())
    print(notes["summary"])
    for decision in notes["decisions"]:
        print(f"Decision: {decision}")
    for action in notes["actions"]:
        print(f"Action ({action['owner']}): {action['text']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="listn", description="Local-only system audio transcription for macOS.")
    p.add_argument("--version", action="version", version=f"listn {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = p.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="Open the local web UI (default)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    ui.set_defaults(func=cmd_ui)

    dv = sub.add_parser("devices", help="List input devices and check BlackHole setup")
    dv.add_argument("--test", nargs="?", const="", metavar="DEVICE", help="Measure input level for 2 seconds")
    dv.set_defaults(func=cmd_devices)

    lv = sub.add_parser("live", help="Record and transcribe live in the terminal")
    lv.add_argument("--device", help="Device name, part of a name, or index (default: BlackHole if present)")
    lv.add_argument("--title")
    lv.add_argument("--chunk", type=float, metavar="SEC", help="Seconds per transcription chunk (2-30)")
    lv.add_argument("--no-audio", action="store_true", help="Don't keep the WAV recording")
    lv.add_argument("--output", help="Also write the transcript to this file")
    lv.add_argument("--format", choices=EXPORT_FORMATS, default="txt")
    _add_model_args(lv)
    lv.set_defaults(func=cmd_live)

    tr = sub.add_parser("transcribe", help="Transcribe audio files (wav, mp3, m4a, ...)")
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
    sm.set_defaults(func=cmd_summarize)
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
