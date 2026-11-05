"""Run one backend's model load and inference in a separate process.

Why a separate process: CTranslate2's CUDA path can end a process with
SIGSEGV or SIGABRT straight from native code — no Python exception can
catch that. This module is the child: ``listn.transcriber`` spawns it with
``python -m listn.inference_worker <request-fd> <response-fd>`` and only
ever writes library files after the child has answered with a complete
result. A child that dies mid-load or mid-inference simply closes its
response pipe, and the parent turns that into a truthful error or a CPU
fallback — never a dead listn process, never a partial transcript.

Protocol (``multiprocessing.connection`` frames, length-prefixed):

* parent -> child: one JSON UTF-8 header, then exactly ``nbytes`` raw
  little-endian float32 audio bytes when the header says so.
* child -> parent: JSON messages. ``{"event": "status"|"progress", ...}``
  while working, then exactly one final ``{"ok": true, ...}`` or
  ``{"ok": false, "kind": "cuda"|"error", "error": "..."}``.

The child builds the backend through the same code the parent would run
(``_build_backend``), so device resolution, the CPU fallback on a broken
CUDA runtime and the download messaging are unchanged — they just happen
where a native crash is survivable. Tests inject a crashing backend via
the optional ``factory`` header field (``"module:attr"``); production never
sends it.
"""

from __future__ import annotations

import importlib
import json
import resource
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

# The child's model instance, built by the load request and reused for
# every transcribe request on this connection.
_instance: Any = None


def _send(conn: Connection, message: dict) -> None:
    conn.send_bytes(json.dumps(message).encode("utf-8"))


def _status(message: str) -> dict:
    return {"event": "status", "message": str(message)}


def _build(header: dict, notify) -> object:
    """Construct the backend exactly as the parent would, in this process."""
    knobs = dict(header.get("knobs") or {})
    backend = header.get("backend") or "faster-whisper"
    factory_path = header.get("factory")
    if factory_path:
        # Test seam: a registered backend factory from the test suite, so a
        # test can make THIS process die on purpose without any GPU present.
        module, _, attr = str(factory_path).partition(":")
        factory = getattr(importlib.import_module(module), attr)
        return factory(header.get("model"), download=notify, **knobs)
    from listn.transcriber import _build_backend

    if backend != "faster-whisper":
        knobs = {}  # only faster-whisper takes the performance knobs
    return _build_backend(backend, str(header.get("model") or ""), knobs,
                          download=notify)


def _op_load(header: dict, conn: Connection) -> None:
    global _instance

    def notify(message) -> None:
        _send(conn, _status(message))

    instance = _build(header, notify)
    _instance = instance
    _send(conn, {
        "ok": True,
        "device": getattr(instance, "device", None),
        "device_message": getattr(instance, "device_message", None),
    })


def _op_transcribe(header: dict, conn_in: Connection, conn_out: Connection) -> None:
    if _instance is None:
        _send(conn_out, {"ok": False, "kind": "error",
                         "error": "the worker received a transcription request before a model load"})
        return
    payload = conn_in.recv_bytes()
    if header.get("nbytes") and len(payload) != int(header["nbytes"]):
        _send(conn_out, {"ok": False, "kind": "error",
                         "error": f"expected {header['nbytes']} audio bytes, got {len(payload)}"})
        return
    # bytearray copy: frombuffer over bytes would be read-only, and some
    # native paths want a writable array.
    audio = (np.frombuffer(bytearray(payload), dtype=np.float32) if payload
             else np.zeros(0, dtype=np.float32))
    progress: Callable[[float], None] | None = None
    if header.get("progress"):
        def _progress(value: float) -> None:
            _send(conn_out, {"event": "progress", "value": float(value)})

        progress = _progress

    segments = _instance.transcribe(
        audio,
        language=header.get("language"),
        prompt=header.get("prompt"),
        vad=bool(header.get("vad", True)),
        offset=float(header.get("offset") or 0.0),
        hotwords=header.get("hotwords"),
        progress=progress,
    )
    rows = [[float(s.start), float(s.end), str(s.text), float(s.no_speech_prob)]
            for s in segments]
    _send(conn_out, {"ok": True, "segments": rows})


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # A crashing child must not litter core dumps next to the user's data.
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    conn_in = Connection(int(argv[0]))
    conn_out = Connection(int(argv[1]))
    while True:
        try:
            raw = conn_in.recv_bytes()
        except (EOFError, OSError):
            break  # parent went away: nothing left to do
        try:
            header = json.loads(raw)
        except ValueError:
            break
        op = header.get("op")
        if op == "quit":
            break
        try:
            if op == "load":
                _op_load(header, conn_out)
            elif op == "transcribe":
                _op_transcribe(header, conn_in, conn_out)
            else:
                _send(conn_out, {"ok": False, "kind": "error",
                                 "error": f"unknown worker op {op!r}"})
        except BrokenPipeError:
            break
        except BaseException as exc:  # noqa: BLE001 - report, never die silently
            from listn.transcriber import CudaUnavailable, _is_cuda_error

            # A CUDA-shaped refusal is the same signal the in-process
            # backend treats as a device failure, so the parent can apply
            # the explicit-vs-auto policy to it.
            kind = "cuda" if isinstance(exc, CudaUnavailable) or _is_cuda_error(exc) else "error"
            _send(conn_out, {"ok": False, "kind": kind,
                             "error": str(exc) or exc.__class__.__name__})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
