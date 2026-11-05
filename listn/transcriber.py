"""Local Whisper backends behind one small interface.

* ``faster-whisper`` (default): CTranslate2 int8 on CPU, built-in Silero VAD,
  streams segments so long files report progress. Roughly 4x faster than
  openai-whisper on CPU with a fraction of the memory, and no PyTorch.
* ``mlx``: Apple's MLX port, runs on the Apple Silicon GPU. Optional extra.
* ``openai-whisper``: the original reference implementation. Optional extra.

Models are downloaded once from Hugging Face (or OpenAI's CDN for
openai-whisper) and loaded from the local cache afterwards. A cached load
sets ``HF_HUB_OFFLINE`` first, so it does not contact the Hub. The
unauthenticated-request warning only happens on that first download, and
only when ``HF_TOKEN`` is unset.

Where the default backend runs is the ``compute_device`` setting: ``cpu``
(the safe default, never touch the GPU), ``auto`` (GPU when this machine's
CUDA runtime is really usable, CPU otherwise), or ``cuda`` (require the GPU
and fail with a readable message instead of falling back). See
:func:`resolve_device`.

Model load and inference for the faster-whisper backend run in a child
process (:class:`IsolatedTranscriber`, worker: ``listn.inference_worker``)
so a native crash — SIGSEGV/SIGABRT from the CUDA path — can never kill
the caller or leave a partial transcript behind. ``auto`` then falls back
to CPU truthfully; explicit ``cuda`` fails with an actionable message.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

from listn.audio import TARGET_RATE, rms
from listn.hub import load_local_or_download

log = logging.getLogger("listn")

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

SPEECH_RMS_GATE = 0.003

_ALWAYS_HALLUCINATED = {
    "thanks for watching", "thank you for watching", "thanks for watching and please subscribe",
    "please subscribe", "like and subscribe", "subscribe to my channel", "blank audio",
    "music", "applause", "silence", "inaudible",
}
_FILLERS = {"you", "thank you", "thanks", "bye", "bye bye", "goodbye", "okay", "oh", "uh", "um", "hmm", "mm", "ah", "so"}
_HALLUCINATED_FRAGMENTS = ("amara.org", "subtitles by", "transcribed by", "captions by")


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']|_", " ", text.lower()).split())


def is_hallucination(text: str, no_speech_prob: float = 0.0, quiet: bool = False) -> bool:
    """Whisper invents stock phrases on silence or noise. Drop those.

    Short real words ("OK", "No", "Yes") are kept; generic fillers are only
    dropped when the model itself thinks there's no speech or the audio was
    very quiet.
    """
    lower = text.lower()
    if any(frag in lower for frag in _HALLUCINATED_FRAGMENTS):
        return True
    norm = _normalize(text)
    if not norm:
        return True
    if norm in _ALWAYS_HALLUCINATED:
        return True
    if norm in _FILLERS and (no_speech_prob >= 0.4 or quiet):
        return True
    return False


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


def available_backends() -> dict[str, bool]:
    out = {}
    for name, mod in (("faster-whisper", "faster_whisper"), ("mlx", "mlx_whisper"), ("openai-whisper", "whisper")):
        try:
            __import__(mod)
            out[name] = True
        except Exception:
            out[name] = False
    return out


class CudaUnavailable(RuntimeError):
    """``compute_device=cuda`` was asked for and this machine cannot run it.

    Raised before any model is built, so the user gets one readable line
    instead of a CTranslate2 failure (or a native crash) half-way through
    loading.
    """


# Shared libraries CTranslate2 dlopens by soname when a model is built for
# CUDA. On a machine with an NVIDIA driver but without the CUDA runtime
# libraries the load fails with "Library libcublas.so.12 is not found or
# cannot be loaded" — the failure seen on the CachyOS laptop.
CUDA_RUNTIME_LIBS = ("libcublas.so.12",)

# Runs in a child process. Loading a CUDA library, or anything CTranslate2
# would do with it, can terminate a process without ever raising a Python
# exception; the child takes that risk and the parent only reads one line of
# JSON back, so a broken GPU can never take listn down with it. Even the
# CTranslate2 device-count API is isolated because it calls the CUDA driver.
_PROBE_SCRIPT = """
import ctypes, json, sys
result = {"missing": [], "devices": 0}
try:
    import ctranslate2
    result["devices"] = int(ctranslate2.get_cuda_device_count())
except Exception as exc:
    result["device_error"] = str(exc)
for soname in sys.argv[1:]:
    try:
        ctypes.CDLL(soname)
    except OSError as exc:
        result["missing"].append("%s: %s" % (soname, exc))
print(json.dumps(result))
"""


def _run_cuda_probe(sonames: tuple[str, ...], timeout: float) -> tuple[dict | None, str | None]:
    """Run the native CUDA checks out-of-process; ``None`` means unsafe/unavailable."""
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE_SCRIPT, *sonames],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"the CUDA library probe timed out after {timeout:g}s"
    if done.returncode != 0:
        how = (f"signal {-done.returncode}" if done.returncode < 0
               else f"exit code {done.returncode}")
        return None, f"the CUDA library probe terminated with {how}"
    try:
        return json.loads(done.stdout.strip().splitlines()[-1]), None
    except (IndexError, ValueError):
        return None, "the CUDA library probe returned no usable answer"


def cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can see; query in a child process.

    A native driver/runtime bug becomes a quiet zero rather than killing listn.
    The actionable reason is emitted by :func:`cuda_runtime_ready` when a GPU
    device was requested explicitly.
    """
    result, _reason = _run_cuda_probe((), timeout=30.0)
    return int(result.get("devices", 0)) if result else 0


def cuda_runtime_ready(sonames: tuple[str, ...] = CUDA_RUNTIME_LIBS,
                       timeout: float = 30.0) -> tuple[bool, str]:
    """Can the CUDA runtime libraries CTranslate2 needs actually load?

    Returns ``(ready, detail)``. The native loader and CTranslate2 device
    query run in a child process: either can terminate the process without a
    Python exception. A child that dies by signal, times out, or prints
    nothing means "not ready" — returned as a reason, never as a crash.
    """
    result, error = _run_cuda_probe(sonames, timeout)
    if error or result is None:
        return False, error or "the CUDA library probe returned no usable answer"
    missing = result.get("missing", [])
    if missing:
        return False, "; ".join(missing)
    if result.get("device_error"):
        return False, f"CTranslate2 could not query the CUDA driver ({result['device_error']})"
    return True, ""


def _is_cuda_error(exc: BaseException) -> bool:
    """True when an exception is CTranslate2 refusing (or failing) to use the GPU."""
    text = str(exc).lower()
    return any(word in text for word in ("cuda", "cublas", "cudnn", "nccl", "libcuda", "gpu"))


def resolve_device(requested: str) -> tuple[str, str | None]:
    """Turn the ``compute_device`` setting into a WhisperModel device.

    Returns ``(device, note)``. ``note`` explains a CPU fallback when there is
    one and is ``None`` otherwise. Raises :class:`CudaUnavailable` when CUDA
    was requested by name and this machine cannot run it — checked before any
    model is built, so the answer is a message rather than a failure inside
    CTranslate2.

    ``auto`` means GPU only when a CUDA device is visible *and* the runtime
    libraries load; otherwise it stays on the historical behaviour (``auto``,
    which CTranslate2 resolves to CPU) or pins ``cpu`` when a GPU is visible
    but unusable.
    """
    want = (requested or "auto").strip().lower() or "auto"
    if want == "cpu":
        return "cpu", None
    if sys.platform == "darwin":
        # macOS has no CUDA; "auto" has always meant CPU for faster-whisper here.
        if want == "cuda":
            raise CudaUnavailable(
                "compute_device=cuda was requested, but macOS has no CUDA. "
                "Use compute_device=cpu, or leave it on auto.")
        return "auto", None
    if cuda_device_count() <= 0:
        if want == "cuda":
            raise CudaUnavailable(
                "compute_device=cuda was requested, but CTranslate2 sees no CUDA "
                "device (no NVIDIA driver, or none exposed to this process). "
                "Run on CPU with --compute-device cpu, or leave it on auto.")
        return "auto", None
    ready, detail = cuda_runtime_ready()
    if not ready:
        reason = f"CUDA is visible but the CUDA runtime cannot be used ({detail})"
        if want == "cuda":
            raise CudaUnavailable(
                f"compute_device=cuda was requested, but {reason}. Install the "
                "CUDA 12 libraries (docs/GPU_LINUX.md), or run on CPU with "
                "--compute-device cpu.")
        return "cpu", f"{reason}; see docs/GPU_LINUX.md"
    return ("cuda" if want == "cuda" else "auto"), None


def _reported_device(model: object, asked: str) -> str:
    """The device CTranslate2 says the loaded model is actually running on."""
    reported = getattr(getattr(model, "model", None), "device", None)
    return str(reported) if reported else asked


class Transcriber:
    backend = "base"
    # Set by backends that know where they are running; reported verbatim.
    device_message: str | None = None

    def __init__(self, model: str):
        self.model_name = model

    def segments(self, audio: np.ndarray, language: str | None = None,
                 prompt: str | None = None, vad: bool = True, hotwords: str | None = None) -> Iterator[RawSegment]:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: str | None = None,
                   vad: bool = True, offset: float = 0.0, hotwords: str | None = None,
                   progress: Callable[[float], None] | None = None) -> list[RawSegment]:
        """Transcribe and filter. Segment times are shifted by ``offset``."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        quiet = rms(audio) < SPEECH_RMS_GATE * 2
        duration = audio.size / TARGET_RATE
        out = []
        for seg in self.segments(audio, language, prompt, vad, hotwords):
            text = seg.text.strip()
            if progress and duration > 0:
                progress(min(1.0, seg.end / duration))
            if not text or is_hallucination(text, seg.no_speech_prob, quiet):
                continue
            out.append(RawSegment(seg.start + offset, seg.end + offset, text, seg.no_speech_prob))
        if progress:
            progress(1.0)
        return out


class FasterWhisper(Transcriber):
    backend = "faster-whisper"
    # What CTranslate2 reports after the model loads, not what we asked for.
    device = "cpu"

    def __init__(self, model: str, download: Callable[[str], None] | None = None, *,
                 compute_type: str = "int8", cpu_threads: int | None = None,
                 beam_size: int = 5, compute_device: str = "auto"):
        super().__init__(model)
        from faster_whisper import WhisperModel

        # Defaults stay byte-identical to the old hard-coded call: int8, no
        # cpu_threads passed (CTranslate2 picks its own count), and "auto" on
        # any machine without a usable CUDA runtime.
        self.beam_size = beam_size
        device, note = resolve_device(compute_device)
        notes = [note] if note else []
        kwargs: dict = dict(device=device, compute_type=compute_type)
        if cpu_threads:
            kwargs["cpu_threads"] = int(cpu_threads)
        want_cuda = (compute_device or "auto").strip().lower() == "cuda"
        failures: list[Exception] = []

        def load(dev: str):
            """Build the model on ``dev``.

            A device failure on the cached attempt is recorded in ``failures``
            so the download path never runs: a broken GPU is not a missing
            model, and retrying it over the network would both waste a round
            trip and claim a download that is not happening.
            """
            kw: dict = dict(kwargs, device=dev)
            failures.clear()

            def cached():
                try:
                    return WhisperModel(model, local_files_only=True, **kw)
                except Exception as exc:
                    if _is_cuda_error(exc):
                        failures.append(exc)
                    raise

            def fetched():
                if failures:
                    raise failures[0]
                return WhisperModel(model, **kw)

            def notify():
                if download and not failures:
                    download(f"Downloading Whisper '{model}' (one time)...")

            return load_local_or_download(cached, fetched,
                                          on_download=notify if download else None)

        try:
            self._model = load(device)
        except Exception as exc:
            if not failures and not _is_cuda_error(exc):
                raise  # genuinely not a device problem: bad files, no memory, ...
            if want_cuda:
                raise CudaUnavailable(
                    "compute_device=cuda was requested, but the model could not "
                    f"be built on the GPU: {exc}. Install the CUDA 12 libraries "
                    "(docs/GPU_LINUX.md), or run on CPU with "
                    "--compute-device cpu.") from exc
            # auto: a broken GPU must not stop transcription.
            notes.append(f"CUDA unusable ({exc}); running on CPU")
            self._model = load("cpu")

        self.device = _reported_device(self._model, device)
        # Truthful, user-facing: what the model reports, plus any fallback
        # reason. get_transcriber passes this to the status callback on every
        # call, cache hit or not.
        self.device_message = f"device: {self.device}"
        if notes:
            self.device_message += " — " + "; ".join(notes)

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        kwargs = dict(
            language=language, initial_prompt=prompt, vad_filter=vad,
            beam_size=self.beam_size, condition_on_previous_text=False,
        )
        if hotwords:
            kwargs["hotwords"] = hotwords
        try:
            segs, _info = self._model.transcribe(audio, **kwargs)
        except TypeError:
            kwargs.pop("hotwords", None)
            segs, _info = self._model.transcribe(audio, **kwargs)
        for s in segs:
            yield RawSegment(s.start, s.end, s.text, getattr(s, "no_speech_prob", 0.0))


class _WindowedBackend(Transcriber):
    """Backends that don't stream: feed long audio in windows for progress."""

    window_seconds = 120

    def _run(self, audio, language, prompt) -> list[dict]:
        raise NotImplementedError

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        step = self.window_seconds * TARGET_RATE
        carry = prompt
        for start in range(0, audio.size, step):
            piece = audio[start:start + step]
            if rms(piece) < SPEECH_RMS_GATE / 2:
                continue
            base = start / TARGET_RATE
            for d in self._run(piece, language, carry):
                yield RawSegment(base + float(d["start"]), base + float(d["end"]), str(d["text"]),
                                 float(d.get("no_speech_prob", 0.0)))
                carry = str(d["text"])[-200:]


class MlxWhisper(_WindowedBackend):
    backend = "mlx"

    def __init__(self, model: str, download: Callable[[str], None] | None = None):
        super().__init__(model)
        import mlx_whisper
        from huggingface_hub import snapshot_download

        repo = MLX_REPOS[model]

        def cached():
            return snapshot_download(repo, local_files_only=True)

        def fetched():
            return snapshot_download(repo)

        self._path = load_local_or_download(
            cached,
            fetched,
            on_download=(lambda: download(f"Downloading Whisper '{model}' for MLX (one time)...")) if download else None,
        )
        self._mlx = mlx_whisper

    def _run(self, audio, language, prompt):
        result = self._mlx.transcribe(audio, path_or_hf_repo=self._path, language=language,
                                      initial_prompt=prompt, verbose=None,
                                      condition_on_previous_text=False)
        return result.get("segments", [])


class OpenAIWhisper(_WindowedBackend):
    backend = "openai-whisper"

    def __init__(self, model: str, download: Callable[[str], None] | None = None):
        super().__init__(model)
        import whisper

        if download:
            download(f"Loading openai-whisper '{model}' (downloads once if missing)...")
        self._model = whisper.load_model(model)

    def _run(self, audio, language, prompt):
        result = self._model.transcribe(audio, language=language, initial_prompt=prompt,
                                        fp16=False, verbose=None, condition_on_previous_text=False)
        return result.get("segments", [])


_BACKENDS = {"faster-whisper": FasterWhisper, "mlx": MlxWhisper, "openai-whisper": OpenAIWhisper}
# Keyed by (backend, model, compute_type, cpu_threads, beam_size,
# compute_device, isolate) so a knob change reloads the model instead of
# reusing one built with the old settings.
_cache: dict[tuple, Transcriber] = {}
_cache_lock = threading.Lock()


def _build_backend(name: str, model: str, knobs: dict,
                   download: Callable[[str], None] | None = None) -> Transcriber:
    """Build backend ``name`` in THIS process.

    Shared by the in-process path and by the isolation worker child, so
    both construct models through exactly the same code — device
    resolution, the CPU fallback on a broken CUDA runtime and the download
    messaging all behave identically wherever they run. Whether that code
    is isolated in a child process is the caller's decision; this function
    never spawns anything.
    """
    try:
        return _BACKENDS[name](model, download=download, **knobs)
    except ImportError as exc:
        extra = {"mlx": "mlx", "openai-whisper": "openai"}.get(name, "")
        hint = f" Install it with: pip install 'listn[{extra}]'" if extra else ""
        raise RuntimeError(f"The {name} backend is not installed.{hint}") from exc


# --- Process isolation for native inference ---------------------------------
# CTranslate2's CUDA path can end the process with SIGSEGV/SIGABRT from
# native code; no Python exception handler in this process can catch that.
# IsolatedTranscriber therefore runs the COMPLETE model load + inference
# operation in a child process (listn.inference_worker) and only hands a
# result back once it is complete. Callers — engine jobs, live sessions —
# write to the library only after that answer arrives, so a native crash
# can never kill listn or leave a partial/duplicate transcript behind.

# Test seam: "module:attr" factory the worker builds instead of the real
# backend, so tests can kill a real child process on purpose. Production
# leaves it None and the worker uses the registered backend.
_FACTORY_PATH: str | None = None


class _Retry:
    """Policy answer: restart the worker pinned to CPU."""


_RETRY = _Retry()


class _ChildDied(RuntimeError):
    """The worker closed its response pipe without answering."""

    def __init__(self, detail: str, phase: str | None = None):
        super().__init__(detail)
        self.detail = detail
        self.phase = phase


class _CudaFailure(RuntimeError):
    """The worker answered ``kind="cuda"``: CTranslate2 refused the GPU."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class _BackendError(RuntimeError):
    """The worker answered ``kind="error"``: a normal Python failure.

    The worker itself is healthy, so this does not restart it.
    """


def _drain_stderr(stream, lines: deque) -> None:
    """Keep the worker's stderr flowing (so it can never block) and tail it."""
    try:
        for raw in iter(stream.readline, b""):
            lines.append(raw.decode("utf-8", "replace").rstrip())
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


class _WorkerSession:
    """One worker child process and the two pipes that talk to it."""

    def __init__(self, proc, to_child: Connection, from_child: Connection,
                 stderr_lines: deque):
        self.proc = proc
        self._to = to_child
        self._from = from_child
        self._stderr = stderr_lines

    @classmethod
    def spawn(cls) -> "_WorkerSession":
        req_r, req_w = os.pipe()   # parent writes requests, child reads
        res_r, res_w = os.pipe()   # child writes responses, parent reads
        env = dict(os.environ)
        # Make `python -m listn.inference_worker` resolvable no matter the cwd.
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "listn.inference_worker",
                 str(req_r), str(res_w)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, pass_fds=(req_r, res_w),
                env=env, close_fds=True)
        except BaseException:
            for fd in (req_r, req_w, res_r, res_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        os.close(req_r)
        os.close(res_w)
        lines: deque[str] = deque(maxlen=80)
        threading.Thread(target=_drain_stderr, args=(proc.stderr, lines),
                         daemon=True).start()
        return cls(proc, Connection(req_w), Connection(res_r), lines)

    def death_detail(self) -> str:
        """How the worker died, for a message a user can act on."""
        proc = self.proc
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return "stopped responding"
        if code is None:
            return "stopped responding"
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
            return f"was killed by {name}"
        if code > 0:
            tail = self._stderr[-1] if self._stderr else ""
            detail = f"exited with code {code}"
            return f"{detail}: {tail}" if tail else detail
        return "exited without answering"

    def request(self, header: dict, payload=None, *, on_status=None,
                on_progress=None) -> dict:
        """Send one request and read events until its final answer.

        A worker that died mid-request surfaces as :class:`_ChildDied` —
        never as a partial result, because the final answer is the only
        thing this method returns.
        """
        try:
            self._to.send_bytes(json.dumps(header).encode("utf-8"))
            if payload is not None:
                view = payload if isinstance(payload, (bytes, bytearray)) \
                    else memoryview(payload).cast("B")
                self._to.send_bytes(view)
        except (OSError, ValueError) as exc:
            raise _ChildDied(self.death_detail()) from exc
        while True:
            try:
                raw = self._from.recv_bytes()
            except (EOFError, OSError) as exc:
                raise _ChildDied(self.death_detail()) from exc
            try:
                message = json.loads(raw)
            except ValueError:
                raise _ChildDied("returned an unreadable answer") from None
            if "ok" in message:
                return message
            event = message.get("event")
            if event == "status" and on_status:
                on_status(message.get("message"))
            elif event == "progress" and on_progress:
                on_progress(float(message.get("value") or 0.0))

    def close(self) -> None:
        for conn in (self._to, self._from):
            try:
                conn.close()
            except Exception:
                pass
        proc = self.proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass


class IsolatedTranscriber(Transcriber):
    """A backend whose model load and inference run in a child process.

    The child takes every native crash. If it dies by signal mid-load or
    mid-inference, the parent sees a closed pipe — not a dead listn — and
    applies the compute-device policy:

    * ``auto`` — a crash that happened (or could have happened) on the
      GPU restarts the worker pinned to CPU, exactly once, and says so on
      the status line: a truthful fallback, never a silent one. A crash
      that was already on CPU, or a second crash after the fallback, is
      an error.
    * ``cuda`` — any crash or CUDA refusal raises :class:`CudaUnavailable`
      with the CPU advice; the worker is never restarted on CPU.
    * ``cpu`` — a crash is an error: there is no other device to fall
      back to.

    Only complete answers cross the process boundary, so callers write to
    the library exactly when (and because) inference finished.
    """

    backend = "faster-whisper"
    device = "cpu"

    def __init__(self, model: str, download: Callable[[str], None] | None = None, *,
                 compute_type: str = "int8", cpu_threads: int | None = None,
                 beam_size: int = 5, compute_device: str = "auto"):
        super().__init__(model)
        self._notify = download
        requested = (compute_device or "auto").strip().lower() or "auto"
        self._requested_device = requested
        self._effective_device = requested
        self._knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads,
                           beam_size=beam_size, compute_device=requested)
        self._session: _WorkerSession | None = None
        self._loaded = False
        self._device: str | None = None
        self._notes: list[str] = []
        self._fatal: tuple[type[BaseException], str] | None = None
        self.device_message = "device: unknown"
        self._lock = threading.Lock()
        # The first load happens here — where the in-process backend would
        # have built its model — so callers see identical timing.
        self._run("loading the model", self._worker)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Stop the worker process (its model and all) and release it."""
        with self._lock:
            self._kill_worker()

    def __del__(self):
        try:
            self._kill_worker()
        except Exception:
            pass

    def _kill_worker(self) -> None:
        session, self._session = self._session, None
        self._loaded = False
        if session is not None:
            session.close()

    def _refresh_message(self) -> None:
        self.device_message = f"device: {self._device or 'unknown'}"
        if self._notes:
            self.device_message += " — " + "; ".join(self._notes)

    def _worker(self) -> _WorkerSession:
        """The worker process, spawned and with its model loaded."""
        if self._session is None:
            self._session = _WorkerSession.spawn()
            self._loaded = False
        if not self._loaded:
            header = {"op": "load", "backend": self.backend,
                      "model": self.model_name,
                      "knobs": dict(self._knobs,
                                    compute_device=self._effective_device),
                      "factory": _FACTORY_PATH}
            try:
                response = self._session.request(header, on_status=self._notify)
            except _ChildDied as death:
                raise _ChildDied(death.detail,
                                 phase="loading the model") from None
            if not response.get("ok"):
                detail = str(response.get("error") or "the model could not be loaded")
                if response.get("kind") == "cuda":
                    raise _CudaFailure(detail)
                raise _BackendError(detail)
            self._loaded = True
            reported = response.get("device")
            if reported:
                self.device = str(reported)
                self._device = str(reported)
            if self._notes:
                # Keep the fallback note visible beside what the fresh
                # worker reports; the note is why it is running on CPU.
                self._refresh_message()
            elif response.get("device_message"):
                self.device_message = str(response["device_message"])
            else:
                self._refresh_message()
        return self._session

    # -- fallback policy --------------------------------------------------

    def _run(self, phase: str, fn):
        """Run ``fn``; turn a worker death into an error or a CPU fallback."""
        if self._fatal:
            cls, message = self._fatal
            raise cls(message)
        retried = False
        while True:
            try:
                return fn()
            except _ChildDied as death:
                answer = self._on_death(death.detail, death.phase or phase, retried)
                if isinstance(answer, _Retry):
                    retried = True
                    continue
                raise answer
            except _CudaFailure as failure:
                answer = self._on_cuda_failure(failure.detail, phase, retried)
                if isinstance(answer, _Retry):
                    retried = True
                    continue
                raise answer
            except _BackendError:
                raise  # a normal Python error: the worker itself is fine
            except BaseException:
                # Cancelled / KeyboardInterrupt: stop native work at once.
                self._kill_worker()
                raise

    def _set_fatal(self, cls: type, message: str) -> None:
        # Store type+text, not the exception: it gets re-raised on every
        # later call and must not accumulate tracebacks.
        self._fatal = (cls, message)

    def _pin_cpu(self, detail: str, phase: str) -> None:
        """Truthful ``auto`` fallback: restart on CPU and say exactly why."""
        self._effective_device = "cpu"
        self._knobs["compute_device"] = "cpu"
        self._device = "cpu"
        self.device = "cpu"
        self._notes.append(
            f"the transcription process {detail} while {phase}; running on CPU")
        self._refresh_message()
        self._kill_worker()
        if self._notify:
            self._notify(str(self.device_message))

    def _on_death(self, detail: str, phase: str, retried: bool) -> BaseException | _Retry:
        want = self._requested_device
        gpu_suspect = self._device != "cpu"  # None = died before reporting
        if want == "cuda":
            message = (f"compute_device=cuda was requested, but the transcription "
                       f"process {detail} while {phase}. Install the CUDA 12 "
                       "libraries (docs/GPU_LINUX.md), or run on CPU with "
                       "--compute-device cpu.")
            self._set_fatal(CudaUnavailable, message)
            self._kill_worker()
            return CudaUnavailable(message)
        if want == "auto" and gpu_suspect and not retried:
            self._pin_cpu(detail, phase)
            return _RETRY
        if retried:
            message = (f"The transcription process {detail} while {phase} even "
                       "after falling back to CPU; the model cannot run on this "
                       "machine. See docs/GPU_LINUX.md.")
        else:
            message = (f"The transcription process {detail} while {phase} "
                       f"(listn was running on "
                       f"{self._device or self._effective_device}, so there is "
                       "no other device to fall back to). See docs/GPU_LINUX.md.")
        self._set_fatal(RuntimeError, message)
        self._kill_worker()
        return RuntimeError(message)

    def _on_cuda_failure(self, detail: str, phase: str, retried: bool) -> BaseException | _Retry:
        want = self._requested_device
        if want == "auto" and not retried:
            self._pin_cpu(f"failed ({detail})", phase)
            return _RETRY
        if want == "cuda":
            message = detail if "--compute-device cpu" in detail else (
                f"{detail} (compute_device=cuda was requested). Install the "
                "CUDA 12 libraries (docs/GPU_LINUX.md), or run on CPU with "
                "--compute-device cpu.")
        else:
            message = f"{detail} (compute device: {self._device or self._effective_device})"
        self._set_fatal(CudaUnavailable, message)
        self._kill_worker()
        return CudaUnavailable(message)

    # -- inference --------------------------------------------------------

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None, vad: bool = True,
                   offset: float = 0.0, hotwords: str | None = None,
                   progress: Callable[[float], None] | None = None) -> list[RawSegment]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        audio = np.ascontiguousarray(audio)
        with self._lock:
            self._run("loading the model", self._worker)
            return self._run(
                "transcribing",
                lambda: self._send_transcribe(audio, language, prompt, vad,
                                              offset, hotwords, progress))

    def _send_transcribe(self, audio, language, prompt, vad, offset, hotwords,
                         progress) -> list[RawSegment]:
        session = self._worker()
        header = {"op": "transcribe", "language": language, "prompt": prompt,
                  "vad": bool(vad), "offset": float(offset or 0.0),
                  "hotwords": hotwords, "progress": progress is not None,
                  "nbytes": int(audio.nbytes)}
        response = session.request(header, audio, on_status=self._notify,
                                   on_progress=progress)
        if not response.get("ok"):
            detail = str(response.get("error") or "transcription failed")
            if response.get("kind") == "cuda":
                raise _CudaFailure(detail)
            raise _BackendError(detail)
        return [RawSegment(float(row[0]), float(row[1]), str(row[2]),
                           float(row[3]))
                for row in response.get("segments") or []]

    def segments(self, audio, language=None, prompt=None, vad=True, hotwords=None):
        # The worker runs the full transcribe (which applies the offset and
        # the hallucination filter); expose that result as an iterator.
        return iter(self.transcribe(audio, language=language, prompt=prompt,
                                    vad=vad, hotwords=hotwords))


def _close_cached(keep: tuple | None = None) -> None:
    """Stop the workers of cached models that are about to be evicted."""
    for key, instance in list(_cache.items()):
        if key == keep:
            continue
        close = getattr(instance, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            log.debug("closing a cached transcriber worker failed", exc_info=True)


# Children must not outlive the parent process; pipes would end their lives
# anyway at exit, but terminate them promptly instead of at process teardown.
atexit.register(_close_cached)


def get_transcriber(model: str, backend: str = "auto",
                    status: Callable[[str], None] | None = None,
                    *, compute_type: str | None = None, cpu_threads: int | None = None,
                    beam_size: int | None = None,
                    compute_device: str | None = None,
                    isolate: bool | None = None) -> Transcriber:
    """Build (and cache) a backend.

    The performance knobs and the compute device default to the effective
    ``Settings`` (settings.json plus ``LISTN_*`` env overrides), so every
    caller — live capture, file jobs, the ingest socket — picks them up
    without being changed. Pass them explicitly to override. Only the
    faster-whisper backend uses them; ``status`` receives a truthful
    ``device: …`` line once the model reports where it is actually running.

    ``isolate`` says where the work runs. ``None`` (the default) isolates
    the faster-whisper backend — the one whose CUDA path can die on a
    native signal — in a child process (:class:`IsolatedTranscriber`), and
    runs every other backend in-process. ``False`` forces in-process (the
    unit tests inspect the constructed model); ``True`` forces a child.
    """
    name = resolve_backend(backend)
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")
    knobs: dict = {}
    if name == "faster-whisper":
        if (compute_type is None or cpu_threads is None or beam_size is None
                or compute_device is None):
            from listn.config import Settings

            s = Settings.load()
            compute_type = s.compute_type if compute_type is None else compute_type
            cpu_threads = s.cpu_threads if cpu_threads is None else cpu_threads
            beam_size = s.beam_size if beam_size is None else beam_size
            compute_device = s.compute_device if compute_device is None else compute_device
        knobs = dict(compute_type=compute_type, cpu_threads=cpu_threads,
                     beam_size=beam_size, compute_device=compute_device)
    if isolate is None:
        isolate = name == "faster-whisper"
    key = (name, model, knobs.get("compute_type"), knobs.get("cpu_threads"),
           knobs.get("beam_size"), knobs.get("compute_device"), bool(isolate))
    with _cache_lock:
        instance = _cache.get(key)
        if instance is None:
            if isolate:
                instance = IsolatedTranscriber(model, download=status, **knobs)
            else:
                instance = _build_backend(name, model, knobs, download=status)
            _close_cached(keep=key)
            _cache.clear()  # keep one model resident; they're large
            _cache[key] = instance
    # Reported on every call, cache hit or not: where the model really runs.
    message = getattr(instance, "device_message", None)
    if status and message:
        status(message)
    return instance


def register_backend(name: str, factory) -> None:
    """Hook for tests: register a fake backend."""
    _BACKENDS[name] = factory
    with _cache_lock:
        _close_cached()
        _cache.clear()
