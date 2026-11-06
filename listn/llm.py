"""Local text generation.

The default model is ``mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit``
(about 17 GB on disk and in memory while loaded). It is a mixture-of-experts
model with about 3B active parameters, so it generates about as fast as a 7B
model while writing clearly better notes: on real meetings it kept owners
straight, found commitments the 7B missed, and invented fewer items. The
previous default, Qwen2.5-7B-Instruct 4-bit (4.3 GB), still works and is the
choice for Macs with less memory; see ``RECOMMENDED_LLMS``. The model is
unloaded after a few idle minutes, so it does not sit next to Whisper.

The weights download once from Hugging Face. After that, loads set
``HF_HUB_OFFLINE``. Machines without MLX (and the test suite) use the
extractive fallback in ``listn.summary`` instead of calling this module.

Threading: MLX streams are per thread. ``mlx_lm`` creates its generation
stream when it is imported, so a model imported on one thread and run on
another fails with "There is no Stream(gpu, 0) in current thread". That is
what happened when notes were written from the after-recording thread or a
server worker. Every load and every generation therefore runs on one
dedicated worker thread, which also keeps the loaded model between calls and
drops it after a few idle minutes.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from listn.hub import load_local_or_download

log = logging.getLogger("listn")

DEFAULT_LLM = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"

# Model id -> rough download size in GB. All run locally with MLX.
RECOMMENDED_LLMS = {
    "mlx-community/Qwen2.5-7B-Instruct-4bit": 4.3,
    "mlx-community/Qwen2.5-14B-Instruct-4bit": 8.3,
    "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit": 17.2,
}

IDLE_UNLOAD_SECONDS = 300.0

# The notes backends. ``auto`` resolves to MLX when it is installed (macOS,
# unchanged), otherwise the operator's Ollama endpoint when one is configured,
# otherwise nothing at all - listn then writes extractive notes. Selecting
# ``ollama`` or configuring ``llm_ollama_url`` is the opt-in; with neither,
# listn makes no outbound connection for notes.
LLM_BACKENDS = ("auto", "mlx", "ollama", "extractive")

# One HTTP request to an existing Ollama service. listn holds no model of its
# own on this path: loading and unloading the weights is the Ollama host's
# job, so there is nothing to unload here.
OLLAMA_TIMEOUT_SECONDS = 180.0

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="listn-llm")
_state: dict = {"repo": None, "model": None, "tokenizer": None, "last": 0.0, "stream_ready": False}
_timer_lock = threading.Lock()
_timer: list = [None]


def mlx_lm_available() -> bool:
    """True when mlx_lm is installed. Does not import it on the caller's thread."""
    try:
        return importlib.util.find_spec("mlx_lm") is not None
    except (ImportError, ValueError):
        return False


class OllamaError(RuntimeError):
    """An Ollama request failed.

    Carries no prompt, transcript or model output on purpose: this message is
    what reaches the log when a notes run falls back.
    """


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the allowlist is the one configured origin.

    A 301/302/303 followed off that origin would contact a second endpoint
    and accept its body as the operator's notes while still labelling them
    ``ollama/<model>`` (security review F1, egress check 2).
    ``redirect_request`` returning ``None`` makes the 3xx surface as an
    ``HTTPError`` carrying its status instead of being followed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _redacted_origin(raw) -> str:
    """``scheme://host[:port]`` of a rejected value, for an error message.

    Credentials, path, query and fragment are dropped on purpose: these
    messages surface in ``SettingsError`` and in logs, so a rejected URL is
    never echoed in full (security review N1).
    """
    parts = urllib.parse.urlsplit(str(raw or "").strip())
    if not parts.scheme:
        return "no scheme"
    if not parts.hostname:
        return f"{parts.scheme}://"
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    try:
        port = parts.port
    except ValueError:
        return f"{parts.scheme}://{host}"
    origin = f"{parts.scheme}://{host}"
    return f"{origin}:{port}" if port else origin


def validate_ollama_url(raw) -> str:
    """Return the bare ``scheme://host[:port]`` origin of an Ollama endpoint.

    Configuration is the allowlist: listn talks to exactly one endpoint the
    operator wrote down, and only over plain HTTP(S) as a bare origin - no
    credentials, path or query. Anything else raises ``ValueError``.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise ValueError(
            "llm_ollama_url must be an http(s) origin such as "
            f"http://10.10.10.6:11434 - got {_redacted_origin(raw)}")
    if not parts.hostname:
        raise ValueError(f"llm_ollama_url has no host - got {_redacted_origin(raw)}")
    if parts.username or parts.password:
        raise ValueError("llm_ollama_url must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("llm_ollama_url must be a bare origin (no query or fragment)")
    if parts.path.strip("/"):
        raise ValueError(f"llm_ollama_url must be the bare origin, not a path - got {raw!r}")
    try:
        port = parts.port
    except ValueError:
        raise ValueError(f"llm_ollama_url has an invalid port - got {raw!r}") from None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    origin = f"{parts.scheme}://{host}"
    return f"{origin}:{port}" if port else origin


def notes_backend(settings=None, *, mlx_available: bool | None = None) -> str | None:
    """Which generative backend this machine has: ``"mlx"``, ``"ollama"`` or ``None``.

    ``None`` means no generative backend is available or wanted, and callers
    must use the extractive notes in ``listn.summary``.
    """
    if settings is None:
        from listn.config import Settings

        settings = Settings.load()
    if mlx_available is None:
        mlx_available = mlx_lm_available()
    choice = str(getattr(settings, "llm_backend", "auto") or "auto").strip().lower()
    endpoint = str(getattr(settings, "llm_ollama_url", "") or "").strip()
    model = str(getattr(settings, "llm_ollama_model", "") or "").strip()
    ollama = "ollama" if (endpoint and model) else None
    if choice == "extractive":
        return None
    if choice == "mlx":
        return "mlx" if mlx_available else None
    if choice == "ollama":
        return ollama
    return "mlx" if mlx_available else ollama


def ollama_generate(prompt: str, model: str, url: str, max_tokens: int = 900,
                    system: str | None = None, temperature: float = 0.0,
                    timeout: float | None = None) -> str:
    """One chat request to an existing Ollama service. Raises ``OllamaError``.

    ``think`` is disabled on purpose: Qwen3-class models otherwise spend most
    of the reply on a private reasoning field, which burns the note budget and
    can leave ``content`` empty. listn never pulls a model and never installs a
    runtime - the weights live on the Ollama host and listn is an HTTP client.

    Only numbers are logged. The prompt, the transcript and the reply
    never appear in a log line or an exception message.
    """
    timeout = OLLAMA_TIMEOUT_SECONDS if timeout is None else float(timeout)
    origin = validate_ollama_url(url)
    if not origin:
        raise OllamaError("no Ollama endpoint configured (llm_ollama_url is empty)")
    if not str(model or "").strip():
        raise OllamaError("no Ollama model configured (llm_ollama_model is empty)")
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": str(model),
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": int(max_tokens), "temperature": float(temperature)},
    }
    request = urllib.request.Request(
        origin + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.time()
    # Proxies are bypassed on purpose: this is a direct request to the
    # operator's own endpoint, and a stray HTTP_PROXY must not reroute the
    # transcript through a third party. _NoRedirect (F1) is installed
    # alongside: a redirect is answered, never followed, so the allowlisted
    # origin stays the only one listn ever talks to.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            # Second gate (F1): only the allowlisted origin's own URL may
            # be read, and it is read before the body is parsed, so a body
            # from anywhere else can never become notes.
            if response.geturl() != origin + "/api/chat":
                raise OllamaError(
                    f"Ollama at {origin} answered from a different URL; "
                    "listn follows no redirects (single allowlisted origin)")
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise OllamaError(
                f"Ollama at {origin} answered HTTP {exc.code} (redirect refused: "
                "only the allowlisted origin may answer)") from None
        raise OllamaError(f"Ollama at {origin} answered HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"Ollama at {origin} is unreachable ({type(exc).__name__})") from None
    except ValueError:
        raise OllamaError(f"Ollama at {origin} answered with a body that is not JSON") from None
    elapsed = time.time() - started
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaError(f"Ollama at {origin} returned an empty reply")
    generated = body.get("eval_count") or 0
    eval_seconds = (body.get("eval_duration") or 0) / 1e9
    if generated and eval_seconds > 0:
        log.info("ollama: %s prompt tokens, %s generated at %.1f tok/s (%.1fs request)",
                 body.get("prompt_eval_count"), generated, generated / eval_seconds, elapsed)
    else:
        log.info("ollama: %s prompt tokens, %s generated (%.1fs request)",
                 body.get("prompt_eval_count"), generated, elapsed)
    return content


def _snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    def cached():
        return snapshot_download(repo, local_files_only=True)

    def fetched():
        return snapshot_download(repo)

    return load_local_or_download(cached, fetched)


def _ensure_stream() -> None:
    """Give mlx_lm a generation stream that belongs to this (worker) thread."""
    if _state["stream_ready"]:
        return
    import mlx.core as mx

    module = sys.modules.get("mlx_lm.generate")
    if module is not None and hasattr(module, "generation_stream"):
        module.generation_stream = mx.new_stream(mx.default_device())
    _state["stream_ready"] = True


def _load(repo: str):
    import mlx_lm

    _ensure_stream()
    if _state["repo"] != repo or _state["model"] is None:
        _unload_now()
        path = _snapshot(repo)
        os.environ["HF_HUB_OFFLINE"] = "1"
        started = time.time()
        model, tokenizer = mlx_lm.load(path)
        log.info("Loaded %s in %.1fs", repo, time.time() - started)
        _state.update(repo=repo, model=model, tokenizer=tokenizer)
    return _state["model"], _state["tokenizer"]


def _unload_now() -> None:
    if _state["model"] is None:
        return
    _state.update(repo=None, model=None, tokenizer=None)
    try:
        import gc

        import mlx.core as mx

        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:
        pass


def _maybe_unload() -> None:
    if _state["model"] is not None and time.time() - _state["last"] >= IDLE_UNLOAD_SECONDS - 1:
        log.info("Unloading idle notes model")
        _unload_now()


def _schedule_unload() -> None:
    with _timer_lock:
        if _timer[0] is not None:
            _timer[0].cancel()
        timer = threading.Timer(IDLE_UNLOAD_SECONDS, lambda: _executor.submit(_maybe_unload))
        timer.daemon = True
        timer.start()
        _timer[0] = timer


def chat_prompt(tokenizer, prompt: str, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return (system + "\n\n" if system else "") + prompt
    for extra in ({"enable_thinking": False}, {}):
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True, **extra)
        except TypeError:
            continue
        except Exception:
            break
    return (system + "\n\n" if system else "") + prompt


def _run(prompt: str, repo: str, max_tokens: int, system: str | None, temperature: float) -> str:
    import mlx_lm

    model, tokenizer = _load(repo)
    kwargs = {"max_tokens": max_tokens, "verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(temp=max(0.0, float(temperature)))
    except Exception:
        pass
    try:
        return mlx_lm.generate(model, tokenizer, prompt=chat_prompt(tokenizer, prompt, system), **kwargs)
    finally:
        _state["last"] = time.time()


def _load_settings():
    from listn.config import Settings

    return Settings.load()


def generate(prompt: str, model_id: str = DEFAULT_LLM, max_tokens: int = 900,
             system: str | None = None, temperature: float = 0.0,
             *, settings=None, backend: str | None = None) -> str:
    """Run one prompt on the machine's local notes backend.

    With ``backend="ollama"`` (or ``settings`` whose ``llm_backend`` resolves to
    Ollama) the prompt goes to the operator's Ollama endpoint as one HTTP
    request. Otherwise it runs on the local MLX model exactly as before - which
    is what raises when MLX is not installed, so callers keep their extractive
    fallback.
    """
    if backend is None and settings is not None:
        backend = notes_backend(settings)
    if backend == "ollama":
        settings = settings or _load_settings()
        return ollama_generate(
            prompt,
            str(getattr(settings, "llm_ollama_model", "") or ""),
            str(getattr(settings, "llm_ollama_url", "") or ""),
            max_tokens=max_tokens, system=system, temperature=temperature,
        )
    repo = (model_id or DEFAULT_LLM).strip()
    try:
        return _executor.submit(_run, prompt, repo, max_tokens, system, temperature).result()
    finally:
        _schedule_unload()


def unload() -> None:
    """Drop the cached model now (frees its memory)."""
    _executor.submit(_unload_now).result()
