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
import logging
import os
import sys
import threading
import time
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


def generate(prompt: str, model_id: str = DEFAULT_LLM, max_tokens: int = 900,
             system: str | None = None, temperature: float = 0.0) -> str:
    """Run one prompt on a local MLX model. Raises if MLX is not installed."""
    repo = (model_id or DEFAULT_LLM).strip()
    try:
        return _executor.submit(_run, prompt, repo, max_tokens, system, temperature).result()
    finally:
        _schedule_unload()


def unload() -> None:
    """Drop the cached model now (frees its memory)."""
    _executor.submit(_unload_now).result()
