"""Hugging Face downloads, then silence.

The first time a model is missing, listn may download it. That request is
unauthenticated unless a token is passed for a gated model, and the Hub warns
about that. Once the files are cached, loads set ``HF_HUB_OFFLINE`` (and
``TRANSFORMERS_OFFLINE``) so the load itself does not touch the network.
A failed offline attempt clears the flag for one download, then sets it again.

``huggingface_hub`` reads ``HF_HUB_OFFLINE`` once, at import, so the
environment variable alone does nothing after faster-whisper has imported
it. The module constant is switched as well.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def _set_hub_constant(offline: bool) -> None:
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and hasattr(constants, "HF_HUB_OFFLINE"):
        constants.HF_HUB_OFFLINE = offline


def enable_hub_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _set_hub_constant(True)


def disable_hub_offline() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    _set_hub_constant(False)


def hub_offline() -> bool:
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and hasattr(constants, "HF_HUB_OFFLINE"):
        return bool(constants.HF_HUB_OFFLINE)
    return os.environ.get("HF_HUB_OFFLINE") == "1"


def is_cached(repo_id: str, filename: str) -> bool:
    """True when ``filename`` of ``repo_id`` is in the local Hub cache. No network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        found = try_to_load_from_cache(repo_id, filename)
    except Exception:
        return False
    return isinstance(found, str) and os.path.exists(found)


def redact(text: object, *secrets: str | None) -> str:
    """``text`` with every token replaced, so errors can be logged and shown."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "[token]")
    return out


def load_local_or_download(
    load_local: Callable[[], T],
    load_remote: Callable[[], T],
    *,
    on_download: Callable[[], None] | None = None,
) -> T:
    """Try ``load_local`` with the Hub forced offline.

    If that raises, call ``load_remote`` with the network allowed, then turn
    the Hub back off so anything that follows reads the cache.
    """
    enable_hub_offline()
    try:
        return load_local()
    except Exception:
        disable_hub_offline()
    if on_download:
        on_download()
    try:
        return load_remote()
    finally:
        enable_hub_offline()
