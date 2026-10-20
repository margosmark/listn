"""Hugging Face downloads, then silence.

The first time a model is missing, listn may download it. That request is
unauthenticated unless ``HF_TOKEN`` is set, and the Hub warns about that.
Once the files are cached, loads set ``HF_HUB_OFFLINE`` (and
``TRANSFORMERS_OFFLINE``) so the load itself does not touch the network.
A failed offline attempt clears the flag for one download, then sets it again.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def enable_hub_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def disable_hub_offline() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


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
