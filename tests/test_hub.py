"""Cached model loads must not touch the Hub."""

import os

import pytest

from listn.hub import load_local_or_download


def test_cached_load_sets_hub_offline_and_skips_the_network(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    calls = []

    def local():
        calls.append(os.environ.get("HF_HUB_OFFLINE"))
        return "/cache/model"

    def remote():
        raise AssertionError("cached model must not be downloaded")

    assert load_local_or_download(local, remote) == "/cache/model"
    assert calls == ["1"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_missing_model_downloads_once_then_locks_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    seen = []
    notes = []

    def local():
        seen.append(("local", os.environ.get("HF_HUB_OFFLINE")))
        raise FileNotFoundError("not cached")

    def remote():
        seen.append(("remote", os.environ.get("HF_HUB_OFFLINE")))
        return "downloaded"

    assert load_local_or_download(local, remote, on_download=lambda: notes.append("dl")) == "downloaded"
    assert seen == [("local", "1"), ("remote", None)]
    assert notes == ["dl"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_failed_download_still_restores_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def local():
        raise FileNotFoundError("not cached")

    def remote():
        raise RuntimeError("hub down")

    with pytest.raises(RuntimeError, match="hub down"):
        load_local_or_download(local, remote)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
