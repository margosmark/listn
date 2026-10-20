"""User terms passed to Whisper as a prompt, and as hotwords when supported."""

from __future__ import annotations


def clean_vocabulary(words) -> list[str]:
    if isinstance(words, str):
        words = words.replace(",", "\n").splitlines()
    if not isinstance(words, list):
        return []
    out = []
    for item in words:
        text = " ".join(str(item).split())
        if not text or text in out:
            continue
        out.append(text[:60])
        if len(out) >= 200:
            break
    return out


def prompt_with_vocabulary(vocabulary: list[str], recent: str | None) -> str | None:
    parts = []
    words = clean_vocabulary(vocabulary)
    if words:
        parts.append("Terms: " + ", ".join(words))
    if recent and recent.strip():
        parts.append(recent.strip())
    text = " ".join(parts).strip()
    return text[:800] or None


def hotword_string(vocabulary: list[str]) -> str | None:
    words = clean_vocabulary(vocabulary)
    if not words:
        return None
    return ", ".join(words)[:400]
