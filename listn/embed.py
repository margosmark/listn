"""Local text embeddings with no model download.

Each text becomes a unit vector by feature hashing: word tokens, word
pairs, and character trigrams. Shared names and phrases land in the same
buckets, so a query can match a line that does not contain it as a
substring. The index stays on disk and never contacts the network.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

DIM = 256
_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    words = tokens(text)
    if not words:
        return vec
    features: list[str] = []
    for word in words:
        features.append("w:" + word)
        padded = f" {word} "
        features.extend("c:" + padded[i:i + 3] for i in range(len(padded) - 2))
    features.extend("b:" + a + "_" + b for a, b in zip(words, words[1:]))
    for feature in features:
        digest = hashlib.blake2s(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))
