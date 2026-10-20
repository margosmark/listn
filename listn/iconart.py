"""Draw the listn app icon as PNG files. No Pillow.

The mark is an indigo rounded square and a white waveform. ``iconutil`` on
a Mac turns the iconset into AppIcon.icns. The PNGs themselves are what the
repo stores, so the build does not need a design tool.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ICONSET = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def _png(width: int, height: int, rgba: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = []
    stride = width * 4
    for y in range(height):
        rows.append(b"\x00" + rgba[y * stride:(y + 1) * stride])
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")


def _inside(x: int, y: int, size: int, radius: int) -> bool:
    if radius <= 0:
        return True
    dx = 0
    if x < radius:
        dx = radius - x
    elif x >= size - radius:
        dx = x - (size - radius - 1)
    dy = 0
    if y < radius:
        dy = radius - y
    elif y >= size - radius:
        dy = y - (size - radius - 1)
    return dx * dx + dy * dy <= radius * radius


def render(size: int) -> bytes:
    radius = max(1, int(size * 0.22))
    pixels = bytearray(size * size * 4)
    bars = 5
    gap = max(1, size // 28)
    bar_w = max(1, (size // 2) // bars)
    span = bars * bar_w + (bars - 1) * gap
    origin = (size - span) // 2
    heights = (0.34, 0.62, 0.9, 0.55, 0.28)
    for y in range(size):
        for x in range(size):
            i = (y * size + x) * 4
            if not _inside(x, y, size, radius):
                continue
            pixels[i:i + 4] = bytes((79, 70, 229, 255))
            for index, height in enumerate(heights):
                left = origin + index * (bar_w + gap)
                right = left + bar_w
                half = int(size * height / 2)
                top = size // 2 - half
                bottom = size // 2 + half
                if left <= x < right and top <= y < bottom:
                    pixels[i:i + 4] = bytes((255, 255, 255, 255))
    return _png(size, size, bytes(pixels))


def write_iconset(directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, size in ICONSET:
        (directory / name).write_bytes(render(size))
    return directory
