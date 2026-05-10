#!/usr/bin/env python3
# Generates deterministic PPM images for PaliGemma2 QK BF16 A/B sweeps.
#
# These are not a replacement for real validation images. They are small,
# dependency-free fixtures that exercise different pixel distributions so the
# deterministic generation A/B script can catch obvious precision instability
# before a larger real-image set is available.

from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paligemma" / "testdata" / "qk_bf16_sweep"
SIZE = 448


def clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def write_ppm(path: Path, pixels: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{SIZE} {SIZE}\n255\n".encode("ascii"))
        f.write(bytes(channel for pixel in pixels for channel in pixel))


def gradient_blocks() -> list[tuple[int, int, int]]:
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            block_x = x // 56
            block_y = y // 56
            r = (x * 255) / (SIZE - 1)
            g = (y * 255) / (SIZE - 1)
            b = 48 + ((block_x * 31 + block_y * 47) % 160)
            pixels.append((clamp(r), clamp(g), clamp(b)))
    return pixels


def checker_edges() -> list[tuple[int, int, int]]:
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            tile = ((x // 28) + (y // 28)) & 1
            edge = 255 if (x % 28 in (0, 1) or y % 28 in (0, 1)) else 0
            base = 220 if tile else 28
            pixels.append((clamp(base + edge * 0.10), clamp(base), clamp(255 - base)))
    return pixels


def color_quadrants() -> list[tuple[int, int, int]]:
    colors = (
        (230, 36, 48),
        (36, 190, 82),
        (42, 98, 230),
        (238, 210, 48),
    )
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            idx = (1 if x >= SIZE // 2 else 0) + (2 if y >= SIZE // 2 else 0)
            r, g, b = colors[idx]
            stripe = 24 if ((x + y) // 16) & 1 else -24
            pixels.append((clamp(r + stripe), clamp(g + stripe), clamp(b + stripe)))
    return pixels


def dark_radial() -> list[tuple[int, int, int]]:
    pixels = []
    center = (SIZE - 1) / 2.0
    max_dist = math.sqrt(2.0 * center * center)
    for y in range(SIZE):
        for x in range(SIZE):
            dx = x - center
            dy = y - center
            dist = math.sqrt(dx * dx + dy * dy) / max_dist
            ring = 0.5 + 0.5 * math.sin(dist * 42.0)
            value = 18 + 90 * (1.0 - dist) + 28 * ring
            pixels.append((clamp(value * 0.75), clamp(value * 0.95), clamp(value * 1.25)))
    return pixels


def main() -> None:
    images = {
        "gradient_blocks.ppm": gradient_blocks,
        "checker_edges.ppm": checker_edges,
        "color_quadrants.ppm": color_quadrants,
        "dark_radial.ppm": dark_radial,
    }
    for name, generator in images.items():
        path = OUT_DIR / name
        write_ppm(path, generator())
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
