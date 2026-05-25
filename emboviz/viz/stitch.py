"""Vertical PIL section stitcher — combine several rendered PNGs into one."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def stitch_vertical(image_paths: Iterable[Path], out_path: Path,
                    gap: int = 24, padding: int = 20) -> Path:
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    if not imgs:
        raise ValueError("stitch_vertical needs at least one image")
    max_w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1) + 2 * padding
    canvas = Image.new("RGB", (max_w + 2 * padding, total_h), "white")
    y = padding
    for im in imgs:
        x = (canvas.width - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
