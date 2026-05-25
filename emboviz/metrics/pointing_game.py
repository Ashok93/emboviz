"""Pointing-game metric — does attention land inside the target bbox?

Standard interpretability metric (Q-GroundCAM and predecessors): take the
attention from a noun token to the image, find the peak patch, check
whether it falls inside the ground-truth bounding box of the referent.
"""

from __future__ import annotations

import numpy as np

from emboviz.core.types import AttentionMaps
from emboviz.metrics.base import Metric


class PointingGameMetric(Metric):
    """Returns hit-rate (0–1) across (layer, head) — fraction whose argmax
    patch overlaps the supplied bbox."""

    name = "pointing_game"

    def compute(
        self,
        attention: AttentionMaps,
        bbox_normalized: tuple[float, float, float, float],
    ) -> float:
        """`bbox_normalized` is (x0, y0, x1, y1) in [0, 1]^2 coordinates."""
        x0, y0, x1, y1 = bbox_normalized
        img = attention.image_weights()  # (L, H, G, G)
        L, H, G, _ = img.shape
        hits = 0
        total = 0
        for li in range(L):
            for hi in range(H):
                row, col = np.unravel_index(np.argmax(img[li, hi]), (G, G))
                cx = (col + 0.5) / G
                cy = (row + 0.5) / G
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    hits += 1
                total += 1
        return hits / max(total, 1)
