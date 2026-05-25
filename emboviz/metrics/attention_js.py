"""Attention Jensen-Shannon divergence — compare image-attention between two runs.

Used by the cross-modal binding diagnostic: for each (layer, head), how
different is the image-token attention under noun A vs noun B?
"""

from __future__ import annotations

import numpy as np

from emboviz.core.divergences import jensen_shannon
from emboviz.core.types import AttentionMaps
from emboviz.metrics.base import Metric


class AttentionJSMetric(Metric):
    """Returns per-head Jensen-Shannon divergence map between two AttentionMaps."""

    name = "attention_js"

    def compute(self, a: AttentionMaps, b: AttentionMaps) -> np.ndarray:
        """Returns (n_layers, n_heads) JS divergence array on image attention."""
        img_a = a.image_weights()    # (L, H, G, G)
        img_b = b.image_weights()
        L, H, G, _ = img_a.shape
        out = np.zeros((L, H), dtype=np.float32)
        for li in range(L):
            for hi in range(H):
                out[li, hi] = jensen_shannon(img_a[li, hi].flatten(),
                                              img_b[li, hi].flatten())
        return out

    def compute_scalar(self, a: AttentionMaps, b: AttentionMaps) -> float:
        """Convenience scalar: max JS across heads (largest per-head difference)."""
        return float(self.compute(a, b).max())
