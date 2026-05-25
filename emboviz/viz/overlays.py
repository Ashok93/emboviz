"""Heatmap-overlay primitive."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55,
                    cmap_name: str = "jet") -> np.ndarray:
    import matplotlib.pyplot as plt

    if heatmap.shape != frame.shape[:2]:
        pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        heatmap = np.asarray(pil, dtype=np.float32) / 255.0
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
