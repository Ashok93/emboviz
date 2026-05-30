"""Action-space distance functions.

Used by metrics to compare two ActionResults. Adapters may override the
default L2 by providing their own distance via `VLAModel.compare_actions`,
but these here are the model-agnostic fallbacks.
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.types import ActionResult


def l2_distance(a: ActionResult, b: ActionResult) -> float:
    """Euclidean distance between two action vectors."""
    return float(np.linalg.norm(a.action - b.action))


def l1_distance(a: ActionResult, b: ActionResult) -> float:
    return float(np.abs(a.action - b.action).sum())


def per_dim_abs_diff(a: ActionResult, b: ActionResult) -> np.ndarray:
    """Per-dimension absolute differences — for diagnostic dim-wise inspection."""
    return np.abs(a.action - b.action)


def cosine_distance(a: ActionResult, b: ActionResult, eps: float = 1e-9) -> float:
    """1 − cosine similarity; useful for direction comparisons."""
    na = np.linalg.norm(a.action)
    nb = np.linalg.norm(b.action)
    if na < eps or nb < eps:
        return 1.0
    return float(1.0 - np.dot(a.action, b.action) / (na * nb))


def normalized_l2(
    a: ActionResult, b: ActionResult, action_scale: np.ndarray
) -> float:
    """L2 with per-dimension scaling — for action spaces with mixed units (m, rad).

    `action_scale` is a per-dim normalizer (e.g., q99 − q01 from training stats).
    """
    diff = (a.action - b.action) / np.where(action_scale > 1e-9, action_scale, 1.0)
    return float(np.linalg.norm(diff))
