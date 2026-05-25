"""Distribution-divergence and similarity metrics.

Used for attention comparison (JS over image-token attention), distribution
shift detection on hidden states, and probe rank-correlation.
"""

from __future__ import annotations

import numpy as np


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JS divergence (base-2, in [0, 1]) between two non-negative arrays.

    Inputs are auto-normalized into probability distributions.
    """
    p = np.clip(p.astype(np.float64), eps, None)
    q = np.clip(q.astype(np.float64), eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(np.float64), eps, None)
    q = np.clip(q.astype(np.float64), eps, None)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log2(p) - np.log2(q))))


def entropy_normalized(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy normalized to [0, 1] (1 = uniform, 0 = delta)."""
    p = np.clip(p.astype(np.float64), eps, None)
    p /= p.sum()
    h = -float(np.sum(p * np.log2(p)))
    return h / max(np.log2(p.size), 1e-9)


def concentration(p: np.ndarray) -> float:
    """1 − normalized entropy; high = peaked, low = uniform."""
    return 1.0 - entropy_normalized(p)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Rank-correlation between two arrays.

    Imports scipy lazily — only when called — so this module imports clean
    without scipy present.
    """
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)
