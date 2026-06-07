# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action normalization — VENDORED from NVIDIA cosmos-framework.

Source: ``cosmos_framework/data/vfm/action/action_normalization.py`` (OpenMDW-1.1),
vendored verbatim except that ``normalize_action`` operates on NumPy (the torch
``.clamp`` calls become ``np.clip``) and the cosmos-framework logger import is
dropped. The math is unchanged — the quantile mapping is exactly what the model
was trained against, so it must stay byte-faithful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    with path.open("r") as f:
        raw = json.load(f)
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(f"Action normalization stats block {stats_key!r} in {stats_path} must be a dict.")
    elif stats_key != "global":
        raise KeyError(f"Action normalization stats block {stats_key!r} not found in {stats_path}.")
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    return {key: np.array(value, dtype=np.float32) for key, value in raw.items() if key in stat_keys}


def normalize_action(
    action: np.ndarray,
    method: str,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Normalize an action array (NumPy port of cosmos-framework's normalize_action)."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        denom = np.clip(q99 - q01, 1e-8, None)
        return np.clip(2.0 * (action - q01) / denom - 1.0, -1.0, 1.0)
    if method == "meanstd":
        return (action - stats["mean"]) / np.clip(stats["std"], 1e-8, None)
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        denom = np.clip(hi - lo, 1e-8, None)
        return np.clip(2.0 * (action - lo) / denom - 1.0, -1.0, 1.0)
    raise ValueError(f"Unknown normalization method: {method!r}")
