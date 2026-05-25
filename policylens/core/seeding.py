"""Reproducibility helpers.

Every diagnostic should accept a seed for the random elements
(image perturbations with stochastic placement, attention rollout sampling
when probabilistic, etc.). We hash a tuple into a deterministic 32-bit seed
so multiple diagnostics on the same scene get distinct but stable seeds.
"""

from __future__ import annotations

import hashlib
from typing import Any


def deterministic_seed(*parts: Any) -> int:
    """Hash arbitrary args into a 32-bit positive integer seed."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF
