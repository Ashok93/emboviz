"""Probe-confidence metric — does a linear probe recover X from hidden states?

If a probe recovers object color with >0.9 accuracy on the hidden state but
the model's action is invariant to color → 'information is present but
unused' (the most damning mechanistic claim we can make).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from policylens.core.types import HiddenStates
from policylens.metrics.base import Metric


class ProbeConfidenceMetric(Metric):
    name = "probe_confidence"

    def __init__(self, probe_fn: Callable[[np.ndarray], float]):
        """`probe_fn` consumes a (n_layers, hidden_dim) array and returns
        a calibrated confidence (0–1) for whatever the probe was trained on."""
        self.probe_fn = probe_fn

    def compute(self, hs: HiddenStates) -> float:
        return float(self.probe_fn(hs.states))
