"""Probe protocol + a minimal pure-numpy logistic-regression probe.

Why pure numpy? It keeps probes self-serializable to JSON without pickle and
avoids a hard sklearn dependency at runtime. Training uses sklearn (a single
LBFGS fit), then we extract the learned `(W, b)` and reconstruct the probe
as a pure-numpy callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProbeSpec:
    """Metadata describing what a probe predicts and how it was trained."""

    name: str                       # e.g. "object_color_3class"
    target_description: str         # human-readable: "Color of target object (red/blue/green)"
    model_id: str                   # which VLA the probe is fit for
    layer_indices: list[int]        # which hidden-state layers feed the probe
    classes: list[str]              # ordered class labels (binary: 2 labels)
    train_n: int = 0                # samples used to fit
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LinearProbe:
    """A trained linear probe — pure numpy weights for portability.

    Predicts class probabilities by `softmax(features @ W.T + b)`.

    `features` is the flattened concatenation of the requested layers'
    hidden states at the query position: shape (n_layers * hidden_dim,).
    """

    spec: ProbeSpec
    weights: np.ndarray             # (n_classes, feature_dim)
    bias: np.ndarray                # (n_classes,)

    @property
    def n_classes(self) -> int:
        return self.weights.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.weights.shape[1]

    def features_from_hidden_states(self, hs_states: np.ndarray) -> np.ndarray:
        """Convert (n_layers, hidden_dim) → (feature_dim,) by flattening."""
        if hs_states.shape[0] != len(self.spec.layer_indices):
            raise ValueError(
                f"hidden states have {hs_states.shape[0]} layers, probe expects "
                f"{len(self.spec.layer_indices)}"
            )
        return hs_states.astype(np.float32).flatten()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class-probability vector of length n_classes."""
        logits = features @ self.weights.T + self.bias
        # softmax with numerical-stability shift
        logits = logits - logits.max()
        e = np.exp(logits)
        return (e / e.sum()).astype(np.float32)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) for argmax class."""
        probs = self.predict_proba(features)
        idx = int(np.argmax(probs))
        return self.spec.classes[idx], float(probs[idx])
