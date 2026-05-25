"""Train a LinearProbe on (hidden_states, labels) pairs.

Uses sklearn LogisticRegression for the fit; extracts the learned linear
parameters into the pure-numpy `LinearProbe` for inference. Imports sklearn
lazily so importing this module is cheap.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from emboviz.probes.base import LinearProbe, ProbeSpec


def train_linear_probe(
    hidden_states: np.ndarray,             # (n_samples, n_layers, hidden_dim)
    labels: Sequence,                       # length n_samples
    spec_base: ProbeSpec,                   # filled in with train stats below
    val_fraction: float = 0.2,
    C: float = 0.1,                         # stronger regularization than sklearn default
    max_iter: int = 1000,
    random_state: int = 0,
) -> LinearProbe:
    """Fit a multinomial logistic regression on flattened hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    n_samples = hidden_states.shape[0]
    if n_samples != len(labels):
        raise ValueError(f"hidden_states has {n_samples} samples but labels has {len(labels)}")
    if n_samples < 4:
        raise ValueError("need at least 4 samples to train a probe")

    # Flatten hidden states into features
    X = hidden_states.astype(np.float32).reshape(n_samples, -1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Train / val split (interleaved by index)
    n_val = max(1, int(n_samples * val_fraction))
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_samples)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    clf = LogisticRegression(
        C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs",
    )
    clf.fit(X[train_idx], y[train_idx])

    train_acc = float(clf.score(X[train_idx], y[train_idx]))
    val_acc = float(clf.score(X[val_idx], y[val_idx])) if n_val > 0 else float("nan")

    # Extract weights — handle binary vs multinomial.
    W = clf.coef_                           # (n_classes-or-1, feature_dim)
    b = clf.intercept_                      # (n_classes-or-1,)
    if W.shape[0] == 1:                     # binary case
        W = np.vstack([-W, W])
        b = np.array([-b[0], b[0]])

    classes = [str(c) for c in encoder.classes_]
    spec = ProbeSpec(
        name=spec_base.name,
        target_description=spec_base.target_description,
        model_id=spec_base.model_id,
        layer_indices=spec_base.layer_indices,
        classes=classes,
        train_n=int(len(train_idx)),
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        metadata={**spec_base.metadata, "C": C, "max_iter": max_iter, "random_state": random_state},
    )
    return LinearProbe(spec=spec, weights=W.astype(np.float32), bias=b.astype(np.float32))
