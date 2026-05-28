"""Past actions fed back to the policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ActionSource = Literal[
    "model",      # the policy's OWN prior predictions (deployment / closed-loop)
    "expert",     # ground-truth actions from the dataset (offline / teacher-forced)
]


@dataclass(frozen=True)
class ActionHistory:
    """Recent actions provided as input to the policy.

    `source` is load-bearing semantics, not metadata: a diagnostic that
    measures "does the model recover when given its own bad history?"
    is meaningful only with source=='model'. A diagnostic that measures
    "is the model conditioning correctly on its history at all?" works
    with either source but means different things. Callers MUST be
    explicit; this dataclass enforces it.
    """

    actions: np.ndarray            # shape (T, action_dim) — T timesteps back
    source: ActionSource
    timesteps_back: int            # equals actions.shape[0]
