"""State-jitter perturber — add Gaussian noise to proprioception.

A well-grounded policy is sensitive to small state perturbations
(action should change with state). A policy that ignores state will
produce the same action under noisy state — flagged CRITICAL by the
Counterfactual diagnostic.

Default noise std (0.05) is calibrated for unit-normalized state. For
real units (rad, m), users override via the `std` kwarg or rely on
their RobotProfile.state to interpret semantics downstream.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

import numpy as np

from emboviz.core.observations import Proprioception
from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.state._state_utils import make_perturbed_state_scene


class StateJitterPerturber(Perturber):
    """Add Gaussian noise to the proprioceptive state vector."""

    name = "state_jitter"
    axis = "state.jitter"
    affects = frozenset({"state"})

    def __init__(self, std: float = 0.05, n_variants: int = 3, seed: int = 0):
        self.std = std
        self.n_variants = n_variants
        self.seed = seed

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        state = scene.observations.state
        if state is None:
            return

        rng = np.random.default_rng(self.seed)
        for i in range(self.n_variants):
            noise = rng.normal(0.0, self.std, size=state.values.shape).astype(state.values.dtype)
            new_values = state.values + noise
            new_state = replace(state, values=new_values)
            yield make_perturbed_state_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"jitter_{i}",
                new_state=new_state,
                description=f"+N(0, {self.std}) on state ({state.convention})",
                parameters={
                    "std": float(self.std),
                    "convention": state.convention,
                    "noise_norm": float(np.linalg.norm(noise)),
                },
            )
