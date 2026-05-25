"""History-scramble perturber — shuffle action_history along time.

Sibling to `HistoryAblatePerturber`. Where ablate tests "does the model
care that history exists at all?", scramble tests "does the model use
the temporal ORDER of history, or just the bag of recent actions?"

A policy whose action is invariant to temporal shuffling is treating
action_history as a global summary rather than a sequence.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from emboviz.core.observations import ActionHistory
from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.state._state_utils import make_perturbed_history_scene


class HistoryScramblePerturber(Perturber):
    """Shuffle action_history along the time axis."""

    name = "history_scramble"
    axis = "state.history_scramble"
    affects = frozenset({"action_history"})

    def __init__(self, n_variants: int = 3, seed: int = 0):
        self.n_variants = n_variants
        self.seed = seed

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        history = scene.observations.action_history
        if history is None:
            return
        if history.actions.shape[0] < 2:
            return  # cannot meaningfully shuffle a length-1 history

        rng = np.random.default_rng(self.seed)
        for i in range(self.n_variants):
            perm = rng.permutation(history.actions.shape[0])
            scrambled = history.actions[perm].copy()
            new_history = replace(history, actions=scrambled)
            yield make_perturbed_history_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"shuffle_{i}",
                new_history=new_history,
                description=f"action_history shuffled (perm={perm.tolist()})",
                parameters={
                    "permutation": perm.tolist(),
                    "timesteps_back": int(history.timesteps_back),
                },
            )
