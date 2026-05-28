"""History-ablate perturber — zero out the action_history input.

If the policy reads action history (declared via required_inputs.action_history)
but produces the same action when history is zeroed, the model isn't
conditioning on its past actions — it's running open-loop on the current
observation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from emboviz.core.observations import ActionHistory
from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.state._state_utils import make_perturbed_history_scene


class HistoryAblatePerturber(Perturber):
    """Replace action_history with all-zeros."""

    name = "history_ablate"
    axis = "state.history_ablate"
    affects = frozenset({"action_history"})

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        history = scene.observations.action_history
        if history is None:
            return
        zeroed = np.zeros_like(history.actions)
        new_history = replace(history, actions=zeroed)
        yield make_perturbed_history_scene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="zeros",
            new_history=new_history,
            description=f"action_history zeroed ({history.timesteps_back} steps)",
            parameters={
                "timesteps_back": int(history.timesteps_back),
                "original_history_norm": float(np.linalg.norm(history.actions)),
            },
        )
