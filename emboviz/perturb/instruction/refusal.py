"""Refusal-on-absent-object test.

Replace the target noun with an object that is unlikely to be in the scene
(an exotic noun from a different category). A grounded model should refuse
or hesitate; a vision-blind one will still act.
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import (
    make_perturbed_scene,
    replace_word,
)
from emboviz.perturb.instruction.noun_swap import _pick_target_noun
from emboviz.taxonomy.object_categories import OBJECT_CATEGORIES


# Words from non-overlapping categories — unlikely to actually be present
# in any Bridge scene whose instruction mentions a utensil/container.
EXOTIC_NOUNS = [
    "elephant", "trombone", "violin", "telescope",
    "umbrella", "lampshade", "guitar",
]


class RefusalPerturber(Perturber):
    """Swap the target noun for an exotic out-of-distribution noun."""

    name = "refusal_absent"
    axis = "language.refusal_absent"
    affects = frozenset({"instruction"})

    def __init__(self, max_swaps: int = 2):
        self.max_swaps = max_swaps

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        target, _ = _pick_target_noun(scene.instruction)
        if target is None:
            return
        for swap in EXOTIC_NOUNS[: self.max_swaps]:
            new_instr = replace_word(scene.instruction, target, swap)
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"{target}_to_{swap}",
                new_instruction=new_instr,
                description=f"target='{swap}' (absent from scene)",
                parameters={"target_original": target, "exotic": swap},
            )
