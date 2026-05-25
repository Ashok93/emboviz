"""Color-attribute swap — tests color binding.

When the instruction names a color ("red cup"), swap it for another color
("blue cup") and check whether the action follows. Combined with a
color-perturbed scene (image_recolor), this becomes a powerful binding test.
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import (
    make_perturbed_scene,
    replace_word,
)
from emboviz.taxonomy.object_categories import COLOR_WORDS


class ColorSwapPerturber(Perturber):
    """Swap a color word in the instruction for another color."""

    name = "color_swap"
    axis = "language.color_swap"
    domain = "instruction"

    def __init__(self, max_swaps: int = 2):
        self.max_swaps = max_swaps

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        words = scene.instruction.lower().split()
        present = [w.strip(".,!?;:") for w in words if w.strip(".,!?;:") in COLOR_WORDS]
        if not present:
            return
        original = present[0]
        candidates = [c for c in COLOR_WORDS if c != original]
        for i, swap in enumerate(candidates[: self.max_swaps]):
            new_instr = replace_word(scene.instruction, original, swap)
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"{original}_to_{swap}",
                new_instruction=new_instr,
                description=f"{original} → {swap}",
                parameters={"from": original, "to": swap},
            )
