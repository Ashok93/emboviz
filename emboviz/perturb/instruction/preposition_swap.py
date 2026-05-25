"""Preposition swap — tests spatial-relation grounding.

If the instruction contains "on top of", we swap to "underneath" and check
whether the action changes. Models that follow the instruction will move
in the opposite direction; models that ignore spatial language won't.
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import make_perturbed_scene
from emboviz.taxonomy.spatial_prepositions import PREPOSITION_PAIRS


def _find_preposition(instruction: str) -> tuple[str, str] | None:
    """Find the first preposition pair where one side appears in instruction."""
    lowered = instruction.lower()
    for a, b in PREPOSITION_PAIRS:
        if f" {a} " in f" {lowered} ":
            return a, b
        if f" {b} " in f" {lowered} ":
            return b, a
    return None


class PrepositionSwapPerturber(Perturber):
    """Replace one spatial preposition with its opposite."""

    name = "preposition_swap"
    axis = "language.preposition_swap"
    domain = "instruction"

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        found = _find_preposition(scene.instruction)
        if found is None:
            return
        original, opposite = found
        # Case-preserving replace: replace once.
        idx = scene.instruction.lower().find(original)
        new_instr = (
            scene.instruction[:idx]
            + opposite
            + scene.instruction[idx + len(original) :]
        )
        yield make_perturbed_scene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id=f"{original}_to_{opposite}".replace(" ", "_"),
            new_instruction=new_instr,
            description=f"{original} → {opposite}",
            parameters={"from": original, "to": opposite},
        )
