"""Noun-swap perturber — the workhorse.

Replaces the manipulated-object noun in the instruction with another
within-category noun (spoon → fork, bowl → plate). The cleanest test of
linguistic binding: same scene, same grammar, only the referent changes.

If a model produces the same action under noun-swap, it is doing
vision-driven imitation rather than language-conditioned action selection.
"""

from __future__ import annotations

from typing import Iterable, Optional

from policylens.core.types import PerturbedScene, Scene
from policylens.perturb.base import Perturber
from policylens.perturb.instruction._text_utils import (
    make_perturbed_scene,
    replace_word,
)
from policylens.taxonomy.object_categories import (
    OBJECT_CATEGORIES,
    category_for_word,
)


def _pick_target_noun(instruction: str) -> tuple[Optional[str], Optional[str]]:
    """Heuristic: highest-priority object word in the instruction.

    Priority: utensil > food > toy > cloth > container > tool — manipulated
    objects are usually the first four; containers/tools are typically the
    container or destination, not the target.
    """
    priority = ["utensil", "food", "toy", "cloth", "tool", "container"]
    words = instruction.lower().split()
    for cat in priority:
        for w in words:
            stripped = w.strip(".,!?;:")
            if stripped in OBJECT_CATEGORIES.get(cat, []):
                return stripped, cat
    return None, None


class NounSwapPerturber(Perturber):
    """Swap the target noun for `max_swaps` alternatives in the same category."""

    name = "noun_swap"
    axis = "language.noun_swap"
    domain = "instruction"

    def __init__(self, max_swaps: int = 3):
        self.max_swaps = max_swaps

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        target, cat = _pick_target_noun(scene.instruction)
        if target is None or cat is None:
            return  # not applicable to this instruction
        candidates = [w for w in OBJECT_CATEGORIES[cat] if w != target]
        for i, swap in enumerate(candidates[: self.max_swaps]):
            new_instr = replace_word(scene.instruction, target, swap)
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"{target}_to_{swap}",
                new_instruction=new_instr,
                description=f"{target} → {swap}",
                parameters={"target": target, "swap_to": swap, "category": cat},
            )
