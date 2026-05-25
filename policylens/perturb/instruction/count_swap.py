"""Count / ordinal swap — tests numeric grounding."""

from __future__ import annotations

from typing import Iterable

from policylens.core.types import PerturbedScene, Scene
from policylens.perturb.base import Perturber
from policylens.perturb.instruction._text_utils import (
    make_perturbed_scene,
    replace_word,
)
from policylens.taxonomy.object_categories import COUNT_WORDS, ORDINAL_WORDS


class CountSwapPerturber(Perturber):
    """Swap count or ordinal in the instruction (one → two, first → last)."""

    name = "count_swap"
    axis = "language.count_swap"
    domain = "instruction"

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        for vocab, label in ((COUNT_WORDS, "count"), (ORDINAL_WORDS, "ordinal")):
            words_lower = [w.strip(".,!?;:") for w in scene.instruction.lower().split()]
            present = [w for w in words_lower if w in vocab]
            if not present:
                continue
            original = present[0]
            for swap in vocab:
                if swap == original:
                    continue
                new_instr = replace_word(scene.instruction, original, swap)
                yield make_perturbed_scene(
                    scene=scene,
                    perturber_name=self.name,
                    axis=self.axis,
                    variant_id=f"{label}_{original}_to_{swap}",
                    new_instruction=new_instr,
                    description=f"{original} → {swap}",
                    parameters={"from": original, "to": swap, "kind": label},
                )
                break  # one swap per kind
