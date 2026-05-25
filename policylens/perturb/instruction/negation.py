"""Negation injection — tests whether the model honours 'don't'."""

from __future__ import annotations

from typing import Iterable

from policylens.core.types import PerturbedScene, Scene
from policylens.perturb.base import Perturber
from policylens.perturb.instruction._text_utils import make_perturbed_scene


class NegationPerturber(Perturber):
    """Prefix instruction with a negation ('do not', 'never'). Behaviourally
    the model should change action (refuse / pick something else); if it
    doesn't, it's blind to negation."""

    name = "negation"
    axis = "language.negation"
    domain = "instruction"

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        if not scene.instruction.strip():
            return
        for prefix, vid in (("do not ", "do_not"), ("never ", "never")):
            new_instr = prefix + scene.instruction
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=vid,
                new_instruction=new_instr,
                description=f"'{prefix.strip()}' prefix",
                parameters={"prefix": prefix.strip()},
            )
