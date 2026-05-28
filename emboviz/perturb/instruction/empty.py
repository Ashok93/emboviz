"""Empty-instruction perturber — the pure-vision reference.

If the model produces a similar action under no instruction at all, it
isn't using language. Useful as a lower-bound reference for ISS scores.
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import make_perturbed_scene


class EmptyInstructionPerturber(Perturber):
    name = "empty"
    axis = "language.empty"
    affects = frozenset({"instruction"})

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        if scene.instruction == "":
            return  # nothing to compare
        yield make_perturbed_scene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="empty",
            new_instruction="",
            description="(empty instruction — pure vision)",
        )
