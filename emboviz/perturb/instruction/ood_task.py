"""OOD-task perturber — the upper-bound reference.

Replaces the instruction with one for an entirely different task that
*should* produce a very different action. Provides the ceiling against
which other ISS values are normalised.
"""

from __future__ import annotations

from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.instruction._text_utils import make_perturbed_scene


DEFAULT_OOD_TASKS = [
    "press the red button",
    "open the drawer",
    "wave hello",
]


class OODTaskPerturber(Perturber):
    name = "ood_task"
    axis = "language.ood_task"
    domain = "instruction"

    def __init__(self, ood_tasks: list[str] | None = None):
        self.ood_tasks = ood_tasks or DEFAULT_OOD_TASKS

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        for i, task in enumerate(self.ood_tasks):
            if task == scene.instruction:
                continue
            yield make_perturbed_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"ood_{i}",
                new_instruction=task,
                description=f"OOD: {task}",
                parameters={"task": task},
            )
            break  # one OOD reference is enough
