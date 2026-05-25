"""Perturber protocol.

A Perturber is a stateless object with a `variants(scene)` method that
yields one or more PerturbedScenes. The Diagnostic layer composes
Perturbers with Metrics + Runners — Perturbers themselves know nothing
about VLAs, metrics, or scoring.

Every Perturber declares `affects`: a frozenset of strings naming the
Scene input modalities it mutates. The vocabulary mirrors
`RequiredInputs.consumes()`:

  - "instruction"
  - "images.<camera_id>"   (e.g. "images.primary", "images.wrist_left")
  - "state", "gripper", "action_history", "depth", "force_torque", "tactile"
  - "extras.<key>"

Diagnostics cross-check `affects` against the model's
`required_inputs` to auto-skip perturbations that mutate inputs the
model doesn't consume (Severity.UNKNOWN with a clear reason).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from emboviz.core.types import PerturbedScene, Scene


class Perturber(ABC):
    """One axis of perturbation. Stateless. Reusable across scenes."""

    name: str
    axis: str
    affects: frozenset[str]

    @abstractmethod
    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        """Yield one or more PerturbedScenes derived from `scene`.

        A Perturber may yield 0 variants if the scene isn't applicable
        (e.g., NounSwap when the instruction has no recognized noun).
        Diagnostics treat 0-variant outcomes as "not applicable, skip."
        """


class NullPerturber(Perturber):
    """Yields the scene unchanged — useful as a baseline placeholder."""

    name = "null"
    axis = "baseline"
    affects = frozenset()

    def variants(self, scene: Scene):
        yield PerturbedScene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="baseline",
            description="(baseline; no change)",
        )
