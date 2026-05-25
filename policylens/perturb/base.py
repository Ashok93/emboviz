"""Perturber protocol.

A Perturber is a stateless object with a `variants(scene)` method that
yields one or more PerturbedScenes. The Diagnostic layer composes
Perturbers with Metrics + Runners — Perturbers themselves know nothing
about VLAs, metrics, or scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Literal

from policylens.core.types import PerturbedScene, Scene


PerturbationDomain = Literal["instruction", "image", "joint"]


class Perturber(ABC):
    """One axis of perturbation. Stateless. Reusable across scenes."""

    name: str
    axis: str
    domain: PerturbationDomain

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
    domain: PerturbationDomain = "joint"

    def variants(self, scene: Scene):
        yield PerturbedScene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="baseline",
            description="(baseline; no change)",
        )
