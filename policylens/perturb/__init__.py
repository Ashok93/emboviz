"""Perturbations — model-agnostic transforms over Scenes.

A Perturber takes one Scene and yields N PerturbedScenes (variants).
Diagnostics consume Perturbers polymorphically — they don't know whether
they're swapping a word or recoloring an object.
"""

from policylens.perturb.base import Perturber, NullPerturber

__all__ = ["Perturber", "NullPerturber"]
