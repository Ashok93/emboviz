"""Image perturbers — change pixels, leave instruction alone.

Each perturber tests a specific vision-axis of robustness. Light-weight
perturbers (no extra models) live alongside heavy ones (SAM+IP2P recolor).
"""

from emboviz.perturb.image.distractor import DistractorInjectionPerturber
from emboviz.perturb.image.lighting import LightingShiftPerturber
from emboviz.perturb.image.noise import GaussianNoisePerturber
from emboviz.perturb.image.occlusion import OcclusionPerturber
from emboviz.perturb.image.recolor import ObjectRecolorPerturber
from emboviz.perturb.image.target_remove import TargetRemovalPerturber
from emboviz.perturb.image.viewpoint import ViewpointJitterPerturber

__all__ = [
    "DistractorInjectionPerturber",
    "LightingShiftPerturber",
    "GaussianNoisePerturber",
    "ObjectRecolorPerturber",
    "OcclusionPerturber",
    "TargetRemovalPerturber",
    "ViewpointJitterPerturber",
]
