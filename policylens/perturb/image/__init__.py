"""Image perturbers — change pixels, leave instruction alone.

Each perturber tests a specific vision-axis of robustness. Light-weight
perturbers (no extra models) live alongside heavy ones (SAM+IP2P recolor).
"""

from policylens.perturb.image.distractor import DistractorInjectionPerturber
from policylens.perturb.image.lighting import LightingShiftPerturber
from policylens.perturb.image.noise import GaussianNoisePerturber
from policylens.perturb.image.occlusion import OcclusionPerturber
from policylens.perturb.image.recolor import ObjectRecolorPerturber
from policylens.perturb.image.target_remove import TargetRemovalPerturber
from policylens.perturb.image.viewpoint import ViewpointJitterPerturber

__all__ = [
    "DistractorInjectionPerturber",
    "LightingShiftPerturber",
    "GaussianNoisePerturber",
    "ObjectRecolorPerturber",
    "OcclusionPerturber",
    "TargetRemovalPerturber",
    "ViewpointJitterPerturber",
]
