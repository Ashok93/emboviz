"""Vendored NVIDIA cosmos-framework action encoding (numpy/scipy only).

These modules are byte-faithful copies of the relevant functions from
``cosmos_framework/data/vfm/action`` (OpenMDW-1.1), with the torch in/out wrapping
removed so they depend on numpy + scipy alone — never torch or the full Cosmos
stack. They reproduce Cosmos's action encoding exactly so emboviz feeds the model
actions in the same representation it was trained on. Do not modify the algorithm.
"""

from emboviz_cosmos3._cosmos_action.action_normalization import (
    load_action_stats,
    normalize_action,
)
from emboviz_cosmos3._cosmos_action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
)

__all__ = [
    "build_abs_pose_from_components",
    "convert_rotation",
    "load_action_stats",
    "normalize_action",
    "pose_abs_to_rel",
]
