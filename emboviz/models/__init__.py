"""Model abstraction layer.

Public surface:
    VLAModel        — the protocol every adapter implements
    Capability      — flags for what an adapter supports
    RequiredInputs  — what a model needs from a Scene
    NotSupported    — raised when a diagnostic requests an unsupported op
    REGISTRY        — name → adapter factory

Adapters live in this package as one file each. Day-one ship list:

    - mock          — deterministic, GPU-free; supports state/gripper/history-blind modes
    - openvla-7b    — OpenVLA-7B + variants (set hf_repo)
    - lerobot       — LeRobotPolicyAdapter: any policy on the LeRobot Hub
                       (ACT, Diffusion Policy, TDMPC2, VQ-BeT)

Planned: π0/π0.5, GR00T-N1, RDT-1B, Octo, OpenVLA-OFT (one file each).
"""

from emboviz.models.protocol import (
    Capability,
    NotSupported,
    RequiredInputs,
    VLAModel,
)
from emboviz.models.registry import REGISTRY, register_model, get_model

__all__ = [
    "Capability",
    "NotSupported",
    "RequiredInputs",
    "VLAModel",
    "REGISTRY",
    "register_model",
    "get_model",
]
