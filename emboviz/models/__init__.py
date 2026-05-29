"""Model abstraction layer (host side).

Public surface:
    VLAModel        — the protocol every adapter implements
    Capability      — flags for what an adapter supports
    RequiredInputs  — what a model needs from a Scene
    NotSupported    — raised when a diagnostic requests an unsupported op
    REGISTRY        — name → in-process adapter factory

This package holds only the LIGHT, in-process adapter that runs in the
host venv:

    - mock     — deterministic, GPU-free; state/gripper/history-blind modes

The VLA families (OpenVLA, OpenVLA-OFT, π0/π0.5, GR00T-N1.7) live in their
own ``emboviz-<name>`` adapter packages and run as isolated ZMQ workers —
reached through ``emboviz.adapters`` (registry + lifecycle), not this
in-process registry. (A LeRobot-policy adapter — ACT / Diffusion Policy /
etc. — is planned as a future isolated worker, not an in-process model.)
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
