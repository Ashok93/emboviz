"""Model abstraction layer.

Public surface:
    VLAModel        — the protocol every adapter implements
    Capability      — flags for what an adapter supports
    NotSupported    — raised when a diagnostic requests an unsupported op
    REGISTRY        — name → adapter factory

Adapters live in this package as one file each (openvla.py, pi0.py, ...).
"""

from emboviz.models.protocol import (
    Capability,
    NotSupported,
    VLAModel,
)
from emboviz.models.registry import REGISTRY, register_model, get_model

__all__ = [
    "Capability",
    "NotSupported",
    "VLAModel",
    "REGISTRY",
    "register_model",
    "get_model",
]
