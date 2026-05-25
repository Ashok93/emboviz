"""Adapter registry — string name to adapter factory.

CLI does `get_model("openvla-7b")(**kwargs)` to instantiate any adapter
without importing it directly. Adapters self-register at import time.
"""

from __future__ import annotations

from typing import Callable

from emboviz.models.protocol import VLAModel


REGISTRY: dict[str, Callable[..., VLAModel]] = {}


def register_model(name: str):
    """Decorator: register the decorated class/factory under `name`."""

    def _wrap(factory):
        if name in REGISTRY:
            raise ValueError(f"Model adapter '{name}' already registered")
        REGISTRY[name] = factory
        return factory

    return _wrap


def get_model(name: str) -> Callable[..., VLAModel]:
    """Look up an adapter factory. Imports the adapter module lazily."""
    # Eagerly try registering known built-ins on demand. This keeps the
    # `emboviz` import cheap (no torch at top level) while still letting
    # `get_model("openvla-7b")` work from anywhere.
    if name not in REGISTRY:
        _try_lazy_register(name)
    if name not in REGISTRY:
        raise KeyError(f"No adapter registered for '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]


def _try_lazy_register(name: str) -> None:
    """Trigger the adapter module's side-effect import for known names."""
    builtin = {
        "openvla-7b": "emboviz.models.openvla",
        "openvla": "emboviz.models.openvla",
        "mock": "emboviz.models.mock",
        "lerobot": "emboviz.models.lerobot_policy",
        "gr00t": "emboviz.models.gr00t",
        "gr00t-n1": "emboviz.models.gr00t",
        "openvla-oft": "emboviz.models.openvla_oft",
        # "pi0": "emboviz.models.pi0",      # future — needs custom adapter (lerobot version conflicts)
    }
    module = builtin.get(name)
    if module:
        import importlib
        importlib.import_module(module)
