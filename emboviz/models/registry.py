"""Legacy in-process adapter registry — `mock` and `lerobot` only.

The four production VLA families (OpenVLA, OFT, π0, GR00T) have moved
to standalone packages under ``adapters/`` and are reached via the new
ZeroMQ-actor path (``--model openvla`` → ``adapter:openvla``). This
registry only retains the lightweight in-process adapters that still
make sense to run inside the user's main venv:

  • ``mock``   — deterministic test fixture, no GPU
  • ``lerobot``— delegates to a stock LeRobotDataset policy (CPU OK)

Both of those import cheaply (no torch at module level) so the
``--model mock`` and ``--model lerobot:<repo>`` codepaths continue to
work the way they did before the refactor.
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
    if name not in REGISTRY:
        _try_lazy_register(name)
    if name not in REGISTRY:
        raise KeyError(
            f"No in-process adapter registered for '{name}'. "
            f"Available: {sorted(REGISTRY)}. "
            "(VLA adapters live in separate packages now — "
            "use `--model openvla|oft|pi0|gr00t|sam3` to drive them "
            "via their ZMQ workers.)"
        )
    return REGISTRY[name]


def _try_lazy_register(name: str) -> None:
    """Trigger the in-process adapter module's side-effect import.

    Only the two remaining in-process adapters live here. Every other
    model name resolves through :mod:`emboviz.adapters.registry`
    instead.
    """
    builtin = {
        "mock":    "emboviz.models.mock",
        "lerobot": "emboviz.models.lerobot_policy",
    }
    module = builtin.get(name)
    if module:
        import importlib
        importlib.import_module(module)
