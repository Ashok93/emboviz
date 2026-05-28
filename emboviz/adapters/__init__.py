"""The adapter subsystem — every VLA backend talks through Ray actors.

emboviz core has **no model dependencies** (no torch, no transformers,
no lerobot). Each VLA family lives in its own pip-installable adapter
package (``emboviz-openvla``, ``emboviz-oft``, ``emboviz-pi0``,
``emboviz-gr00t``, ``emboviz-sam3``) whose heavy deps install into an
isolated venv on first use.

When a diagnostic asks for ``model.predict(scene)``:

1. The CLI consults :func:`emboviz.adapters.registry.find_adapter` to
   resolve a CLI alias (``"openvla"``) → :class:`AdapterSpec` declared
   by the matching adapter package's entry point.
2. :func:`emboviz.adapters.lifecycle.connect` spawns or attaches to a
   Ray actor running the adapter's actor class in its isolated venv
   (``runtime_env={"py_executable": "/path/to/venv/bin/python"}``).
3. :class:`emboviz.adapters.client.RayVLAClient` wraps the Ray handle
   in a :class:`emboviz.models.protocol.VLAModel`-compatible facade so
   diagnostics never know they're talking to another process.

This is the only architecture in which we can offer
``emboviz-openvla`` and ``emboviz-oft`` together — their pinned
transformers / lerobot versions are mutually incompatible at the venv
level. Each adapter's runtime venv is walled off.
"""

from emboviz.adapters.protocol import AdapterSpec
from emboviz.adapters.registry import find_adapter, list_adapters
from emboviz.adapters.lifecycle import connect, shutdown, install_venv
from emboviz.adapters.client import RayVLAClient
from emboviz.adapters.actor_base import BaseAdapterActor

__all__ = [
    "AdapterSpec",
    "BaseAdapterActor",
    "RayVLAClient",
    "find_adapter",
    "list_adapters",
    "connect",
    "shutdown",
    "install_venv",
]
