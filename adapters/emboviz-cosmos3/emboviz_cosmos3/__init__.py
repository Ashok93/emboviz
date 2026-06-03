"""emboviz-cosmos3 — NVIDIA Cosmos3-Nano world-model adapter.

Thin shim for the user's main venv: it advertises an :class:`AdapterSpec`
via the ``emboviz.world_models`` entry point and exposes the
:class:`Cosmos3WorldModel` adapter. The heavy lifting (the BF16 model) lives
in a separate vLLM-Omni server; this adapter is a pure HTTP client.
"""

from __future__ import annotations

from emboviz_cosmos3.model import Cosmos3WorldModel
from emboviz_cosmos3.spec import SPEC

__all__ = ["Cosmos3WorldModel", "SPEC"]
