"""Base class adapter packages subclass for their Ray actor.

Each adapter package defines (in its own venv) a small actor class::

    from emboviz.adapters.actor_base import BaseAdapterActor
    from emboviz_openvla.model import OpenVLA

    class OpenVLAActor(BaseAdapterActor):
        def _build_model(self, **kwargs):
            return OpenVLA(**kwargs)

The base class implements the wire methods :class:`RayVLAClient`
expects (``static_metadata``, ``predict``, ``extract_attention``,
etc.) by delegating to the underlying :class:`VLAModel`. Each method
is small and exists ONLY so the Ray ``@ray.remote`` wrapper picks it
up — the heavy lifting lives in the wrapped model.

Why not just decorate the VLAModel itself with ``@ray.remote``? Two
reasons:

  1. The model's ``__init__`` does the heavy load. We want lifecycle
     to be visible (Ray prints "creating actor …") and to keep
     model-construction kwargs separable from the actor-creation
     plumbing.
  2. Cached static metadata (the bundle returned by
     ``static_metadata``) is computed inside the actor, so the
     RayVLAClient on the driver side incurs ONE round-trip for every
     constant property rather than N.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz.core.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    Scene,
    TokenSelector,
)
from emboviz.models.protocol import VLAModel


class BaseAdapterActor:
    """Ray actor wrapping one :class:`VLAModel`.

    Subclasses MUST override :meth:`_build_model` to load and return
    the concrete VLAModel. Everything else is implemented here.
    """

    def __init__(self, **model_kwargs: Any):
        self._model: VLAModel = self._build_model(**model_kwargs)

    # ----- subclass hook ---------------------------------------------

    def _build_model(self, **kwargs: Any) -> VLAModel:
        raise NotImplementedError(
            "Subclass must override _build_model() to load the wrapped "
            "VLAModel. See emboviz_openvla.actor.OpenVLAActor for a "
            "minimal example."
        )

    # ----- single-shot static-metadata bundle ------------------------

    def static_metadata(self) -> dict[str, Any]:
        """Return all immutable model properties in one round-trip.

        :class:`RayVLAClient` calls this on first access of any const
        property and caches the result. Saves N-1 inter-process round
        trips on every CLI invocation.
        """
        m = self._model
        return {
            "model_id":        m.model_id,
            # ``Capability`` is a Flag enum; transport as int so the
            # driver-side reconstructs without importing the same enum
            # class (it does, but transport via int sidesteps any
            # interpreter-version pickle subtleties).
            "capabilities":    int(m.capabilities.value),
            "required_inputs": m.required_inputs,
            "action_dim":      int(m.action_dim),
            "action_scale":    m.action_scale,
            "num_layers":      m.num_layers,
            "num_heads":       m.num_heads,
            "hidden_dim":      m.hidden_dim,
        }

    # ----- inference -------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        return self._model.predict(scene)

    # ----- internal inspection ---------------------------------------

    def extract_attention(self, scene: Scene, query: TokenSelector) -> AttentionMaps:
        return self._model.extract_attention(scene, query)

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        return self._model.extract_hidden_states(scene, layer_indices, query)

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> FFNActivations:
        return self._model.extract_ffn_activations(scene, layer_indices, query)

    # ----- vocabulary projection -------------------------------------

    def get_ffn_value_vector_norms(self, layer_indices: list[int]) -> dict[int, np.ndarray]:
        return self._model.get_ffn_value_vector_norms(layer_indices)

    def project_to_vocab(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        return self._model.project_to_vocab(vector, top_k)

    # ----- interventions ---------------------------------------------

    def predict_with_neuron_ablation(
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        return self._model.predict_with_neuron_ablation(scene, ablations)

    def predict_with_residual_patch(
        self, scene: Scene, patches: dict[int, np.ndarray],
        patch_position: Optional[int] = None,
    ) -> ActionResult:
        return self._model.predict_with_residual_patch(
            scene, patches, patch_position,
        )

    # ----- tokenization helpers --------------------------------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return self._model.find_token_positions(instruction, word)

    # ----- lifecycle -------------------------------------------------

    def close(self) -> None:
        try:
            self._model.close()
        finally:
            self._model = None  # type: ignore[assignment]
