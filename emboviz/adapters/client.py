"""RayVLAClient — a VLAModel facade backed by a Ray actor.

Diagnostics consume :class:`emboviz.models.protocol.VLAModel`. This
class implements that ABC by forwarding every call to a Ray actor
running in the adapter's isolated runtime venv.

Why every method blocks on ``ray.get``: emboviz runs ONE actor per
model and serializes diagnostic calls (memorization, attention,
modality dropout, etc.) — they share GPU memory anyway. Concurrency
across diagnostics buys nothing and risks racing on the same KV
cache. The cost we pay is one inter-process round-trip per call;
that's microseconds compared to a VLA forward pass.

What Ray does for us at the boundary:

  • Numpy arrays travel through Plasma (zero-copy when both ends are
    on the same node, which is our default).
  • PIL images and dataclasses round-trip via pickle protocol 5.
  • Errors raised inside the actor are re-raised on the driver with
    a clean RayTaskError chain.

Cached actor properties (``model_id``, ``capabilities``,
``required_inputs``, ``action_dim``, etc.) are fetched ONCE on first
access and stored locally — they don't change during a session and
fetching them every call would waste a round-trip.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz.adapters.lifecycle import ActorHandle
from emboviz.core.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    Scene,
    TokenSelector,
)
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel


class RayVLAClient(VLAModel):
    """Wraps an :class:`ActorHandle` in the VLAModel interface.

    The runner builds one of these per ``emboviz analyze`` invocation
    and hands it to diagnostics. They see a plain VLAModel.

    Important: the *types* exchanged (Scene, ActionResult, etc.) are
    defined in ``emboviz.core.types``. The adapter's runtime venv
    therefore depends on the ``emboviz`` core package — its types
    module is import-light (numpy only) so this costs nothing.
    """

    def __init__(self, handle: ActorHandle):
        self._handle = handle
        self._actor = handle.actor
        # Lazy-fetched once and frozen:
        self._cached_model_id: Optional[str] = None
        self._cached_capabilities: Optional[Capability] = None
        self._cached_required_inputs: Optional[RequiredInputs] = None
        self._cached_action_dim: Optional[int] = None
        self._cached_action_scale: Optional[np.ndarray] = None
        self._cached_num_layers: Optional[int] = None
        self._cached_num_heads: Optional[int] = None
        self._cached_hidden_dim: Optional[int] = None
        self._cached_action_scale_loaded: bool = False
        self._cached_dims_loaded: bool = False
        self._closed = False

    # ----- helpers ----------------------------------------------------

    def _call(self, method_name: str, *args, **kwargs):
        """ray.get a remote method, surfacing actor exceptions plainly."""
        import ray

        remote_method = getattr(self._actor, method_name)
        return ray.get(remote_method.remote(*args, **kwargs))

    def _load_static_metadata(self) -> None:
        """One-shot fetch of all const properties — saves N-1 RTTs."""
        meta = self._call("static_metadata")
        self._cached_model_id = meta["model_id"]
        self._cached_capabilities = Capability(meta["capabilities"])
        self._cached_required_inputs = meta["required_inputs"]
        self._cached_action_dim = int(meta["action_dim"])
        self._cached_action_scale = meta.get("action_scale")
        self._cached_action_scale_loaded = True
        self._cached_num_layers = meta.get("num_layers")
        self._cached_num_heads = meta.get("num_heads")
        self._cached_hidden_dim = meta.get("hidden_dim")
        self._cached_dims_loaded = True

    # ----- identification ---------------------------------------------

    @property
    def model_id(self) -> str:
        if self._cached_model_id is None:
            self._load_static_metadata()
        return self._cached_model_id  # type: ignore[return-value]

    @property
    def capabilities(self) -> Capability:
        if self._cached_capabilities is None:
            self._load_static_metadata()
        return self._cached_capabilities  # type: ignore[return-value]

    @property
    def required_inputs(self) -> RequiredInputs:
        if self._cached_required_inputs is None:
            self._load_static_metadata()
        return self._cached_required_inputs  # type: ignore[return-value]

    @property
    def action_dim(self) -> int:
        if self._cached_action_dim is None:
            self._load_static_metadata()
        return self._cached_action_dim  # type: ignore[return-value]

    @property
    def action_scale(self) -> Optional[np.ndarray]:
        if not self._cached_action_scale_loaded:
            self._load_static_metadata()
        return self._cached_action_scale

    @property
    def num_layers(self) -> Optional[int]:
        if not self._cached_dims_loaded:
            self._load_static_metadata()
        return self._cached_num_layers

    @property
    def num_heads(self) -> Optional[int]:
        if not self._cached_dims_loaded:
            self._load_static_metadata()
        return self._cached_num_heads

    @property
    def hidden_dim(self) -> Optional[int]:
        if not self._cached_dims_loaded:
            self._load_static_metadata()
        return self._cached_hidden_dim

    # ----- core inference --------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        return self._call("predict", scene)

    # ----- internal inspection ---------------------------------------

    def extract_attention(self, scene: Scene, query: TokenSelector) -> AttentionMaps:
        return self._call("extract_attention", scene, query)

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        return self._call("extract_hidden_states", scene, layer_indices, query)

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> FFNActivations:
        return self._call("extract_ffn_activations", scene, layer_indices, query)

    # ----- vocabulary projection -------------------------------------

    def get_ffn_value_vector_norms(self, layer_indices: list[int]) -> dict[int, np.ndarray]:
        return self._call("get_ffn_value_vector_norms", layer_indices)

    def project_to_vocab(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        return self._call("project_to_vocab", vector, top_k)

    # ----- interventions ---------------------------------------------

    def predict_with_neuron_ablation(
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        return self._call("predict_with_neuron_ablation", scene, ablations)

    def predict_with_residual_patch(
        self, scene: Scene, patches: dict[int, np.ndarray],
        patch_position: Optional[int] = None,
    ) -> ActionResult:
        return self._call(
            "predict_with_residual_patch", scene, patches, patch_position,
        )

    # ----- tokenization helpers --------------------------------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return self._call("find_token_positions", instruction, word)

    # ----- lifecycle -------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._call("close")
        except Exception:
            # Closing is best-effort — the actor may already be gone.
            pass
        from emboviz.adapters.lifecycle import shutdown
        shutdown(self._handle)
        self._closed = True

    def __enter__(self) -> "RayVLAClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
