"""The VLAModel protocol.

Every adapter must implement this interface. Diagnostics only call methods
declared here — they never import a specific adapter — so swapping VLAs
is a one-line change in the CLI.

Capabilities are FLAGS. An adapter declares what it supports; diagnostics
check `Capability.X in model.capabilities` and gracefully skip if absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Optional

import numpy as np

from policylens.core.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    ImageLike,
    TokenSelector,
)


class NotSupported(RuntimeError):
    """Raised when a diagnostic requests an operation the adapter doesn't support.

    Diagnostics should ideally check `required_capabilities` against
    `model.capabilities` first and skip cleanly; if not, they will raise this.
    """


class Capability(Flag):
    """What an adapter can do. Adapters declare these in __init__."""

    INFERENCE = auto()             # predict()
    PROBABILITY_OUTPUT = auto()    # ActionResult.action_distribution populated
    ATTENTION = auto()             # extract_attention()
    HIDDEN_STATES = auto()         # extract_hidden_states()
    FFN_ACTIVATIONS = auto()       # extract_ffn_activations()
    FFN_VALUE_VECTORS = auto()     # get_ffn_value_vector_norms()
    VOCAB_LOGIT_LENS = auto()      # project_to_vocab() — needs a discrete vocab
    NEURON_ABLATION = auto()       # predict_with_neuron_ablation()
    IMAGE_PERTURBATION = auto()    # predict_with_image() (always derivable from predict())
    GRADIENT = auto()              # backprop possible (for captum-style attribution)
    BATCH_INFERENCE = auto()
    CHUNK_PREDICTION = auto()      # predict_chunk() — multi-step action chunks
    ACTIVATION_PATCHING = auto()   # predict_with_layer_patch() — for causal mediation


class VLAModel(ABC):
    """The interface every VLA adapter implements.

    Adapters are responsible for:
      • Loading their model checkpoint.
      • Handling their own image preprocessing / prompt templating.
      • Translating their native action representation into a continuous numpy
        vector for cross-model comparison.
    """

    # ----- identification ---------------------------------------------------

    @property
    @abstractmethod
    def model_id(self) -> str:
        """A short stable identifier — e.g. 'openvla-7b', 'pi0', 'gr00t-n1'."""

    @property
    @abstractmethod
    def capabilities(self) -> Capability:
        """OR'd flags describing what this adapter implements."""

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    def action_scale(self) -> Optional[np.ndarray]:
        """Per-dim scale used to normalize action-space distances.

        e.g., for OpenVLA on bridge_orig this is (q99 − q01) from the unnorm
        stats. Used by `normalized_l2`. Adapters may return None.
        """
        return None

    @property
    def num_layers(self) -> Optional[int]:
        return None

    @property
    def num_heads(self) -> Optional[int]:
        return None

    @property
    def hidden_dim(self) -> Optional[int]:
        return None

    # ----- core inference --------------------------------------------------

    @abstractmethod
    def predict(self, image: ImageLike, instruction: str) -> ActionResult:
        """Produce an action for one (image, instruction) pair.

        The returned ActionResult.action is always a continuous numpy vector
        of length `self.action_dim`, even if the model internally produces
        discrete action tokens or flow-matching trajectories — adapters
        decode to that common space.
        """

    def predict_with_image(
        self, perturbed_image: ImageLike, instruction: str
    ) -> ActionResult:
        """Convenience identical to `predict` with a different image.

        Override only if there's a cheaper path (e.g., shared text encoding).
        Default uses `predict`.
        """
        return self.predict(perturbed_image, instruction)

    # ----- internal inspection (capability-gated) -------------------------

    def extract_attention(
        self, image: ImageLike, instruction: str, query: TokenSelector,
    ) -> AttentionMaps:
        raise NotSupported(f"{self.model_id} does not support attention extraction.")

    def extract_hidden_states(
        self, image: ImageLike, instruction: str, layer_indices: list[int],
        query: TokenSelector,
    ) -> HiddenStates:
        raise NotSupported(f"{self.model_id} does not support hidden-state extraction.")

    def extract_ffn_activations(
        self, image: ImageLike, instruction: str, layer_indices: list[int],
        query: TokenSelector,
    ) -> FFNActivations:
        raise NotSupported(f"{self.model_id} does not support FFN-activation extraction.")

    # ----- vocabulary projection (capability-gated) -----------------------

    def get_ffn_value_vector_norms(self, layer_indices: list[int]) -> dict[int, np.ndarray]:
        """Per-layer (intermediate_dim,) arrays of L2 norms of FFN down_proj columns.

        Used by concept-decomposition diagnostics to weight per-neuron
        activations by the magnitude of their contribution to the residual.
        """
        raise NotSupported(f"{self.model_id} does not expose FFN value vectors.")

    def project_to_vocab(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Project a residual-stream vector onto the vocabulary embedding.

        Returns top-k (token_string, score) pairs — the 'logit lens' label
        for the vector. Capability: VOCAB_LOGIT_LENS.
        """
        raise NotSupported(f"{self.model_id} does not support vocab logit lens.")

    # ----- interventions (capability-gated) -------------------------------

    def predict_with_neuron_ablation(
        self, image: ImageLike, instruction: str,
        ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        """Run inference with specific (layer_idx, neuron_idx) activations
        forced to a scalar value before the FFN's down_proj."""
        raise NotSupported(f"{self.model_id} does not support neuron ablation.")

    def predict_with_residual_patch(
        self, image: ImageLike, instruction: str,
        patches: dict[int, np.ndarray],
        patch_position: Optional[int] = None,
    ) -> ActionResult:
        """Run inference, replacing the residual-stream output of each
        named layer at `patch_position` with the supplied vector.

        `patches` maps `layer_idx → vector of shape (hidden_dim,)`. The
        vector replaces the residual output of decoder layer L at the
        chosen sequence position (default: action-prediction position).

        The standard 'activation patching' move from causal-mediation
        analysis. Capability: ACTIVATION_PATCHING.
        """
        raise NotSupported(f"{self.model_id} does not support residual patching.")

    # ----- tokenization helpers (default impls; override if model differs)

    @abstractmethod
    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        """All sub-token positions in the *prompt* that compose `word`.

        Adapters know their tokenizer and prompt template; they return
        positions in the model's *internal sequence* (after image-token
        insertion if any).
        """

    # ----- action-space distance (default L2; override per model) --------

    def compare_actions(self, a: ActionResult, b: ActionResult) -> float:
        """Default: scaled L2 if action_scale is available, else plain L2."""
        scale = self.action_scale
        if scale is not None and scale.shape[-1] == a.action.shape[-1]:
            from policylens.core.distances import normalized_l2
            return normalized_l2(a, b, scale)
        from policylens.core.distances import l2_distance
        return l2_distance(a, b)

    # ----- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Optional: release resources (GPU memory, file handles)."""
