"""The VLAModel protocol.

Every adapter implements this interface. Diagnostics only call methods
declared here — they never import a specific adapter — so swapping
VLAs is a one-line change in the CLI.

Two dimensions of declarative metadata travel on every adapter:

- `capabilities`: what the model can EXPOSE (attention, hidden states,
  patching, etc.). Diagnostics that need a capability check before
  running and emit a clean "skipped, reason: X" result if absent.

- `required_inputs`: what the model needs to CONSUME from a Scene
  (which cameras, whether it uses state/gripper/action_history). The
  runner validates a Scene satisfies these before predict() runs;
  perturbers use them (via Perturber.affects) to auto-skip mutations
  against modalities the model doesn't read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, Optional

import numpy as np

from emboviz_wire.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    Scene,
    TokenSelector,
    average_action_results,
)

if TYPE_CHECKING:
    pass


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
    GRADIENT = auto()              # backprop possible (for captum-style attribution)
    BATCH_INFERENCE = auto()
    CHUNK_PREDICTION = auto()      # reserved for a dedicated predict_chunk() API.
                                   # No diagnostic gates on this flag: chunk
                                   # diagnostics key off ActionResult.action_chunk
                                   # being populated (which OFT / pi0 / GR00T do
                                   # from predict()), so they need not declare it.
    ACTIVATION_PATCHING = auto()   # predict_with_residual_patch() — causal mediation


@dataclass(frozen=True)
class RequiredInputs:
    """What a model needs from a Scene to make a prediction.

    Adapters declare this once; the framework validates Scenes against
    it before calling predict (loud error at the boundary instead of a
    silent wrong inference). Perturbers cross-check their `affects`
    against this to auto-skip irrelevant perturbations.
    """

    cameras: frozenset[str] = frozenset({"primary"})
    instruction: bool = True
    state: bool = False
    gripper: bool = False
    action_history: bool = False
    depth: bool = False
    force_torque: bool = False
    tactile: bool = False
    extras: frozenset[str] = frozenset()

    def validate(self, scene: Scene) -> Optional[str]:
        """Returns None if `scene` satisfies the requirements, else a
        human-readable reason."""
        obs = scene.observations
        for cam in self.cameras:
            if cam not in obs.images:
                return f"missing camera '{cam}' in scene.observations.images"
        if self.instruction and not scene.instruction:
            return "model requires instruction but scene.instruction is empty"
        if self.state and obs.state is None:
            return "model requires proprioceptive state but scene.observations.state is None"
        if self.gripper and obs.gripper is None:
            return "model requires gripper but scene.observations.gripper is None"
        if self.action_history and obs.action_history is None:
            return "model requires action_history but scene.observations.action_history is None"
        if self.depth and not obs.depth:
            return "model requires depth but scene.observations.depth is None/empty"
        if self.force_torque and obs.force_torque is None:
            return "model requires force_torque but scene.observations.force_torque is None"
        if self.tactile and obs.tactile is None:
            return "model requires tactile but scene.observations.tactile is None"
        for key in self.extras:
            if key not in obs.extras:
                return f"model requires extras['{key}'] but it is missing"
        return None

    def consumes(self, affect: str) -> bool:
        """Does this model consume the input modality named by `affect`?

        Affect strings follow the perturber convention:
          - "instruction"
          - "images.<camera_id>"   (e.g. "images.primary", "images.wrist_left")
          - "state", "gripper", "action_history", "depth", "force_torque", "tactile"
          - "extras.<key>"
        """
        if affect == "instruction":
            return self.instruction
        if affect == "images.*":
            return bool(self.cameras)
        if affect.startswith("images."):
            return affect.split(".", 1)[1] in self.cameras
        if affect == "state":
            return self.state
        if affect == "gripper":
            return self.gripper
        if affect == "action_history":
            return self.action_history
        if affect == "depth":
            return self.depth
        if affect == "force_torque":
            return self.force_torque
        if affect == "tactile":
            return self.tactile
        if affect.startswith("extras."):
            return affect.split(".", 1)[1] in self.extras
        return False


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
    def required_inputs(self) -> RequiredInputs:
        """What the adapter needs from a Scene to make a prediction."""

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
    def predict(self, scene: Scene) -> ActionResult:
        """Produce an action for one Scene.

        Adapters read the fields they declared in `required_inputs`. The
        returned ActionResult.action is always a continuous numpy vector
        of length `self.action_dim`, even if the model internally produces
        discrete action tokens or flow-matching trajectories.
        """

    def predict_batch(
        self, scenes: list[Scene], n_samples: int = 1,
    ) -> list[ActionResult]:
        """Predict actions for many scenes at once — the parallel hot path.

        Returns one ``ActionResult`` per input scene, **in input order**,
        each averaged over ``n_samples`` independent samples (for stochastic
        policies). Semantically identical to averaging ``n_samples`` calls of
        :meth:`predict` per scene — batching changes throughput only, never
        the numbers. The diagnostics that fan out hundreds of perturbed
        scenes (sensitivity grid, modality dropout, memorization fills) call
        this so the GPU runs one batched forward instead of N round-trips.

        This default loops :meth:`predict` so every adapter works
        unmodified. An adapter that runs a TRUE batched GPU forward
        overrides this and advertises ``Capability.BATCH_INFERENCE``. The
        override **owns its batch-size policy** (memory-aware chunking +
        OOM-backoff): callers submit the full list and never pass a batch
        size — that keeps the same call site correct on a 12 GB laptop and
        an 80 GB datacenter GPU.
        """
        if n_samples < 1:
            raise ValueError(
                f"predict_batch: n_samples must be >= 1, got {n_samples}"
            )
        out: list[ActionResult] = []
        for scene in scenes:
            out.append(average_action_results(
                [self.predict(scene) for _ in range(n_samples)]
            ))
        return out

    # ----- internal inspection (capability-gated) -------------------------

    def extract_attention(self, scene: Scene, query: TokenSelector) -> AttentionMaps:
        raise NotSupported(f"{self.model_id} does not support attention extraction.")

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        raise NotSupported(f"{self.model_id} does not support hidden-state extraction.")

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
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
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        """Run inference with specific (layer_idx, neuron_idx) activations
        forced to a scalar value before the FFN's down_proj."""
        raise NotSupported(f"{self.model_id} does not support neuron ablation.")

    def predict_with_residual_patch(
        self, scene: Scene, patches: dict[int, np.ndarray],
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
            from emboviz_wire.distances import normalized_l2
            return normalized_l2(a, b, scale)
        from emboviz_wire.distances import l2_distance
        return l2_distance(a, b)

    # ----- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Optional: release resources (GPU memory, file handles)."""
