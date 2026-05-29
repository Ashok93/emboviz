"""A deterministic mock VLA — for unit-testing diagnostics without GPU.

The mock's behaviour is configurable so we can simulate a noun-blind model,
a properly grounded model, an over-sensitive model, or a model that ignores
state/gripper/history. This is what lets us iterate on the core algorithms
without spending GPU time, AND validate that our state-side diagnostics
correctly detect models that ignore their state input.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

import numpy as np

from emboviz.core.types import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    Scene,
    TokenSelector,
)
from emboviz.models.protocol import (
    Capability,
    RequiredInputs,
    VLAModel,
)
from emboviz.models.registry import register_model


Mode = Literal[
    "noun_blind",     # ignores instruction; depends only on image
    "grounded",       # action depends on instruction content
    "random",         # uniformly random per call (controlled by seed)
    "nuanced",        # partial grounding — uses first char of instruction only
    "state_blind",    # consumes state but ignores it (for state-perturber tests)
    "gripper_blind",  # consumes gripper but ignores it
    "history_blind",  # consumes action_history but ignores it
]


@register_model("mock")
class MockVLA(VLAModel):
    """Deterministic adapter for testing diagnostics without a GPU.

    Modes that consume state/gripper/action_history declare it in
    required_inputs even though they ignore the values — that's the
    point of the test: prove our state-side diagnostics detect a model
    that LOOKS like it uses an input but actually doesn't.
    """

    _CAPS = (
        Capability.INFERENCE
        | Capability.ATTENTION
        | Capability.HIDDEN_STATES
        | Capability.FFN_ACTIVATIONS
        | Capability.NEURON_ABLATION
    )

    def __init__(
        self,
        mode: Mode = "noun_blind",
        action_dim: int = 7,
        seed: int = 0,
        n_layers: int = 8,
        n_heads: int = 8,
        hidden_dim: int = 64,
        n_image_tokens: int = 16,
    ):
        self.mode = mode
        self._action_dim = action_dim
        self.seed = seed
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._hidden_dim = hidden_dim
        self._n_image_tokens = n_image_tokens
        self._grid_side = int(np.sqrt(n_image_tokens))
        assert self._grid_side ** 2 == n_image_tokens, "n_image_tokens must be a perfect square"

    # ----- identity -------------------------------------------------------

    @property
    def model_id(self) -> str:
        return f"mock-{self.mode}"

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # Every mode reads {primary image, instruction}. The *_blind modes
        # additionally declare consumption of the modality they're blind to,
        # so state-side diagnostics actually run against them and detect
        # the (intentional) blindness.
        return RequiredInputs(
            cameras=frozenset({"primary"}),
            instruction=True,
            state=(self.mode == "state_blind"),
            gripper=(self.mode == "gripper_blind"),
            action_history=(self.mode == "history_blind"),
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_layers(self) -> Optional[int]:
        return self._n_layers

    @property
    def num_heads(self) -> Optional[int]:
        return self._n_heads

    @property
    def hidden_dim(self) -> Optional[int]:
        return self._hidden_dim

    # ----- inference ------------------------------------------------------

    def _validated_inputs(self, scene: Scene) -> tuple:
        """Validate scene against required_inputs; return (image, instruction)."""
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"MockVLA: {reason}")
        return scene.observations.images["primary"].data, scene.instruction

    def predict(self, scene: Scene) -> ActionResult:
        image, instruction = self._validated_inputs(scene)

        rng = np.random.default_rng(self._seed_for(image, instruction))
        if self.mode == "random":
            action = rng.normal(0, 0.05, self._action_dim).astype(np.float32)
        elif self.mode in ("noun_blind", "state_blind", "gripper_blind", "history_blind"):
            # All "blind" modes depend ONLY on image hash. The variation is in
            # which input they *declare* they consume (see required_inputs).
            action = self._image_action(image)
        elif self.mode == "nuanced":
            seed_for_noun = sum(ord(c) for c in (instruction[:1] or "_"))
            sub_rng = np.random.default_rng(self.seed + seed_for_noun)
            action = (
                self._image_action(image)
                + sub_rng.normal(0, 0.1, self._action_dim).astype(np.float32)
            )
        elif self.mode == "grounded":
            instr_seed = int(hashlib.sha256(instruction.encode()).hexdigest()[:8], 16)
            sub_rng = np.random.default_rng(self.seed + instr_seed)
            action = sub_rng.normal(0, 0.3, self._action_dim).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return ActionResult(action=action, action_dim=self._action_dim)

    # ----- attention ------------------------------------------------------

    def extract_attention(self, scene: Scene, query: TokenSelector) -> AttentionMaps:
        image, instruction = self._validated_inputs(scene)
        # Fake attention whose distribution depends on the instruction so
        # JS divergence between two instructions is non-zero for grounded modes.
        rng = np.random.default_rng(self._seed_for(image, instruction) + 1)
        n_text = 10  # arbitrary text length for mock
        n_keys = 1 + self._n_image_tokens + n_text
        weights = rng.dirichlet(
            np.ones(n_keys) * (0.5 if self.mode == "grounded" else 5.0),
            size=(self._n_layers, self._n_heads),
        ).astype(np.float32)
        if self.mode in ("noun_blind", "state_blind", "gripper_blind", "history_blind"):
            base_rng = np.random.default_rng(self._image_seed(image) + 1)
            weights = base_rng.dirichlet(
                np.ones(n_keys) * 5.0, size=(self._n_layers, self._n_heads),
            ).astype(np.float32)
        return AttentionMaps(
            weights=weights,
            query_position=n_keys - 1,
            n_keys=n_keys,
            image_token_ranges={"primary": [(1, 1 + self._n_image_tokens)]},
            image_grid_sides={"primary": self._grid_side},
            # Mid-layer range (literature default for LLaMA-class VLMs)
            # is plenty for a mock that only exists to exercise the
            # diagnostic plumbing. No sinks since the synthetic
            # attention is dirichlet-sampled, never softmax-routed.
            metadata={
                "attention_profile": {
                    "recommended_layer_range_fraction": (0.25, 0.75),
                    "literature_citation": "mock VLA — synthetic attention",
                },
            },
        )

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        image, instruction = self._validated_inputs(scene)
        rng = np.random.default_rng(self._seed_for(image, instruction) + 2)
        states = rng.normal(0, 1, (len(layer_indices), self._hidden_dim)).astype(np.float32)
        return HiddenStates(
            states=states,
            query_position=0,
            layer_indices=list(layer_indices),
            hidden_dim=self._hidden_dim,
        )

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> FFNActivations:
        image, instruction = self._validated_inputs(scene)
        rng = np.random.default_rng(self._seed_for(image, instruction) + 3)
        by_layer = {
            li: rng.normal(0, 1, self._hidden_dim * 4).astype(np.float32)
            for li in layer_indices
        }
        return FFNActivations(by_layer=by_layer, query_position=0)

    def predict_with_neuron_ablation(
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        # Mock: ablation perturbs the action by a tiny amount per ablated neuron.
        base = self.predict(scene)
        perturbation = np.array([sum(ablations.values())] * self._action_dim) * 0.01
        return ActionResult(
            action=(base.action + perturbation).astype(np.float32),
            action_dim=self._action_dim,
        )

    # ----- tokenization (trivial mock) ------------------------------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        idx = instruction.lower().find(word.lower())
        if idx < 0:
            return []
        return [idx]  # absolute character offset; good enough for tests

    # ----- helpers --------------------------------------------------------

    def _image_seed(self, image) -> int:
        arr = np.asarray(image)
        return int(hashlib.sha256(arr.tobytes()[:512]).hexdigest()[:8], 16)

    def _seed_for(self, image, instruction: str) -> int:
        return self._image_seed(image) + sum(ord(c) for c in instruction)

    def _image_action(self, image) -> np.ndarray:
        rng = np.random.default_rng(self._image_seed(image))
        return rng.normal(0, 0.3, self._action_dim).astype(np.float32)
