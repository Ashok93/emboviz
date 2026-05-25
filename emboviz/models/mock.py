"""A deterministic mock VLA — for unit-testing diagnostics without GPU.

The mock's behaviour is configurable so we can simulate a noun-blind model,
a properly grounded model, or an over-sensitive model — and verify our
diagnostics produce the right verdict. This is what lets us iterate on the
core algorithms without spending GPU time.
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
    ImageLike,
    TokenSelector,
)
from emboviz.models.protocol import Capability, NotSupported, VLAModel
from emboviz.models.registry import register_model


@register_model("mock")
class MockVLA(VLAModel):
    """Deterministic adapter for testing.

    Modes:
      • `noun_blind`  — same action regardless of instruction (visual-only)
      • `grounded`    — action depends linearly on instruction hash
      • `random`      — uniformly random per call (controlled by seed)
      • `nuanced`     — partial grounding: depends on instruction's noun
                        category but ignores fine word distinctions
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
        mode: Literal["noun_blind", "grounded", "random", "nuanced"] = "noun_blind",
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

    def predict(self, image: ImageLike, instruction: str) -> ActionResult:
        rng = np.random.default_rng(self._seed_for(image, instruction))
        if self.mode == "random":
            action = rng.normal(0, 0.05, self._action_dim).astype(np.float32)
        elif self.mode == "noun_blind":
            # Action depends ONLY on image hash, not instruction
            action = self._image_action(image)
        elif self.mode == "nuanced":
            # Action depends on the *first character* of the instruction
            # (proxy for "model uses some language but not subtleties")
            seed_for_noun = sum(ord(c) for c in (instruction[:1] or "_"))
            sub_rng = np.random.default_rng(self.seed + seed_for_noun)
            action = self._image_action(image) + sub_rng.normal(0, 0.1, self._action_dim).astype(np.float32)
        elif self.mode == "grounded":
            # Action depends on instruction content
            instr_seed = int(hashlib.sha256(instruction.encode()).hexdigest()[:8], 16)
            sub_rng = np.random.default_rng(self.seed + instr_seed)
            action = sub_rng.normal(0, 0.3, self._action_dim).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return ActionResult(action=action, action_dim=self._action_dim)

    # ----- attention ------------------------------------------------------

    def extract_attention(
        self, image: ImageLike, instruction: str, query: TokenSelector,
    ) -> AttentionMaps:
        # Fake attention: a (n_layers, n_heads, n_image_tokens) tensor
        # whose distribution depends on the instruction (so JS divergence
        # between two instructions is non-zero for `grounded`/`nuanced`).
        rng = np.random.default_rng(self._seed_for(image, instruction) + 1)
        n_text = 10  # arbitrary text length for mock
        n_keys = 1 + self._n_image_tokens + n_text
        weights = rng.dirichlet(
            np.ones(n_keys) * (0.5 if self.mode != "noun_blind" else 5.0),
            size=(self._n_layers, self._n_heads),
        ).astype(np.float32)
        if self.mode == "noun_blind":
            # Override with instruction-independent distribution.
            base_rng = np.random.default_rng(self._image_seed(image) + 1)
            weights = base_rng.dirichlet(
                np.ones(n_keys) * 5.0, size=(self._n_layers, self._n_heads),
            ).astype(np.float32)
        return AttentionMaps(
            weights=weights,
            query_position=n_keys - 1,
            n_keys=n_keys,
            image_token_range=(1, 1 + self._n_image_tokens),
            image_grid_side=self._grid_side,
        )

    def extract_hidden_states(
        self, image, instruction, layer_indices, query: TokenSelector,
    ) -> HiddenStates:
        rng = np.random.default_rng(self._seed_for(image, instruction) + 2)
        states = rng.normal(0, 1, (len(layer_indices), self._hidden_dim)).astype(np.float32)
        return HiddenStates(
            states=states,
            query_position=0,
            layer_indices=list(layer_indices),
            hidden_dim=self._hidden_dim,
        )

    def extract_ffn_activations(
        self, image, instruction, layer_indices, query: TokenSelector,
    ) -> FFNActivations:
        rng = np.random.default_rng(self._seed_for(image, instruction) + 3)
        by_layer = {
            li: rng.normal(0, 1, self._hidden_dim * 4).astype(np.float32)
            for li in layer_indices
        }
        return FFNActivations(by_layer=by_layer, query_position=0)

    def predict_with_neuron_ablation(
        self, image, instruction, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        # Mock: ablation perturbs the action by a tiny amount per ablated neuron.
        base = self.predict(image, instruction)
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

    def _image_seed(self, image: ImageLike) -> int:
        arr = np.asarray(image)
        return int(hashlib.sha256(arr.tobytes()[:512]).hexdigest()[:8], 16)

    def _seed_for(self, image: ImageLike, instruction: str) -> int:
        return self._image_seed(image) + sum(ord(c) for c in instruction)

    def _image_action(self, image: ImageLike) -> np.ndarray:
        rng = np.random.default_rng(self._image_seed(image))
        return rng.normal(0, 0.3, self._action_dim).astype(np.float32)
