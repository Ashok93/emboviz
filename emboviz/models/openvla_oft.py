"""Adapter for OpenVLA-OFT (Optimized Fine-Tuning) checkpoints.

OFT is Stanford's enhanced fine-tuning recipe for OpenVLA, adding
parallel decoding + L1-regression action head + first-class proprioception
input. It significantly outperforms OpenVLA-7B on LIBERO benchmarks.

**Install (separate virtualenv required):**

OpenVLA-OFT pins a FORK of `transformers` (moojink/transformers-openvla-oft)
for bidirectional attention support. This fork is incompatible with both
mainline `transformers` and other adapters' transformers pins, so OFT
must live in its own virtualenv:

    git clone https://github.com/moojink/openvla-oft.git
    cd openvla-oft && pip install -e .
    pip install -e /path/to/emboviz   # add emboviz on top

Then construct the adapter with a checkpoint id (e.g.
"moojink/openvla-7b-oft-finetuned-libero-spatial").

Capabilities: INFERENCE. Internal-introspection (attention, hidden
states, patching) is not exposed through the OFT inference utilities;
capability-gated diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, Scene
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


_DEFAULT_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial"


@register_model("openvla-oft")
class OpenVLAOFTAdapter(VLAModel):
    """Wraps the openvla-oft inference utilities as a VLAModel.

    Construction:
        OpenVLAOFTAdapter()  # default LIBERO-spatial checkpoint
        OpenVLAOFTAdapter(checkpoint="moojink/openvla-7b-oft-finetuned-libero-10")
        OpenVLAOFTAdapter(checkpoint="...", unnorm_key="libero_10_no_noops",
                          num_images=2, use_proprio=True)
    """

    _CAPS = Capability.INFERENCE

    def __init__(
        self,
        checkpoint: str = _DEFAULT_REPO,
        unnorm_key: str = "libero_spatial_no_noops",
        num_images: int = 2,
        use_proprio: bool = True,
        use_l1_regression: bool = True,
        use_film: bool = False,
        center_crop: bool = True,
    ):
        try:
            from experiments.robot.libero.run_libero_eval import GenerateConfig
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_processor,
                get_proprio_projector,
                get_vla,
                get_vla_action,
            )
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
        except ImportError as e:
            raise ImportError(
                "OpenVLA-OFT requires the openvla-oft repository (separate venv).\n"
                "Setup:\n"
                "    git clone https://github.com/moojink/openvla-oft.git\n"
                "    cd openvla-oft && pip install -e .\n"
                "Then install emboviz on top of that environment."
            ) from e

        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        self.use_proprio = use_proprio

        cfg = GenerateConfig(
            pretrained_checkpoint=checkpoint,
            use_l1_regression=use_l1_regression,
            use_diffusion=False,
            use_film=use_film,
            num_images_in_input=num_images,
            use_proprio=use_proprio,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=center_crop,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key=unnorm_key,
        )
        self._cfg = cfg

        self._vla = get_vla(cfg)
        self._processor = get_processor(cfg)
        self._action_head = get_action_head(cfg, llm_dim=self._vla.llm_dim)
        self._proprio_projector = (
            get_proprio_projector(cfg, llm_dim=self._vla.llm_dim, proprio_dim=PROPRIO_DIM)
            if use_proprio else None
        )
        self._get_vla_action = get_vla_action
        self._proprio_dim = PROPRIO_DIM
        self._action_dim = 7   # standard 7-DOF for OpenVLA-class

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.checkpoint.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # OFT canonically uses one primary camera + an optional wrist camera +
        # proprio + instruction. We declare the primary and let teams add a
        # wrist via custom adapter subclass.
        return RequiredInputs(
            cameras=frozenset({"primary"}),
            instruction=True,
            state=self.use_proprio,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        observation = self._build_observation(scene)
        actions = self._get_vla_action(
            self._cfg, self._vla, self._processor,
            observation, observation["task_description"],
            self._action_head, self._proprio_projector,
        )
        # `actions` is a chunk; take the first immediate action.
        action = np.asarray(actions, dtype=np.float32).reshape(-1, self._action_dim)[0]
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            metadata={
                "checkpoint": self.checkpoint,
                "unnorm_key": self.unnorm_key,
                "chunk_size": int(np.asarray(actions).reshape(-1, self._action_dim).shape[0]),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Format Scene into the dict OFT's `get_vla_action` expects."""
        primary = np.asarray(scene.primary_image_data, dtype=np.uint8)
        wrist_img = scene.observations.images.get("wrist")
        wrist = (
            np.asarray(wrist_img.data, dtype=np.uint8) if wrist_img is not None
            else primary  # fallback: feed primary as wrist if user has no wrist cam
        )
        state_vals = (
            scene.observations.state.values.astype(np.float32)
            if scene.observations.state is not None
            else np.zeros(self._proprio_dim, dtype=np.float32)
        )
        # Pad/truncate to PROPRIO_DIM as OFT expects.
        if state_vals.size < self._proprio_dim:
            state_vals = np.pad(state_vals, (0, self._proprio_dim - state_vals.size))
        else:
            state_vals = state_vals[: self._proprio_dim]
        return {
            "full_image": primary,
            "wrist_image": wrist,
            "state": state_vals,
            "task_description": scene.instruction or "",
        }
