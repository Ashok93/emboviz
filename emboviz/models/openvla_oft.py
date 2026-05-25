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
        wrist_camera: str = "wrist",
    ):
        self.wrist_camera = wrist_camera
        self.num_images = num_images
        try:
            # NOTE: we deliberately do NOT import from
            # ``experiments.robot.libero.run_libero_eval`` — that module hard-imports
            # ``libero.libero`` (the LIBERO simulator) at module load, which we
            # don't need for pure inference. We inline a minimal config dataclass
            # below that duck-types with what OFT's helpers actually read.
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

        from dataclasses import dataclass as _dataclass
        from typing import Union as _Union

        @_dataclass
        class _InferenceConfig:
            """Minimal GenerateConfig replacement — only inference fields.

            Mirrors the fields OFT's helpers read from cfg. We do NOT inherit
            from the upstream GenerateConfig because that one carries LIBERO
            simulator references; we want this adapter to work on any source
            of Scenes, not just LIBERO sim rollouts.
            """
            model_family: str = "openvla"
            pretrained_checkpoint: _Union[str, "Path"] = ""
            use_l1_regression: bool = True
            use_diffusion: bool = False
            num_diffusion_steps_train: int = 50
            num_diffusion_steps_inference: int = 50
            use_film: bool = False
            num_images_in_input: int = 2
            use_proprio: bool = True
            center_crop: bool = True
            num_open_loop_steps: int = 8
            lora_rank: int = 32
            unnorm_key: _Union[str, "Path"] = ""
            load_in_8bit: bool = False
            load_in_4bit: bool = False
            # Set by override below to avoid the LIBERO-coupled TaskSuite enum.
            task_suite_name: str = "libero_spatial"
        GenerateConfig = _InferenceConfig

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
        # OFT consumes ``num_images`` cameras: the primary camera always,
        # plus the wrist camera when num_images >= 2. We declare BOTH so
        # the framework's Scene validator catches a missing wrist camera
        # at the boundary instead of silently feeding primary as wrist.
        cams = {"primary"}
        if self.num_images >= 2:
            cams.add(self.wrist_camera)
        return RequiredInputs(
            cameras=frozenset(cams),
            instruction=True,
            state=self.use_proprio,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"OpenVLAOFTAdapter.predict: {reason}")
        observation = self._build_observation(scene)
        actions = self._get_vla_action(
            self._cfg, self._vla, self._processor,
            observation, observation["task_description"],
            self._action_head, self._proprio_projector,
        )
        # `actions` is a chunk (chunk_len, action_dim); expose the full
        # chunk via action_chunk and the first row as the immediate action.
        chunk = np.asarray(actions, dtype=np.float32).reshape(-1, self._action_dim)
        action = chunk[0]
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "checkpoint": self.checkpoint,
                "unnorm_key": self.unnorm_key,
                "chunk_size": int(chunk.shape[0]),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Format Scene into the dict OFT's `get_vla_action` expects.

        Strict: primary + (optionally) wrist must be present (already
        enforced by required_inputs.validate); state must be the exact
        ``PROPRIO_DIM`` shape OFT was trained on. No silent fallback.
        """
        primary = np.asarray(scene.observations.images["primary"].data, dtype=np.uint8)
        obs: dict = {"full_image": primary}
        if self.num_images >= 2:
            obs["wrist_image"] = np.asarray(
                scene.observations.images[self.wrist_camera].data, dtype=np.uint8,
            )
        if self.use_proprio:
            state_vals = np.asarray(
                scene.observations.state.values, dtype=np.float32,
            ).reshape(-1)
            if state_vals.size != self._proprio_dim:
                raise ValueError(
                    f"OpenVLA-OFT requires proprio dim {self._proprio_dim} "
                    f"but scene provides {state_vals.size}. Fix the dataset "
                    "adapter's state extractor — we never pad/truncate silently."
                )
            obs["state"] = state_vals
        if not scene.instruction:
            raise ValueError(
                "OpenVLA-OFT requires a non-empty instruction but "
                "scene.instruction is empty."
            )
        obs["task_description"] = scene.instruction
        return obs
