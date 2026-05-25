"""Adapter for any policy registered in `lerobot.policies`.

One file, many models: covers ACT, Diffusion Policy, TDMPC2, VQ-BeT, and
any future policy LeRobot ships. Designed for the language-free imitation
policies (no text input) that dominate single-arm tabletop research.

Loading: pass a HuggingFace repo_id pointing to a LeRobot-trained
checkpoint. The adapter inspects the policy's input/output features to
build a correct `RequiredInputs` and feed batches in the format the
policy expects.

Capabilities: INFERENCE only — we don't expose hidden states or attention
because LeRobot policies don't have a uniform hook API. Capability-gated
diagnostics (attention, hidden_states, activation patching) auto-skip.

Heavy imports are deferred so this module imports without torch/lerobot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, Scene
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


@register_model("lerobot")
class LeRobotPolicyAdapter(VLAModel):
    """Wraps any `lerobot.policies.PreTrainedPolicy` as a VLAModel.

    Construction:
        LeRobotPolicyAdapter(repo_id="lerobot/act_aloha_sim_transfer_cube_human")
        LeRobotPolicyAdapter(repo_id="lerobot/diffusion_pusht")
    """

    _CAPS = Capability.INFERENCE

    def __init__(
        self,
        repo_id: str,
        device: str = "cuda",
        image_camera: str = "primary",
    ):
        import torch
        from lerobot.policies.pretrained import PreTrainedPolicy

        self.repo_id = repo_id
        self.device = device
        self.image_camera = image_camera
        self.policy = PreTrainedPolicy.from_pretrained(repo_id).to(device).eval()
        # Reset internal state (some policies keep recurrent state).
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        # Inspect declared input/output features so required_inputs is correct.
        self._input_features: dict = getattr(self.policy.config, "input_features", {})
        self._output_features: dict = getattr(self.policy.config, "output_features", {})

        # Determine action_dim from output_features → action.shape
        action_feat = self._output_features.get("action")
        if action_feat is not None and hasattr(action_feat, "shape"):
            self._action_dim = int(np.prod(action_feat.shape))
        else:
            self._action_dim = 7  # reasonable fallback

        # Cache which input keys the policy actually consumes.
        self._needs_state = any(
            k.startswith("observation.state") for k in self._input_features
        )
        self._needs_env_state = "observation.environment_state" in self._input_features
        self._image_keys = [
            k for k in self._input_features if k.startswith("observation.images")
        ]

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.repo_id.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # LeRobot policies typically take one image (some take multi-cam, but
        # those use named keys like observation.images.wrist, etc.). We map
        # the user-supplied `image_camera` to whatever single image input the
        # policy declares.
        return RequiredInputs(
            cameras=frozenset({self.image_camera}),
            instruction=False,
            state=self._needs_state,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        import torch

        # Build the batch dict in LeRobot's format.
        batch: dict = {}
        if self._image_keys:
            primary_img = scene.observations.images.get(self.image_camera)
            if primary_img is None:
                raise ValueError(
                    f"LeRobotPolicyAdapter({self.repo_id}) requires camera "
                    f"{self.image_camera!r} in scene.observations.images"
                )
            arr = np.asarray(primary_img.data).astype(np.float32) / 255.0
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # HWC → CHW
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            # Populate every declared image key with the same primary image,
            # so single-cam wrappers around multi-cam policies still work.
            for k in self._image_keys:
                batch[k] = tensor

        if self._needs_state:
            state = scene.observations.state
            if state is None:
                raise ValueError(
                    f"LeRobotPolicyAdapter({self.repo_id}) requires "
                    f"scene.observations.state but it is None"
                )
            batch["observation.state"] = (
                torch.from_numpy(state.values.astype(np.float32)).unsqueeze(0).to(self.device)
            )

        with torch.inference_mode():
            action_tensor = self.policy.select_action(batch)
        action = action_tensor.detach().cpu().float().numpy().reshape(-1)
        return ActionResult(
            action=action.astype(np.float32),
            action_dim=int(action.size),
            metadata={"repo_id": self.repo_id, "policy_class": type(self.policy).__name__},
        )

    # ----- tokenization (LeRobot policies are language-free) -------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        # Language-free policies have no tokens to find.
        return []
