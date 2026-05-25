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
        # PreTrainedPolicy.from_pretrained on the abstract base only works in
        # older lerobot. Newer lerobot (0.5+) needs the policy-class-specific
        # from_pretrained; resolve via config type.
        self.policy = self._load_policy(repo_id).to(device).eval()
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
        # Some LeRobot policies are VLAs (SmolVLA, PI0, PI05) and consume
        # tokenized language even when the config's input_features doesn't
        # explicitly list it. We detect both ways: declared in features OR
        # the policy class is one of the known VLA families.
        cls_name = type(self.policy).__name__.lower()
        is_vla_class = any(name in cls_name for name in ("smolvla", "pi0", "pi05"))
        self._needs_language = is_vla_class or any(
            k.startswith("observation.language") for k in self._input_features
        )
        self._language_processor = self._find_language_processor()

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
        # policy declares. VLA-style policies (SmolVLA, PI0) also need an
        # instruction string we tokenize for them.
        return RequiredInputs(
            cameras=frozenset({self.image_camera}),
            instruction=self._needs_language,
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

        if self._needs_language and self._language_processor is not None:
            instruction = scene.instruction or ""
            try:
                tokens = self._language_processor.tokenizer(
                    instruction, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=48,
                )
            except Exception:
                tokens = self._language_processor.tokenizer(instruction, return_tensors="pt")
            batch["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            if "attention_mask" in tokens:
                # SmolVLA expects bool mask; tokenizers return int64.
                batch["observation.language.attention_mask"] = (
                    tokens["attention_mask"].to(self.device).bool()
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

    # ----- helpers -------------------------------------------------------

    def _find_language_processor(self):
        """Locate the policy's text processor / tokenizer if present.

        VLA-style LeRobot policies bury their processor in different places:
        SmolVLA exposes it at `policy.model.vlm_with_expert.processor`. We
        search common locations defensively so this adapter Just Works
        across SmolVLA, future PI0-via-LeRobot, etc.
        """
        candidates = (
            ("model", "vlm_with_expert", "processor"),
            ("model", "vlm", "processor"),
            ("model", "processor"),
            ("processor",),
        )
        for path in candidates:
            obj = self.policy
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and obj is not None and hasattr(obj, "tokenizer"):
                return obj
        return None

    @staticmethod
    def _load_policy(repo_id: str):
        """Resolve the correct LeRobot policy subclass for `repo_id`.

        LeRobot's `PreTrainedPolicy.from_pretrained` only works for the
        concrete subclass — calling it on the abstract base raises because
        Python can't instantiate the abstract class. Newer LeRobot
        registers policy types in `PreTrainedConfig`; we read the config's
        `type` field and dispatch to the matching subclass.
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pretrained import PreTrainedPolicy

        cfg = PreTrainedConfig.from_pretrained(repo_id)
        policy_type = getattr(cfg, "type", None) or type(cfg).__name__.lower()

        # Lazy-import each candidate so missing optional policies don't break
        # adapter import. The mapping covers LeRobot's first-party policies.
        candidates = {
            "smolvla": "lerobot.policies.smolvla.modeling_smolvla:SmolVLAPolicy",
            "pi0":     "lerobot.policies.pi0.modeling_pi0:PI0Policy",
            "pi05":    "lerobot.policies.pi05.modeling_pi05:PI05Policy",
            "act":     "lerobot.policies.act.modeling_act:ACTPolicy",
            "diffusion": "lerobot.policies.diffusion.modeling_diffusion:DiffusionPolicy",
            "tdmpc":   "lerobot.policies.tdmpc.modeling_tdmpc:TDMPCPolicy",
            "tdmpc2":  "lerobot.policies.tdmpc2.modeling_tdmpc2:TDMPC2Policy",
            "vqbet":   "lerobot.policies.vqbet.modeling_vqbet:VQBeTPolicy",
        }

        target = candidates.get(str(policy_type).lower())
        if target is None:
            # Last-ditch: try the abstract from_pretrained (works on older LeRobot).
            return PreTrainedPolicy.from_pretrained(repo_id)

        import importlib
        module_path, _, class_name = target.partition(":")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls.from_pretrained(repo_id)
