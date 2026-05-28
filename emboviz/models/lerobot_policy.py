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
        camera_mapping: Optional[dict[str, str]] = None,
    ):
        """Args:
            repo_id: HuggingFace repo id of a LeRobot-trained policy.
            device: torch device.
            camera_mapping: Scene camera name → LeRobot image key
              (e.g. ``{"primary": "observation.images.top", "wrist_left":
              "observation.images.cam_left_wrist"}``). REQUIRED whenever
              the policy declares more than one image key; auto-mapped only
              when the policy declares exactly one image key (no ambiguity).
        """
        import torch
        from lerobot.policies.pretrained import PreTrainedPolicy

        self.repo_id = repo_id
        self.device = device
        self.policy = self._load_policy(repo_id).to(device).eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self._input_features: dict = getattr(self.policy.config, "input_features", {})
        self._output_features: dict = getattr(self.policy.config, "output_features", {})

        action_feat = self._output_features.get("action")
        if action_feat is None or not hasattr(action_feat, "shape"):
            raise ValueError(
                f"LeRobot policy {repo_id} does not declare an 'action' "
                "output_feature with a .shape — we cannot infer action_dim "
                "and won't guess 7 silently. Fix the policy config."
            )
        self._action_dim = int(np.prod(action_feat.shape))

        self._needs_state = any(
            k.startswith("observation.state") for k in self._input_features
        )
        self._needs_env_state = "observation.environment_state" in self._input_features
        self._image_keys = [
            k for k in self._input_features if k.startswith("observation.images")
        ]

        # Build the camera_mapping. Strict: every declared image key must be
        # covered; never feed the same camera into multiple slots silently.
        if camera_mapping is None:
            if len(self._image_keys) == 0:
                self._camera_mapping: dict[str, str] = {}
            elif len(self._image_keys) == 1:
                self._camera_mapping = {"primary": self._image_keys[0]}
            else:
                raise ValueError(
                    f"LeRobot policy {repo_id} declares {len(self._image_keys)} "
                    f"image keys {self._image_keys}. Pass an explicit "
                    "camera_mapping={scene_cam: lerobot_image_key, ...} — we "
                    "do not silently feed the same camera into multiple slots."
                )
        else:
            mapped_keys = set(camera_mapping.values())
            missing = set(self._image_keys) - mapped_keys
            extra = mapped_keys - set(self._image_keys)
            if missing:
                raise ValueError(
                    f"camera_mapping missing entries for image keys "
                    f"{sorted(missing)}. Every declared key must be routed."
                )
            if extra:
                raise ValueError(
                    f"camera_mapping has entries {sorted(extra)} that are "
                    f"not declared by the policy. Known keys: {self._image_keys}."
                )
            self._camera_mapping = dict(camera_mapping)
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
        return RequiredInputs(
            cameras=frozenset(self._camera_mapping.keys()),
            instruction=self._needs_language,
            state=self._needs_state,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        import torch

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"LeRobotPolicyAdapter.predict: {reason}")

        batch: dict = {}
        for scene_cam, lerobot_key in self._camera_mapping.items():
            arr = np.asarray(
                scene.observations.images[scene_cam].data,
            ).astype(np.float32) / 255.0
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # HWC → CHW
            batch[lerobot_key] = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        if self._needs_state:
            state = scene.observations.state
            batch["observation.state"] = (
                torch.from_numpy(state.values.astype(np.float32)).unsqueeze(0).to(self.device)
            )

        if self._needs_language:
            if self._language_processor is None:
                raise RuntimeError(
                    f"Policy {self.repo_id} declares it needs language but no "
                    "tokenizer/processor was found on the policy object. "
                    "Update _find_language_processor() to locate it."
                )
            instruction = scene.instruction
            # Two known tokenizer signatures; try the more-specific one first
            # and let unexpected errors surface (no bare except).
            try:
                tokens = self._language_processor.tokenizer(
                    instruction, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=48,
                )
            except TypeError:
                # Some tokenizers reject padding="max_length" / max_length kwargs.
                tokens = self._language_processor.tokenizer(
                    instruction, return_tensors="pt",
                )
            batch["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            if "attention_mask" in tokens:
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
