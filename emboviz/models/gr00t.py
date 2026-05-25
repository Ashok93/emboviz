"""Adapter for NVIDIA Isaac GR00T-N1 / N1.7.

GR00T is NVIDIA's open foundation model for generalist humanoid robots,
taking multi-camera video, proprioception, and language instructions to
produce action chunks. The adapter wraps `gr00t.policy.Gr00tPolicy` and
maps our typed `Scene` into GR00T's expected observation dict.

Optional dependency: install with
    uv pip install git+https://github.com/NVIDIA/Isaac-GR00T.git

Loading without the package raises ImportError at adapter construction
time with a clear install hint.

Capabilities: INFERENCE only. GR00T's introspection surface (attention,
hidden states) is not uniformly exposed through Gr00tPolicy; capability-
gated diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, Scene
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


# Sensible default embodiment for tabletop manipulation rollouts. Teams
# with their own embodiment override via the `embodiment_tag` arg.
DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.7-3B"


@register_model("gr00t")
@register_model("gr00t-n1")
class Gr00tAdapter(VLAModel):
    """Wraps `gr00t.policy.Gr00tPolicy` as an Emboviz `VLAModel`.

    Construction:
        Gr00tAdapter()                                          # default checkpoint + embodiment
        Gr00tAdapter(model_path="nvidia/GR00T-N1-2B")
        Gr00tAdapter(embodiment_tag="GR1", camera_mapping={"primary": "video.ego_view"})

    `camera_mapping` translates our Scene camera names into GR00T's
    embodiment-specific video keys. By default we route our `"primary"`
    into the first declared video key of the selected embodiment.
    """

    _CAPS = Capability.INFERENCE

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embodiment_tag: str = DEFAULT_EMBODIMENT,
        device: str = "cuda:0",
        camera_mapping: Optional[dict[str, str]] = None,
    ):
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy import Gr00tPolicy
        except ImportError as e:
            raise ImportError(
                "Gr00tAdapter requires NVIDIA Isaac-GR00T. Install with:\n"
                "    uv pip install git+https://github.com/NVIDIA/Isaac-GR00T.git"
            ) from e

        self.model_path = model_path
        self._embodiment_name = embodiment_tag
        try:
            embodiment_enum = getattr(EmbodimentTag, embodiment_tag)
        except AttributeError as e:
            available = [t.name for t in EmbodimentTag]
            raise ValueError(
                f"Unknown embodiment_tag '{embodiment_tag}'. Available: {available}"
            ) from e

        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_enum,
            device=device,
        )

        self._modality_configs = self.policy.get_modality_config()
        self._video_keys: list[str] = list(self._modality_configs["video"].modality_keys)
        self._state_keys: list[str] = list(self._modality_configs["state"].modality_keys)
        self._action_keys: list[str] = list(self._modality_configs["action"].modality_keys)

        # Map our Scene camera names into GR00T's video keys.
        # Default: route "primary" into the first declared video key.
        if camera_mapping is None and self._video_keys:
            camera_mapping = {"primary": self._video_keys[0]}
        self._camera_mapping: dict[str, str] = camera_mapping or {}

        # Action dim is sum of dims across action_keys; per first inference probe.
        self._action_dim: Optional[int] = None

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.model_path.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # GR00T needs at least one camera + state + (usually) language.
        return RequiredInputs(
            cameras=frozenset(self._camera_mapping.keys()) or frozenset({"primary"}),
            instruction=True,
            state=True,
        )

    @property
    def action_dim(self) -> int:
        # GR00T's per-key action dims aren't trivially known before the first
        # inference; we report 0 until we've seen one prediction.
        return self._action_dim or 0

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        observation = self._build_observation(scene)
        action_dict, _info = self.policy.get_action(observation)

        # Concatenate per-action-key arrays in declared order; take the first
        # timestep so we return a single immediate action consistent with
        # OpenVLA-style outputs.
        flat_parts: list[np.ndarray] = []
        for key in self._action_keys:
            arr = action_dict.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            # Shape conventionally (B, T, D) → take batch 0, first timestep.
            if arr.ndim == 3:
                arr = arr[0, 0, :]
            elif arr.ndim == 2:
                arr = arr[0, :]
            flat_parts.append(arr.reshape(-1))
        action = np.concatenate(flat_parts).astype(np.float32) if flat_parts else np.zeros(0, dtype=np.float32)

        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            metadata={
                "model_path": self.model_path,
                "embodiment_tag": self._embodiment_name,
                "action_keys": list(self._action_keys),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        # GR00T's tokenizer surface isn't uniformly exposed; language-axis
        # token-position queries fall through (diagnostics that need them
        # will fail their capability check anyway).
        return []

    def _state_key_dim(self, state_key: str) -> int:
        """Inferred dim for a GR00T state key.

        GR00T's normalization metadata stores per-key min/max — that's the
        truth. We walk into the policy's processor to read it; if anything
        in that path isn't present we fall back to name-based heuristics
        (9d → 9, joint → 7, gripper → 1).
        """
        try:
            proc = self.policy.processor
            sap = proc.state_action_processor
            embodiments = getattr(sap, "state_norm", None)
            if embodiments:
                params = (
                    embodiments.get(self._embodiment_name.lower())
                    or next(iter(embodiments.values()))
                )
                if params and state_key in params:
                    p = params[state_key]
                    mn = getattr(p, "min", None)
                    if mn is not None and hasattr(mn, "__len__"):
                        return int(len(mn))
        except Exception:
            pass
        k = state_key.lower()
        if "9d" in k:
            return 9
        if "gripper" in k:
            return 1
        return 7

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Convert our typed Scene into GR00T's nested observation dict.

        GR00T's video modality expects shape (B, T, H, W, 3) where T is the
        embodiment-specific temporal horizon (often 2 = current + previous
        frame). When we only have one Scene we repeat it along the time
        axis — this means "no motion" but is a valid input shape.
        """
        # Read the per-modality temporal horizon (T) from the modality config.
        video_cfg = self._modality_configs["video"]
        video_horizon = len(video_cfg.delta_indices)

        video: dict[str, np.ndarray] = {}

        def _to_horizon(arr_3d: np.ndarray) -> np.ndarray:
            """(H, W, 3) → (1, T, H, W, 3)."""
            stacked = np.stack([arr_3d] * video_horizon, axis=0)
            return stacked[np.newaxis, ...]

        # Fill EXPLICITLY mapped cameras first.
        mapped_arrays: dict[str, np.ndarray] = {}
        for cam_name, gr00t_key in self._camera_mapping.items():
            rgb = scene.observations.images.get(cam_name)
            if rgb is None:
                continue
            arr = np.asarray(rgb.data, dtype=np.uint8)
            if arr.ndim == 3:
                video[gr00t_key] = _to_horizon(arr)
                mapped_arrays[gr00t_key] = arr

        # GR00T policies validate that EVERY declared video key is present.
        # For any unmapped video key (e.g. a wrist cam the user doesn't have),
        # we fall back to a mapped camera's image so the model has SOMETHING
        # rather than crashing. Real teams should provide their full multi-cam
        # mapping; the wizard generates one based on robot profile.
        fallback = next(iter(mapped_arrays.values()), None)
        if fallback is not None:
            for vk in self._video_keys:
                if vk not in video:
                    video[vk] = _to_horizon(fallback)

        # State: same temporal-horizon treatment as video.
        state_cfg = self._modality_configs.get("state")
        state_horizon = len(state_cfg.delta_indices) if state_cfg else 1

        # State: distribute scene.observations.state.values across declared state_keys.
        # GR00T expects per-key sub-vectors; for v1 we put the full state under
        # the first declared state key (teams with multi-segment state should
        # provide a custom mapping in their own adapter subclass).
        # GR00T validates every declared state key is present. We map our
        # Scene's typed fields (state, gripper) onto common GR00T key names
        # and fall back to zeros for keys we can't infer.
        state: dict[str, np.ndarray] = {}
        proprio = scene.observations.state
        gripper = scene.observations.gripper
        proprio_vec = (
            proprio.values.astype(np.float32) if proprio is not None
            else np.zeros(7, dtype=np.float32)
        )
        gripper_vec = np.array(
            [gripper.value if gripper is not None else 0.0], dtype=np.float32,
        )

        def _to_state_horizon(vec: np.ndarray) -> np.ndarray:
            stacked = np.stack([vec] * state_horizon, axis=0)
            return stacked[np.newaxis, ...]

        for sk in self._state_keys:
            sk_lower = sk.lower()
            expected_dim = self._state_key_dim(sk)
            if "gripper" in sk_lower:
                vec = gripper_vec
            elif "eef" in sk_lower or "pose" in sk_lower or "joint" in sk_lower:
                vec = proprio_vec
            else:
                vec = np.zeros(expected_dim, dtype=np.float32)
            # Truncate / zero-pad to the dim this key expects.
            if vec.size < expected_dim:
                vec = np.pad(vec, (0, expected_dim - vec.size))
            else:
                vec = vec[:expected_dim]
            state[sk] = _to_state_horizon(vec)

        # Language: GR00T's language modality declares its own keys (e.g.
        # 'annotation.language.language_instruction'). Populate them all
        # with the instruction string.
        language: dict = {}
        lang_cfg = self._modality_configs.get("language")
        if lang_cfg is not None:
            for lk in lang_cfg.modality_keys:
                language[lk] = [[scene.instruction or ""]]
        # Always include a 'task' alias too, for embodiments that expect it.
        language.setdefault("task", [[scene.instruction or ""]])

        return {"video": video, "state": state, "language": language}
