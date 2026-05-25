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

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Convert our typed Scene into GR00T's nested observation dict."""
        # Video: each declared GR00T video key gets a (B=1, T=1, H, W, 3) uint8 tensor.
        video: dict[str, np.ndarray] = {}
        for cam_name, gr00t_key in self._camera_mapping.items():
            rgb = scene.observations.images.get(cam_name)
            if rgb is None:
                continue
            arr = np.asarray(rgb.data, dtype=np.uint8)
            if arr.ndim == 3:
                arr = arr[np.newaxis, np.newaxis, :, :, :]   # (1, 1, H, W, 3)
            video[gr00t_key] = arr

        # State: distribute scene.observations.state.values across declared state_keys.
        # GR00T expects per-key sub-vectors; for v1 we put the full state under
        # the first declared state key (teams with multi-segment state should
        # provide a custom mapping in their own adapter subclass).
        state: dict[str, np.ndarray] = {}
        proprio = scene.observations.state
        if proprio is not None and self._state_keys:
            vals = proprio.values.astype(np.float32)
            if vals.ndim == 1:
                vals = vals[np.newaxis, np.newaxis, :]       # (1, 1, D)
            state[self._state_keys[0]] = vals

        # Language: GR00T expects (B=1, 1) list-of-lists of strings.
        language = {"task": [[scene.instruction or ""]]}

        return {"video": video, "state": state, "language": language}
