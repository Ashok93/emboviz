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

        # Map our Scene camera names into GR00T's video keys. The user MUST
        # provide a mapping that covers every declared video key — otherwise
        # we'd be silently filling missing GR00T keys with the wrong camera.
        # Default: only auto-map when exactly ONE video key exists (no
        # ambiguity about which camera goes where).
        if camera_mapping is None:
            if len(self._video_keys) == 1:
                camera_mapping = {"primary": self._video_keys[0]}
            elif len(self._video_keys) == 0:
                camera_mapping = {}
            else:
                raise ValueError(
                    f"GR00T embodiment '{embodiment_tag}' declares "
                    f"{len(self._video_keys)} video keys "
                    f"({self._video_keys}). Pass an explicit camera_mapping "
                    "from your Scene camera names → GR00T video keys. We "
                    "do not auto-map silently when multiple cameras are needed."
                )
        # Validate: every declared GR00T video key must be in the mapping's
        # values (covered by some Scene camera).
        mapped_keys = set(camera_mapping.values())
        missing = set(self._video_keys) - mapped_keys
        if missing:
            raise ValueError(
                f"camera_mapping does not cover GR00T video keys {sorted(missing)}. "
                f"Mapping must route every declared video key — "
                "we do not silently fill missing keys with another camera."
            )
        self._camera_mapping: dict[str, str] = camera_mapping

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
        # Every Scene camera named in self._camera_mapping is required —
        # missing ones raise at the framework boundary via
        # RequiredInputs.validate(scene).
        return RequiredInputs(
            cameras=frozenset(self._camera_mapping.keys()),
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
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Gr00tAdapter.predict: {reason}")
        observation = self._build_observation(scene)
        action_dict, _info = self.policy.get_action(observation)

        # Concatenate per-action-key arrays in declared order. GR00T's
        # output shape is conventionally (B, T, D_key) — we keep the full
        # T axis as the action_chunk and expose the first timestep as the
        # immediate action (consistent with OpenVLA-style outputs).
        chunks_per_key: list[np.ndarray] = []
        for key in self._action_keys:
            arr = action_dict.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 3:                # (B, T, D)
                key_chunk = arr[0]           # (T, D)
            elif arr.ndim == 2:              # (T, D)
                key_chunk = arr
            else:                            # (D,) — single-step
                key_chunk = arr[np.newaxis, :]
            chunks_per_key.append(key_chunk.astype(np.float32))

        if chunks_per_key:
            # Align time dim across keys (take min T) then concat along D.
            min_t = min(c.shape[0] for c in chunks_per_key)
            chunk = np.concatenate(
                [c[:min_t] for c in chunks_per_key], axis=-1,
            ).astype(np.float32)
            action = chunk[0]
        else:
            chunk = np.zeros((1, 0), dtype=np.float32)
            action = np.zeros(0, dtype=np.float32)

        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "model_path":     self.model_path,
                "embodiment_tag": self._embodiment_name,
                "action_keys":    list(self._action_keys),
                "chunk_shape":    list(chunk.shape),
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
        truth. We walk into the policy's processor to read it; if the
        processor's introspection path is shaped differently (different
        gr00t version), we warn and fall back to name-based heuristics
        (9d → 9, joint → 7, gripper → 1) so the user knows we are
        guessing.
        """
        import warnings as _warnings
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
        except (AttributeError, KeyError, TypeError) as e:
            _warnings.warn(
                f"GR00T state-dim introspection failed for "
                f"key='{state_key}': {type(e).__name__}: {e}. Falling back "
                "to name-based heuristic; verify the dim matches your "
                "embodiment's normalization spec.",
                stacklevel=2,
            )
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

        # Fill every mapped camera. required_inputs.validate() already
        # confirmed each mapped Scene-camera-name is present and the
        # camera_mapping covers every declared GR00T video key, so any
        # KeyError here is a real bug — let it surface, don't paper over.
        for cam_name, gr00t_key in self._camera_mapping.items():
            arr = np.asarray(scene.observations.images[cam_name].data, dtype=np.uint8)
            if arr.ndim != 3:
                raise ValueError(
                    f"Scene camera '{cam_name}' has shape {arr.shape}; "
                    "expected (H, W, 3)."
                )
            video[gr00t_key] = _to_horizon(arr)

        # State: same temporal-horizon treatment as video.
        state_cfg = self._modality_configs.get("state")
        state_horizon = len(state_cfg.delta_indices) if state_cfg else 1

        # State: distribute Scene state/gripper across GR00T's declared
        # state keys. validate() already checked state is present.
        state: dict[str, np.ndarray] = {}
        import warnings as _warnings
        proprio_vec = np.asarray(
            scene.observations.state.values, dtype=np.float32,
        ).reshape(-1)
        gripper = scene.observations.gripper
        if gripper is None:
            # Some GR00T embodiments declare a gripper state key. If the
            # Scene doesn't carry one, that's a real mismatch — raise.
            for sk in self._state_keys:
                if "gripper" in sk.lower():
                    raise ValueError(
                        f"GR00T embodiment requires gripper state key '{sk}' "
                        "but scene.observations.gripper is None. Populate "
                        "gripper in the dataset adapter."
                    )
            gripper_vec = None
        else:
            gripper_vec = np.array([gripper.value], dtype=np.float32)

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
                # GR00T declared a state key whose name we can't classify.
                # Warn so users know it's filled with zeros rather than
                # discovering it silently in a diagnostic report.
                _warnings.warn(
                    f"GR00T state key '{sk}' is not recognised "
                    "(no 'gripper'/'eef'/'pose'/'joint' substring); filling "
                    f"with zeros of shape ({expected_dim},). Override via a "
                    "custom adapter subclass if your embodiment needs it.",
                    stacklevel=2,
                )
                vec = np.zeros(expected_dim, dtype=np.float32)
            if vec.size != expected_dim:
                raise ValueError(
                    f"GR00T state key '{sk}' expects dim {expected_dim} "
                    f"but Scene provides {vec.size}. Fix the dataset adapter "
                    "or the gripper extractor — no silent pad/truncate."
                )
            state[sk] = _to_state_horizon(vec)

        # Language: GR00T's language modality declares its own keys. We
        # populate ALL declared keys with the (validated non-empty) instruction.
        language: dict = {}
        lang_cfg = self._modality_configs.get("language")
        if lang_cfg is not None:
            for lk in lang_cfg.modality_keys:
                language[lk] = [[scene.instruction]]
        language.setdefault("task", [[scene.instruction]])

        return {"video": video, "state": state, "language": language}
