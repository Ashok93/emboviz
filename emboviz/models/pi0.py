"""Adapter for Physical Intelligence's π0 / π0.5 / π0-FAST via openpi.

The `openpi` repository (https://github.com/Physical-Intelligence/openpi)
is PI's official open-source inference path for the π0 family. Each
checkpoint is paired with a platform-specific observation format (DROID,
ALOHA, LIBERO, UR5, custom). This adapter wraps openpi's
`create_trained_policy` and maps our typed `Scene` into the format the
chosen config expects.

**Install (its own virtualenv):**

    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    uv pip install --no-deps -e /path/to/emboviz   # add emboviz on top

Then construct with a config name (e.g. "pi0_aloha_sim", "pi0_libero",
"pi0_fast_droid", "pi05_libero", "pi05_droid").

Strict contract:
  • Each platform builder REQUIRES the cameras / state shape its
    upstream policy was trained on. Missing cameras or mis-shaped state
    raise ValueError — we never silently feed the primary camera into
    the wrist slot or zero-pad an unexpected state vector.
  • Each platform declares its own ``RequiredInputs`` so the framework's
    Scene-validation can surface the failure at the boundary, before
    we ever call the model.

Capabilities: INFERENCE only. openpi's inference path doesn't expose
hidden states or attention through a stable API; capability-gated
diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from emboviz.core.types import ActionResult, Scene
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


# Default checkpoint URI prefix from openpi's published checkpoints
_GCS_PREFIX = "gs://openpi-assets/checkpoints/"


def _to_chw_uint8(pil_or_arr) -> np.ndarray:
    """Convert PIL/HWC array to (3, H, W) uint8."""
    arr = np.asarray(pil_or_arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:   # HWC → CHW
        arr = arr.transpose(2, 0, 1)
    return arr.astype(np.uint8)


def _require_image(scene: Scene, *names: str) -> "np.ndarray":
    """Return the image data for the FIRST name present in the scene.

    Raises KeyError listing the names tried + the cameras actually
    available — no silent substitution.
    """
    images = scene.observations.images
    for n in names:
        if n in images:
            return images[n].data
    raise KeyError(
        f"None of the required cameras {list(names)} are in the scene. "
        f"Available cameras: {sorted(images)}. The dataset adapter must "
        "load one of the listed cameras under the expected name (rename "
        "via image_keys) — we never substitute another camera silently."
    )


def _require_state_exact(scene: Scene, expected_dim: int, platform: str) -> "np.ndarray":
    """Return the state vector, raising if missing or the wrong shape.

    Pads and truncates were previously silent; now the caller is told
    EXACTLY what shape mismatch they have so they can fix the loader.
    """
    state = scene.observations.state
    if state is None:
        raise ValueError(
            f"π0 ({platform}) requires proprioceptive state with dim "
            f"{expected_dim}, but scene.observations.state is None. "
            "Either populate state in the dataset adapter or use a config "
            "that doesn't need state."
        )
    vec = np.asarray(state.values, dtype=np.float32).reshape(-1)
    if vec.size != expected_dim:
        raise ValueError(
            f"π0 ({platform}) requires state dim {expected_dim} but scene "
            f"provides {vec.size}. Fix the dataset adapter's state_key / "
            "gripper_extractor to emit the expected layout — we never "
            "silently pad or truncate."
        )
    return vec


def _aloha_observation_builder(scene: Scene) -> dict:
    """openpi ALOHA observation format — bimanual, 4 cameras, state(14)."""
    state = _require_state_exact(scene, 14, "aloha")
    return {
        "state": state,
        "images": {
            "cam_high":        _to_chw_uint8(_require_image(scene, "cam_high", "head")),
            "cam_low":         _to_chw_uint8(_require_image(scene, "cam_low", "front")),
            "cam_left_wrist":  _to_chw_uint8(_require_image(scene, "cam_left_wrist", "wrist_left")),
            "cam_right_wrist": _to_chw_uint8(_require_image(scene, "cam_right_wrist", "wrist_right")),
        },
        "prompt": _require_instruction(scene, "aloha"),
    }


def _droid_observation_builder(scene: Scene) -> dict:
    """openpi DROID observation format — single arm + wrist + state(7)+gripper."""
    obs: dict = {
        "observation/exterior_image_1_left":
            np.asarray(_require_image(scene, "primary"), dtype=np.uint8),
        "observation/wrist_image_left":
            np.asarray(_require_image(scene, "wrist_left", "wrist"), dtype=np.uint8),
        "observation/joint_position":
            _require_state_exact(scene, 7, "droid"),
        "prompt": _require_instruction(scene, "droid"),
    }
    if scene.observations.gripper is None:
        raise ValueError(
            "π0 (droid) requires gripper state but scene.observations.gripper "
            "is None. Populate gripper in the dataset adapter."
        )
    obs["observation/gripper_position"] = np.array(
        [scene.observations.gripper.value], dtype=np.float32,
    )
    return obs


def _libero_observation_builder(scene: Scene) -> dict:
    """openpi LIBERO observation format — 2 cameras + state(8)."""
    return {
        "observation/image":
            _to_chw_uint8(_require_image(scene, "primary")),
        "observation/wrist_image":
            _to_chw_uint8(_require_image(scene, "wrist", "wrist_left")),
        "observation/state":
            _require_state_exact(scene, 8, "libero"),
        "prompt": _require_instruction(scene, "libero"),
    }


def _require_instruction(scene: Scene, platform: str) -> str:
    instr = scene.instruction
    if not instr:
        raise ValueError(
            f"π0 ({platform}) requires a non-empty instruction but "
            "scene.instruction is None or empty. The dataset adapter "
            "must produce a task string."
        )
    return instr


# Built-in observation builders keyed by config-name fragment.
_BUILDER_REGISTRY: dict[str, Callable[[Scene], dict]] = {
    "aloha":  _aloha_observation_builder,
    "droid":  _droid_observation_builder,
    "libero": _libero_observation_builder,
}


# Per-platform RequiredInputs declarations. Matches what the builders
# actually consume so RequiredInputs.validate() catches missing fields
# at the framework boundary instead of inside the builder.
_REQUIRED_INPUTS_REGISTRY: dict[str, RequiredInputs] = {
    "aloha": RequiredInputs(
        cameras=frozenset({"cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"}),
        instruction=True,
        state=True,
    ),
    "droid": RequiredInputs(
        cameras=frozenset({"primary", "wrist_left"}),
        instruction=True,
        state=True,
        gripper=True,
    ),
    "libero": RequiredInputs(
        cameras=frozenset({"primary", "wrist"}),
        instruction=True,
        state=True,
    ),
}


def _resolve_platform(config_name: str) -> str:
    """Pick a platform key based on substring match in config_name.

    Raises ValueError if the config doesn't match a known platform — we
    do not silently default to DROID (the old behaviour quietly fed
    DROID-shaped observations to a non-DROID checkpoint).
    """
    low = config_name.lower()
    for key in _BUILDER_REGISTRY:
        if key in low:
            return key
    raise ValueError(
        f"Cannot infer π0 platform from config_name='{config_name}'. "
        f"Known platforms: {sorted(_BUILDER_REGISTRY)}. Pass "
        "observation_builder + required_inputs explicitly for a custom "
        "platform."
    )


@register_model("pi0")
@register_model("pi05")
class Pi0Adapter(VLAModel):
    """Wraps `openpi`'s trained policy as a VLAModel.

    Construction:
        Pi0Adapter()                                        # pi0_fast_droid default
        Pi0Adapter(config_name="pi0_libero")
        Pi0Adapter(config_name="pi05_droid",
                   observation_builder=my_custom_builder,
                   required_inputs=my_custom_inputs)

    For known platforms (DROID/ALOHA/LIBERO) the builder + required-inputs
    are picked automatically. For a custom platform pass both explicitly —
    we do not fall back to DROID silently.
    """

    _CAPS = Capability.INFERENCE

    def __init__(
        self,
        config_name: str = "pi0_fast_droid",
        checkpoint_uri: Optional[str] = None,
        observation_builder: Optional[Callable[[Scene], dict]] = None,
        required_inputs: Optional[RequiredInputs] = None,
    ):
        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.shared import download
            from openpi.training import config as _config
        except ImportError as e:
            raise ImportError(
                "Pi0Adapter requires the openpi package (separate venv).\n"
                "Setup:\n"
                "    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
                "    cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync\n"
                "Then install emboviz on top of openpi's venv."
            ) from e

        self.config_name = config_name
        cfg = _config.get_config(config_name)
        ckpt_uri = checkpoint_uri or f"{_GCS_PREFIX}{config_name}"
        checkpoint_dir = download.maybe_download(ckpt_uri)
        self._policy = _policy_config.create_trained_policy(cfg, checkpoint_dir)
        self._cfg = cfg

        if observation_builder is not None and required_inputs is not None:
            self._observation_builder = observation_builder
            self._required_inputs = required_inputs
        elif observation_builder is None and required_inputs is None:
            platform = _resolve_platform(config_name)
            self._observation_builder = _BUILDER_REGISTRY[platform]
            self._required_inputs = _REQUIRED_INPUTS_REGISTRY[platform]
        else:
            raise ValueError(
                "Pi0Adapter: pass BOTH observation_builder and required_inputs "
                "together (for a custom platform), or NEITHER (auto-pick from "
                "config_name). Mixing one custom + one auto would silently lie "
                "about what the model actually consumes."
            )
        self._action_dim = 0

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.config_name

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        return self._required_inputs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        reason = self._required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Pi0Adapter.predict: {reason}")
        observation = self._observation_builder(scene)
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        # openpi returns an action chunk (chunk_len, action_dim). Expose
        # the full chunk via action_chunk so ChunkConsistencyDiagnostic
        # can test chunk[t][1] vs chunk[t+1][0] coherence — that's the
        # actual chunk-planning quality test, not just adjacent-frame
        # single-step delta.
        if actions.ndim >= 2:
            chunk = actions if actions.ndim == 2 else actions.reshape(-1, actions.shape[-1])
            action = chunk[0]
        else:
            chunk = actions[np.newaxis, :]
            action = actions
        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "config_name": self.config_name,
                "chunk_shape": list(actions.shape),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []
