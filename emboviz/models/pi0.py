"""Adapter for Physical Intelligence's π0 / π0.5 / π0-FAST via openpi.

The `openpi` repository (https://github.com/Physical-Intelligence/openpi)
is PI's official open-source inference path for the π0 family. Each
checkpoint is paired with a platform-specific observation format (DROID,
ALOHA, LIBERO, UR5, custom). This adapter wraps openpi's `create_trained_policy`
and maps our typed `Scene` into the format the chosen config expects.

**Install (its own virtualenv):**

    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    uv pip install --no-deps -e /path/to/emboviz   # add emboviz on top

Then construct with a config name (e.g. "pi0_aloha_sim", "pi0_libero",
"pi0_fast_droid", "pi05_libero", "pi05_droid").

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


def _aloha_observation_builder(scene: Scene) -> dict:
    """openpi ALOHA observation format — bimanual, 4 cameras, state(14)."""
    images = scene.observations.images
    primary = images.get("primary")
    high = images.get("cam_high") or images.get("head") or primary
    low = images.get("cam_low") or images.get("front") or primary
    wrist_l = images.get("wrist_left") or images.get("cam_left_wrist") or primary
    wrist_r = images.get("wrist_right") or images.get("cam_right_wrist") or primary
    state = (
        scene.observations.state.values.astype(np.float32)
        if scene.observations.state is not None
        else np.zeros(14, dtype=np.float32)
    )
    if state.size < 14:
        state = np.pad(state, (0, 14 - state.size))
    else:
        state = state[:14]
    zeros = np.zeros((3, 224, 224), dtype=np.uint8)
    return {
        "state": state,
        "images": {
            "cam_high": _to_chw_uint8(high.data) if high else zeros,
            "cam_low": _to_chw_uint8(low.data) if low else zeros,
            "cam_left_wrist": _to_chw_uint8(wrist_l.data) if wrist_l else zeros,
            "cam_right_wrist": _to_chw_uint8(wrist_r.data) if wrist_r else zeros,
        },
        "prompt": scene.instruction or "",
    }


def _droid_observation_builder(scene: Scene) -> dict:
    """openpi DROID observation format — single arm + wrist + state(7)+gripper."""
    images = scene.observations.images
    primary = images.get("primary")
    wrist = images.get("wrist_left") or images.get("wrist") or primary
    obs: dict = {}
    if primary is not None:
        obs["observation/exterior_image_1_left"] = np.asarray(primary.data, dtype=np.uint8)
    if wrist is not None:
        obs["observation/wrist_image_left"] = np.asarray(wrist.data, dtype=np.uint8)
    if scene.observations.state is not None:
        obs["observation/joint_position"] = scene.observations.state.values.astype(np.float32)
    if scene.observations.gripper is not None:
        obs["observation/gripper_position"] = np.array(
            [scene.observations.gripper.value], dtype=np.float32,
        )
    obs["prompt"] = scene.instruction or ""
    return obs


def _libero_observation_builder(scene: Scene) -> dict:
    """openpi LIBERO observation format — 2 cameras + state(8)."""
    images = scene.observations.images
    primary = images.get("primary")
    wrist = images.get("wrist") or primary
    state = (
        scene.observations.state.values.astype(np.float32)
        if scene.observations.state is not None
        else np.zeros(8, dtype=np.float32)
    )
    if state.size < 8:
        state = np.pad(state, (0, 8 - state.size))
    return {
        "observation/image": _to_chw_uint8(primary.data) if primary else np.zeros((3, 256, 256), dtype=np.uint8),
        "observation/wrist_image": _to_chw_uint8(wrist.data) if wrist else np.zeros((3, 256, 256), dtype=np.uint8),
        "observation/state": state[:8],
        "prompt": scene.instruction or "",
    }


# Built-in observation builders keyed by config-name fragment.
_BUILDER_REGISTRY: dict[str, Callable[[Scene], dict]] = {
    "aloha": _aloha_observation_builder,
    "droid": _droid_observation_builder,
    "libero": _libero_observation_builder,
}


def _auto_observation_builder(config_name: str) -> Callable[[Scene], dict]:
    """Pick the right builder based on substring match in config_name."""
    for key, builder in _BUILDER_REGISTRY.items():
        if key in config_name.lower():
            return builder
    return _droid_observation_builder   # DROID is the most generic default


@register_model("pi0")
@register_model("pi05")
class Pi0Adapter(VLAModel):
    """Wraps `openpi`'s trained policy as a VLAModel.

    Construction:
        Pi0Adapter()                                        # pi0_fast_droid default
        Pi0Adapter(config_name="pi0_libero")
        Pi0Adapter(config_name="pi05_droid",
                   observation_builder=my_custom_builder)

    `observation_builder` defaults to a DROID-style dict; override for
    ALOHA or LIBERO setups (see openpi.policies.aloha_policy /
    openpi.policies.libero_policy for the exact dict shapes).
    """

    _CAPS = Capability.INFERENCE

    def __init__(
        self,
        config_name: str = "pi0_fast_droid",
        checkpoint_uri: Optional[str] = None,
        observation_builder: Optional[Callable[[Scene], dict]] = None,
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
        # Auto-select the platform-specific observation builder.
        self._observation_builder = (
            observation_builder or _auto_observation_builder(config_name)
        )
        # Populated on first predict.
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
        # π0's DROID/LIBERO/ALOHA configs canonically expect a primary cam
        # plus state + instruction. Multi-cam configs additionally need
        # wrist_left/wrist_right — the observation_builder handles them.
        return RequiredInputs(
            cameras=frozenset({"primary"}),
            instruction=True,
            state=True,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        observation = self._observation_builder(scene)
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        # openpi returns an action chunk; take the first immediate action.
        if actions.ndim >= 2:
            action = actions[0]
        else:
            action = actions
        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            metadata={
                "config_name": self.config_name,
                "chunk_shape": list(actions.shape),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []
