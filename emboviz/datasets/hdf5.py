"""HDF5 episode source — for Robomimic / ALOHA / NVIDIA Isaac Lab data.

The legacy-but-widespread format for robot demonstration data. A single
``.h5`` file holds N episodes under the ``data/`` group:

    data/
      demo_0/
        obs/
          agentview_image          (T, H, W, 3) uint8
          robot0_eye_in_hand_image (T, H, W, 3) uint8
          robot0_eef_pos           (T, 3)       float
          robot0_eef_quat          (T, 4)       float
          robot0_gripper_qpos      (T, 2)       float
          ...
        actions                    (T, 7)       float
        states                     (T, ...)     (mujoco-specific; optional)
        num_samples                scalar
        model_file                 string       (optional; mujoco mjcf)
      demo_1/
        ...

Robomimic, ALOHA, MimicGen, RoboCasa, and most Isaac Lab Mimic outputs
follow this convention. Different teams put images under different
sub-keys, so the adapter takes explicit ``camera_keys`` and ``state_key``
mappings and refuses to guess.

Install:
  pip install 'emboviz[hdf5]'

Reference: Robomimic docs at
https://robomimic.github.io/docs/datasets/overview.html ;
Isaac Lab Mimic uses the same schema.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import (
    GripperState,
    Proprioception,
    RGBImage,
)
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


GripperExtractor = Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]


def _identity_state(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    return state, None


class HDF5EpisodeSource(EpisodeSource):
    """Robomimic / ALOHA / Isaac-Lab-Mimic HDF5 episode source.

    Args:
      path: filesystem path to the .h5 / .hdf5 file. Multiple files
        are NOT supported in one source — use one HDF5EpisodeSource
        per file. (Most robomimic-style datasets ship a single file
        with all demos.)
      camera_keys: ``{scene_camera_name → h5 key relative to demo group}``
        map. The h5 key includes the ``obs/`` prefix if applicable:
        e.g. ``{"primary": "obs/agentview_image",
                "wrist":   "obs/robot0_eye_in_hand_image"}``.
        Must include an entry named ``"primary"``.
      state_key: h5 key (relative to demo group) for the proprio vector,
        e.g. ``"obs/robot0_eef_pos"``. ``None`` = no state.
      action_key: relative key for the action sequence, default ``"actions"``.
      instruction: optional episode-wide instruction string. HDF5
        schemas typically lack a recorded instruction (the prompt was
        out-of-band); pass it explicitly so the model receives it.
      instruction_attr: optional name of an h5 attribute (on the demo
        group or the file root) that holds the instruction. Checked
        when ``instruction`` is None.
      demo_group: top-level group name (default ``"data"``). Some
        datasets nest under a different group.
      profile: RobotProfile (state + action conventions).
      gripper_extractor: same as the LeRobot adapter — split state
        into (proprio, gripper_value).
    """

    def __init__(
        self,
        path: str,
        *,
        camera_keys: dict[str, str],
        state_key: Optional[str] = None,
        action_key: str = "actions",
        instruction: Optional[str] = None,
        instruction_attr: Optional[str] = None,
        demo_group: str = "data",
        profile: Optional[RobotProfile] = None,
        gripper_extractor: GripperExtractor = _identity_state,
    ):
        if not camera_keys:
            raise ValueError(
                "HDF5EpisodeSource: camera_keys is required. HDF5 datasets "
                "store images under arbitrary sub-keys (agentview_image vs "
                "image vs camera_0); we never guess."
            )
        if "primary" not in camera_keys:
            raise KeyError(
                f"HDF5EpisodeSource: camera_keys must contain a 'primary' "
                f"entry. Got {sorted(camera_keys)}. The adapter caller must "
                "say which view is the primary (exterior) camera."
            )
        self.path             = path
        self.camera_keys      = dict(camera_keys)
        self.state_key        = state_key
        self.action_key       = action_key
        self.instruction      = instruction
        self.instruction_attr = instruction_attr
        self.demo_group       = demo_group
        self.profile          = profile
        self.gripper_extractor = gripper_extractor
        self.name             = f"hdf5:{path}"

        # We do NOT open the file at construction time — open lazily on
        # first read so the constructor is cheap.
        self._file_cache = None
        self._demo_names_cache: Optional[list[str]] = None

    # ── EpisodeSource interface ────────────────────────────────────

    def list_episodes(self) -> list[str]:
        """Demo group names in sorted order (demo_0, demo_1, ...).

        Returns the demo names AS strings so the integer-indexed
        ``load_trajectory`` works by listing index. Use the canonical
        Robomimic naming (``demo_N``) for predictable ordering.
        """
        return self._demo_names()

    def load_episode(self, episode_id: str) -> list[Scene]:
        names = self._demo_names()
        if episode_id in names:
            demo_name = episode_id
        else:
            try:
                demo_name = names[int(episode_id)]
            except (ValueError, IndexError) as e:
                raise IndexError(
                    f"HDF5EpisodeSource: '{episode_id}' is neither a demo "
                    f"name nor a valid index in {len(names)} demos. "
                    f"First five names: {names[:5]}"
                ) from e
        return self._demo_to_scenes(demo_name)

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        names = self._demo_names()
        out: dict[int, list[Scene]] = {}
        for i in episode_indices:
            try:
                demo_name = names[i]
            except IndexError as e:
                raise IndexError(
                    f"episode {i} out of range (have {len(names)} demos)"
                ) from e
            out[i] = self._demo_to_scenes(demo_name)
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        fps = float(scenes[0].metadata.get("fps", 20.0)) if scenes else 20.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.path},
        )

    def all_instructions(self) -> list[str]:
        """Unique per-demo instructions across the file.

        HDF5 schemas typically lack a recorded per-step instruction;
        we return the file-level / demo-level attribute when present,
        or the single ``instruction`` passed to __init__. Returns ``[]``
        if neither source has an instruction.
        """
        out: set[str] = set()
        if self.instruction:
            out.add(self.instruction)
        if self.instruction_attr:
            f = self._h5_file()
            if self.instruction_attr in f.attrs:
                out.add(_bytes_to_str(f.attrs[self.instruction_attr]))
            for name in self._demo_names():
                demo = f[self.demo_group][name]
                if self.instruction_attr in demo.attrs:
                    out.add(_bytes_to_str(demo.attrs[self.instruction_attr]))
        return sorted(out)

    # ── internals ──────────────────────────────────────────────────

    def _h5_file(self):
        if self._file_cache is None:
            try:
                import h5py
            except ImportError as e:
                raise ImportError(
                    "HDF5EpisodeSource needs the `hdf5` extra. Install with: "
                    "pip install 'emboviz[hdf5]'. Underlying error: " + str(e)
                ) from e
            self._file_cache = h5py.File(self.path, "r")
        return self._file_cache

    def _demo_names(self) -> list[str]:
        if self._demo_names_cache is not None:
            return self._demo_names_cache
        f = self._h5_file()
        if self.demo_group not in f:
            raise KeyError(
                f"HDF5EpisodeSource: file '{self.path}' has no top-level "
                f"group named '{self.demo_group}'. Available groups: "
                f"{list(f.keys())}. Pass ``demo_group=...`` if your dataset "
                f"uses a non-default group name."
            )
        group = f[self.demo_group]
        # Sort by numeric suffix when possible (demo_0, demo_1, ..., demo_10).
        def _key(name: str) -> tuple:
            parts = name.rsplit("_", 1)
            try:
                return (0, int(parts[-1]))
            except (ValueError, IndexError):
                return (1, name)
        names = sorted(group.keys(), key=_key)
        self._demo_names_cache = names
        return names

    def _demo_to_scenes(self, demo_name: str) -> list[Scene]:
        f = self._h5_file()
        demo = f[self.demo_group][demo_name]

        # Resolve instruction (per-demo attribute > per-file attribute > __init__ override).
        instr = self.instruction or ""
        if not instr and self.instruction_attr:
            if self.instruction_attr in demo.attrs:
                instr = _bytes_to_str(demo.attrs[self.instruction_attr])
            elif self.instruction_attr in f.attrs:
                instr = _bytes_to_str(f.attrs[self.instruction_attr])

        # Pull all camera streams + state + actions as full arrays.
        # Robomimic stores (T, ...) so we slice per-frame in the loop.
        cams_data: dict[str, np.ndarray] = {}
        n_frames: Optional[int] = None
        for scene_cam, h5_key in self.camera_keys.items():
            if h5_key not in demo:
                raise KeyError(
                    f"HDF5EpisodeSource: camera '{scene_cam}' maps to "
                    f"h5 key '{h5_key}' but demo '{demo_name}' has no "
                    f"such key. Available top-level demo keys: "
                    f"{list(demo.keys())}; obs keys: "
                    f"{list((demo.get('obs') or {}).keys()) if 'obs' in demo else 'N/A'}"
                )
            arr = demo[h5_key][:]
            if n_frames is None:
                n_frames = arr.shape[0]
            elif n_frames != arr.shape[0]:
                raise ValueError(
                    f"camera '{scene_cam}' has {arr.shape[0]} frames but "
                    f"the demo's other cameras have {n_frames}."
                )
            cams_data[scene_cam] = arr

        state_arr: Optional[np.ndarray] = None
        if self.state_key is not None:
            if self.state_key not in demo:
                raise KeyError(
                    f"state_key '{self.state_key}' not in demo '{demo_name}'"
                )
            state_arr = demo[self.state_key][:]

        action_arr: Optional[np.ndarray] = None
        if self.action_key in demo:
            action_arr = demo[self.action_key][:]

        scenes: list[Scene] = []
        for fi in range(int(n_frames or 0)):
            images: dict[str, RGBImage] = {}
            for cam_name, arr in cams_data.items():
                pil = Image.fromarray(np.asarray(arr[fi], dtype=np.uint8))
                images[cam_name] = RGBImage(data=pil, camera_id=cam_name)

            proprio: Optional[Proprioception] = None
            gripper: Optional[GripperState] = None
            raw_state = None
            if state_arr is not None:
                raw_state = np.asarray(state_arr[fi], dtype=np.float32).reshape(-1)
                proprio_vals, gripper_val = self.gripper_extractor(raw_state)
                state_convention = (
                    self.profile.state.convention if self.profile and self.profile.state
                    else "joint_angles"
                )
                proprio = Proprioception(values=proprio_vals.copy(), convention=state_convention)
                if gripper_val is not None and self.profile and self.profile.gripper:
                    gripper = GripperState(
                        value=float(gripper_val),
                        kind=self.profile.gripper.kind,
                        units=self.profile.gripper.units,
                    )

            metadata: dict = {
                "fps":           20.0,
                "frame_index":   fi,
                "episode_index": demo_name,
                "dataset":       self.path,
            }
            if raw_state is not None:
                metadata["raw_state"] = raw_state.tolist()
            if action_arr is not None:
                metadata["expert_action"] = (
                    np.asarray(action_arr[fi], dtype=np.float32).reshape(-1).tolist()
                )

            scenes.append(Scene(
                observations=Observations(images=images, state=proprio, gripper=gripper),
                instruction=instr,
                profile=self.profile,
                metadata=metadata,
                scene_id=f"{self.name}:{demo_name}:{fi}",
            ))
        return scenes


def _bytes_to_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U", "O"):
        if v.size == 0:
            return ""
        item = v.item()
        return _bytes_to_str(item) if not isinstance(item, str) else item
    return str(v) if v is not None else ""
