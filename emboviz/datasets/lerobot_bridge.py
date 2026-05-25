"""BridgeV2 (IPEC-COMMUNITY/bridge_orig_lerobot) episode source.

Bridge has 53k episodes; we lazy-load. `load_episode` accepts integer IDs
and uses batched LeRobotDataset() calls when many episodes are requested.

The raw Bridge state vector is `[x, y, z, roll, pitch, yaw, gripper]` —
6-DOF end-effector pose plus a normalized [0, 1] gripper value. We
unpack that into typed Observations (Proprioception + GripperState) and
attach a `BRIDGE_PROFILE` so perturbers and diagnostics know the layout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from emboviz.core.observations import (
    GripperState,
    Proprioception,
    RGBImage,
)
from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
PRIMARY_IMAGE_KEY = "observation.images.image_0"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


# Default RobotProfile for Bridge rollouts. Attached to every Scene loaded
# from this source so downstream perturbers / diagnostics know how to
# interpret state, gripper, and action.
BRIDGE_PROFILE = RobotProfile(
    name="bridge_orig",
    cameras=[CameraSpec(name="primary")],
    state=StateSpec(
        dim=6,
        convention="ee_pose",
        joint_names=["x", "y", "z", "roll", "pitch", "yaw"],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(
        dim=7,
        dim_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    ),
)


class BridgeEpisodeSource(EpisodeSource):
    """Episode source backed by LeRobotDataset on the IPEC bridge mirror."""

    name = "bridge_v2"

    def __init__(self):
        self._meta_dataset = None  # populated on first call to all_instructions

    def list_episodes(self) -> list[str]:
        # Bridge has 53k. We don't enumerate by default.
        return [str(i) for i in range(53192)]

    def load_episode(self, episode_id: str) -> list[Scene]:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Batched load — single LeRobotDataset init for all indices."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        indices = sorted(set(episode_indices))
        dataset = LeRobotDataset(DATASET_REPO, episodes=indices)
        out: dict[int, list[Scene]] = {i: [] for i in indices}

        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", sample.get("episode_idx", indices[0])))
            if ep_i not in out:
                continue
            instruction = self._resolve_instruction(dataset, ep_i)
            scene = self._build_scene(sample, instruction, ep_i, len(out[ep_i]), dataset.fps)
            out[ep_i].append(scene)
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        """Load one episode as a Trajectory (Scenes in time order)."""
        scenes = self.load_episode(str(episode_idx))
        # FPS comes from the metadata; pull it from any scene that has it.
        fps = float(scenes[0].metadata.get("fps", 5.0)) if scenes else 5.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"bridge:{episode_idx}",
            metadata={"dataset": DATASET_REPO},
        )

    def all_instructions(self) -> list[str]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if self._meta_dataset is None:
            self._meta_dataset = LeRobotDataset(DATASET_REPO, episodes=[0])
        tasks = getattr(self._meta_dataset.meta, "tasks", None)
        if tasks is None:
            return []
        if isinstance(tasks, dict):
            items = list(tasks.values())
        else:
            items = list(tasks)
        out = []
        for it in items:
            if isinstance(it, dict) and "task" in it:
                out.append(str(it["task"]))
            elif isinstance(it, str):
                out.append(it)
        return out

    # ---- helpers ----------------------------------------------------------

    def _build_scene(
        self, sample: dict, instruction: str,
        episode_idx: int, frame_offset: int, fps: float,
    ) -> Scene:
        """Build a typed Scene from one raw Bridge sample."""
        image = self._tensor_to_pil(sample[PRIMARY_IMAGE_KEY])
        state_vals = sample[STATE_KEY].to(torch.float32).reshape(-1).numpy()

        # Bridge state layout: [x, y, z, roll, pitch, yaw, gripper]
        gripper_val = float(state_vals[6]) if state_vals.size >= 7 else 0.0
        proprio_vals = state_vals[:6].copy() if state_vals.size >= 6 else state_vals.copy()

        obs = Observations(
            images={"primary": RGBImage(data=image, camera_id="primary")},
            state=Proprioception(values=proprio_vals, convention="ee_pose"),
            gripper=GripperState(value=gripper_val, kind="parallel_jaw", units="unit"),
        )

        return Scene(
            observations=obs,
            instruction=instruction,
            profile=BRIDGE_PROFILE,
            metadata={
                "expert_action": sample[ACTION_KEY].to(torch.float32).reshape(-1).tolist(),
                "fps": float(fps),
                "frame_index": int(sample.get("frame_index", frame_offset)),
                "episode_index": episode_idx,
                "dataset": DATASET_REPO,
                "raw_state": state_vals.tolist(),
            },
            scene_id=f"bridge:{episode_idx}:{frame_offset}",
        )

    def _tensor_to_pil(self, t) -> Image.Image:
        a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t)
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = a.transpose(1, 2, 0)
        if a.max() <= 1.5:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _resolve_instruction(self, dataset, episode_idx: int) -> str:
        meta = dataset.meta
        tasks = getattr(meta, "tasks", None)
        target_idx: Optional[int] = None
        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", sample.get("episode_idx", -1)))
            if ep_i == episode_idx and "task_index" in sample:
                target_idx = int(sample["task_index"])
                break
        if target_idx is None or tasks is None:
            return f"(task #{target_idx})" if target_idx is not None else "(no instruction)"
        if isinstance(tasks, dict):
            return str(tasks.get(target_idx, f"(task #{target_idx})"))
        if isinstance(tasks, (list, tuple)) and target_idx < len(tasks):
            entry = tasks[target_idx]
            return entry["task"] if isinstance(entry, dict) else str(entry)
        return f"(task #{target_idx})"
