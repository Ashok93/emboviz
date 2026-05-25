"""Rerun `.rrd` episode source.

Ingests recordings produced by Rerun (rerun.io) and emits typed Scenes.
Rerun's schema is open — each team logs to different entity paths — so
this adapter is configured with an explicit mapping from entity paths to
modalities. Most teams need ~5 lines of config to get going.

Lazy imports `rerun-sdk` so this module is free to import without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import GripperState, Proprioception, RGBImage
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


class RerunEpisodeSource(EpisodeSource):
    """Ingest a Rerun `.rrd` recording as a single-episode trajectory.

    Required:
      • `rrd_path`        — path to the recording
      • `profile`         — RobotProfile for the recorded robot
      • `image_entities`  — {camera_name → Rerun entity path of the image
                             component, e.g. "world/camera/image"}

    Optional:
      • `state_entity`    — entity path of a vector logged as state
      • `gripper_entity`  — entity path of a scalar logged as gripper value
      • `instruction`     — fixed instruction string for this recording
                             (Rerun doesn't carry instructions natively)
    """

    def __init__(
        self,
        rrd_path: str,
        profile: RobotProfile,
        image_entities: dict[str, str],
        *,
        state_entity: Optional[str] = None,
        gripper_entity: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        if not image_entities:
            raise ValueError("image_entities must have at least one entry")
        self.rrd_path = Path(rrd_path)
        self.profile = profile
        self.image_entities = dict(image_entities)
        self.state_entity = state_entity
        self.gripper_entity = gripper_entity
        self.instruction = instruction
        self.name = f"rerun:{self.rrd_path.name}"

    def list_episodes(self) -> list[str]:
        return ["0"]

    def load_episode(self, episode_id: str) -> list[Scene]:
        try:
            import rerun.dataframe as rrdf
        except ImportError as e:
            raise ImportError(
                "Loading Rerun .rrd recordings requires the `rerun-sdk` package. "
                "Install with: uv add rerun-sdk"
            ) from e

        recording = rrdf.load_recording(str(self.rrd_path))
        # Collect time-aligned rows across the entities we care about.
        wanted = list(self.image_entities.values())
        if self.state_entity:
            wanted.append(self.state_entity)
        if self.gripper_entity:
            wanted.append(self.gripper_entity)

        view = recording.view(index="log_time", contents=wanted)
        df = view.select().read_all().to_pylist()

        # Group rows by primary camera's timestamp; one Scene per row.
        primary_key = (
            self.image_entities.get("primary")
            or next(iter(self.image_entities.values()))
        )
        scenes: list[Scene] = []
        for i, row in enumerate(df):
            primary_img = self._extract_image(row.get(primary_key))
            if primary_img is None:
                continue

            images: dict[str, RGBImage] = {}
            for cam_name, entity in self.image_entities.items():
                img = self._extract_image(row.get(entity))
                if img is not None:
                    images[cam_name] = RGBImage(data=img, camera_id=cam_name)
            if "primary" not in images:
                images["primary"] = RGBImage(data=primary_img, camera_id="primary")

            state = None
            if self.state_entity:
                vals = self._extract_vector(row.get(self.state_entity))
                if vals is not None and self.profile.state is not None:
                    state = Proprioception(values=vals, convention=self.profile.state.convention)

            gripper = None
            if self.gripper_entity:
                val = self._extract_scalar(row.get(self.gripper_entity))
                if val is not None and self.profile.gripper is not None:
                    gripper = GripperState(
                        value=float(val),
                        kind=self.profile.gripper.kind,
                        units=self.profile.gripper.units,
                    )

            obs = Observations(images=images, state=state, gripper=gripper)
            scenes.append(Scene(
                observations=obs,
                instruction=self.instruction,
                profile=self.profile,
                metadata={"frame_index": i, "source": self.name},
                scene_id=f"{self.name}:0:{i}",
            ))
        return scenes

    def load_trajectory(self, episode_idx: int = 0) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            episode_id=str(episode_idx),
            source=self.name,
            metadata={"rrd_path": str(self.rrd_path)},
        )

    def all_instructions(self) -> list[str]:
        return [self.instruction] if self.instruction else []

    # ----- helpers ---------------------------------------------------

    def _extract_image(self, cell) -> Optional[Image.Image]:
        """Convert a Rerun image cell to a PIL.Image."""
        if cell is None:
            return None
        arr = np.asarray(cell)
        # Rerun image components are typically (H, W, 3) or (H, W) uint8
        if arr.ndim == 0:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _extract_vector(self, cell) -> Optional[np.ndarray]:
        if cell is None:
            return None
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
        return arr if arr.size > 0 else None

    def _extract_scalar(self, cell) -> Optional[float]:
        if cell is None:
            return None
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
        return float(arr[0]) if arr.size > 0 else None
