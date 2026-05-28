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

        def _norm(p: str) -> str:
            return p if p.startswith("/") else "/" + p

        # Rerun's `contents` arg accepts a comma-separated string of entity
        # path expressions (e.g. "/cameras/primary/image, /state").
        image_paths = {cam: _norm(p) for cam, p in self.image_entities.items()}
        wanted_paths = list(image_paths.values())
        state_path = _norm(self.state_entity) if self.state_entity else None
        gripper_path = _norm(self.gripper_entity) if self.gripper_entity else None
        if state_path:
            wanted_paths.append(state_path)
        if gripper_path:
            wanted_paths.append(gripper_path)
        contents_expr = ", ".join(wanted_paths)

        view = recording.view(index="log_time", contents=contents_expr).fill_latest_at()
        rows: list[dict] = view.select().read_all().to_pylist()

        primary_key = image_paths.get("primary") or next(iter(image_paths.values()))

        def _find_component(row: dict, path: str, component: str) -> Optional[object]:
            """Look up row[`{path}:{component}`] tolerantly."""
            target = f"{path}:{component}"
            for k, v in row.items():
                if k == target:
                    return v
            return None

        scenes: list[Scene] = []
        for i, row in enumerate(rows):
            primary_img = self._extract_image_pair(row, primary_key)
            if primary_img is None:
                continue

            images: dict[str, RGBImage] = {}
            for cam_name, path in image_paths.items():
                img = self._extract_image_pair(row, path)
                if img is not None:
                    images[cam_name] = RGBImage(data=img, camera_id=cam_name)
            if "primary" not in images:
                images["primary"] = RGBImage(data=primary_img, camera_id="primary")

            state = None
            if state_path:
                # Rerun's Scalar / Vector components don't have a fixed name
                # — try a few common ones the writer might have used.
                vals = None
                for comp in ("Scalar", "Float32Array", "Vector"):
                    vals = self._extract_vector(_find_component(row, state_path, comp))
                    if vals is not None:
                        break
                if vals is not None and self.profile.state is not None:
                    state = Proprioception(values=vals, convention=self.profile.state.convention)

            gripper = None
            if gripper_path:
                val = self._extract_scalar(_find_component(row, gripper_path, "Scalar"))
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

    def _extract_image_pair(self, row: dict, path: str) -> Optional[Image.Image]:
        """Reconstruct a PIL image from Rerun's ImageBuffer + ImageFormat pair.

        Rerun stores images as two components per entity:
          - ImageBuffer: flat uint8 bytes
          - ImageFormat: dict-like with width / height / channel info
        We grab both, decode the format, and reshape the buffer.
        """
        buf = None
        fmt = None
        for k, v in row.items():
            if k.startswith(path + ":"):
                if k.endswith("ImageBuffer"):
                    buf = v
                elif k.endswith("ImageFormat"):
                    fmt = v
        if buf is None:
            return None
        arr = np.asarray(buf).reshape(-1).astype(np.uint8)
        # Try to read width/height from the format component; fall back to
        # square uint8 RGB if format is missing.
        width, height, channels = self._parse_image_format(fmt)
        if width is None or height is None:
            side = int(np.sqrt(arr.size // 3))
            width = height = side
            channels = 3
        if channels == 1:
            return Image.fromarray(arr.reshape(height, width))
        return Image.fromarray(arr.reshape(height, width, channels))

    def _parse_image_format(self, fmt) -> tuple[Optional[int], Optional[int], int]:
        """Pull width/height/channels out of Rerun's ImageFormat component.

        ImageFormat is delivered as a python dict-or-struct; we extract
        common field names defensively.
        """
        if fmt is None:
            return None, None, 3
        # ImageFormat is often wrapped in a one-element list/array.
        if isinstance(fmt, (list, tuple, np.ndarray)) and len(fmt) > 0:
            fmt = fmt[0]
        width = None
        height = None
        channels = 3
        if hasattr(fmt, "width"):
            width = int(fmt.width)
            height = int(fmt.height)
        elif isinstance(fmt, dict):
            width = int(fmt.get("width", fmt.get("size", [0, 0])[1]) or 0) or None
            height = int(fmt.get("height", fmt.get("size", [0, 0])[0]) or 0) or None
            color_model = fmt.get("color_model") or fmt.get("colorModel")
            if color_model in ("L", "Grayscale", 0, "0"):
                channels = 1
        return width, height, channels

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
