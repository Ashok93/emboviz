"""Foxglove `.mcap` episode source.

Ingests rosbag2 / Foxglove recordings (mcap container) as Scenes. Like
Rerun, mcap is open-schema — each team logs to different topics — so the
adapter is configured with explicit topic mappings.

Lazy imports the `mcap` package and decoders (`mcap-ros2-support` or
`mcap-protobuf-support`) so this module is free to import without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import GripperState, Proprioception, RGBImage
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


# Decode one mcap message payload into a PIL image / numpy vector / scalar.
# Provided by the caller because the encoding (CompressedImage, sensor_msgs/Image,
# protobuf, etc.) is team-specific.
ImageDecoder = Callable[[Any], Optional[Image.Image]]
VectorDecoder = Callable[[Any], Optional[np.ndarray]]
ScalarDecoder = Callable[[Any], Optional[float]]


class FoxgloveEpisodeSource(EpisodeSource):
    """Ingest a Foxglove/rosbag2 `.mcap` file as a single-episode trajectory.

    Required:
      • `mcap_path`     — path to the .mcap file
      • `profile`       — RobotProfile for the recorded robot
      • `image_topics`  — {camera_name → mcap topic name}
      • `image_decoder` — fn(payload) → PIL.Image (team-supplied; sensor_msgs/Image,
                           CompressedImage, or whatever they log)

    Optional:
      • `state_topic`, `state_decoder`     — proprioception
      • `gripper_topic`, `gripper_decoder` — gripper value
      • `instruction`                       — fixed string per recording
    """

    def __init__(
        self,
        mcap_path: str,
        profile: RobotProfile,
        image_topics: dict[str, str],
        image_decoder: ImageDecoder,
        *,
        state_topic: Optional[str] = None,
        state_decoder: Optional[VectorDecoder] = None,
        gripper_topic: Optional[str] = None,
        gripper_decoder: Optional[ScalarDecoder] = None,
        instruction: Optional[str] = None,
    ):
        if not image_topics:
            raise ValueError("image_topics must have at least one entry")
        self.mcap_path = Path(mcap_path)
        self.profile = profile
        self.image_topics = dict(image_topics)
        self.image_decoder = image_decoder
        self.state_topic = state_topic
        self.state_decoder = state_decoder
        self.gripper_topic = gripper_topic
        self.gripper_decoder = gripper_decoder
        self.instruction = instruction
        self.name = f"foxglove:{self.mcap_path.name}"

    def list_episodes(self) -> list[str]:
        return ["0"]

    def load_episode(self, episode_id: str) -> list[Scene]:
        try:
            from mcap.reader import make_reader
        except ImportError as e:
            raise ImportError(
                "Loading mcap recordings requires the `mcap` package. "
                "Install with: uv add mcap"
            ) from e

        # Topic → list of (timestamp_ns, decoded payload)
        topic_streams: dict[str, list[tuple[int, Any]]] = {}
        all_topics = set(self.image_topics.values())
        if self.state_topic:
            all_topics.add(self.state_topic)
        if self.gripper_topic:
            all_topics.add(self.gripper_topic)

        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=list(all_topics)):
                topic_streams.setdefault(channel.topic, []).append(
                    (message.log_time, message.data)
                )

        # Use the primary camera's stream to define frame timing.
        primary_topic = (
            self.image_topics.get("primary")
            or next(iter(self.image_topics.values()))
        )
        primary_stream = topic_streams.get(primary_topic, [])

        scenes: list[Scene] = []
        for i, (ts, payload) in enumerate(primary_stream):
            primary_img = self.image_decoder(payload)
            if primary_img is None:
                continue

            images: dict[str, RGBImage] = {}
            for cam_name, topic in self.image_topics.items():
                # Find the nearest message in this topic at or before ts
                pl = _nearest_at_or_before(topic_streams.get(topic, []), ts)
                if pl is None:
                    continue
                img = self.image_decoder(pl)
                if img is not None:
                    images[cam_name] = RGBImage(data=img, camera_id=cam_name)
            if "primary" not in images:
                images["primary"] = RGBImage(data=primary_img, camera_id="primary")

            state = None
            if self.state_topic and self.state_decoder:
                pl = _nearest_at_or_before(topic_streams.get(self.state_topic, []), ts)
                vals = self.state_decoder(pl) if pl is not None else None
                if vals is not None and self.profile.state is not None:
                    state = Proprioception(values=vals, convention=self.profile.state.convention)

            gripper = None
            if self.gripper_topic and self.gripper_decoder:
                pl = _nearest_at_or_before(topic_streams.get(self.gripper_topic, []), ts)
                val = self.gripper_decoder(pl) if pl is not None else None
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
                metadata={"frame_index": i, "log_time_ns": int(ts), "source": self.name},
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
            metadata={"mcap_path": str(self.mcap_path)},
        )

    def all_instructions(self) -> list[str]:
        return [self.instruction] if self.instruction else []


def _nearest_at_or_before(stream: list[tuple[int, Any]], ts: int) -> Optional[Any]:
    """Return the latest payload in `stream` whose timestamp ≤ ts."""
    if not stream:
        return None
    best = None
    for t, payload in stream:
        if t > ts:
            break
        best = payload
    return best
