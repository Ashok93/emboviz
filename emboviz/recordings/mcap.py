"""MCAP deployment-recording adapter.

Reads a ``.mcap`` file (or directory of them) recorded from a real-
robot rollout. MCAP is the default container in ROS 2 Iron+ and
NVIDIA Isaac SIM, so most 2026-era deployment logs are MCAP.

Auto-decodes the common ROS 2 message types:
  • ``sensor_msgs/msg/Image``           → PIL.Image (rgb8/bgr8/mono8 supported)
  • ``sensor_msgs/msg/CompressedImage`` → PIL.Image (JPEG/PNG)
  • ``sensor_msgs/msg/JointState``      → np.ndarray (position vector)
  • ``geometry_msgs/msg/PoseStamped``   → np.ndarray (xyz + xyzw quaternion)
  • ``std_msgs/msg/String``             → str (used for instruction)
  • ``std_msgs/msg/Float32MultiArray``  → np.ndarray (used for policy action)

The user supplies a ``topic_map`` saying which ROS topic carries each
of {primary camera, wrist camera, state, action, instruction}. Topics
not in the map are ignored. Topic types we don't know how to decode
are skipped with a warning.

Time alignment: messages on different topics arrive at different rates
(cameras 30 Hz, joint_states 100 Hz, policy 10 Hz). We resample at a
configurable ``target_rate_hz`` using sample-and-hold: at each frame
timestamp, take the most-recent value from each topic.

Install:
  pip install 'emboviz[mcap]'   # pulls mcap + mcap-ros2-support

This adapter is DEPLOYMENT-recording aware: every emitted Scene has
``metadata["has_recorded_expert_action"] = False`` so the runner's
``--show-imitation`` gate correctly suppresses BC validation metrics
(there is no expert demonstrator in a deployment rollout).
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import GripperState, Proprioception, RGBImage
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


class MCAPRecording(EpisodeSource):
    """One or many .mcap deployment recordings.

    Args:
      path: path to a single .mcap file OR a directory containing
        multiple .mcap files. When a directory, each file becomes one
        episode (sorted by filename).
      topic_map: which ROS topic carries each scene field. Keys:
        ``"primary"``, ``"wrist"``, etc. (image cameras),
        ``"state"`` (proprioception, e.g. /joint_states),
        ``"action"`` (model's predicted action — std_msgs/Float32MultiArray
        or geometry_msgs/Twist),
        ``"instruction"`` (std_msgs/String).
        Example:
          {"primary": "/camera_front/image",
           "wrist":   "/camera_wrist/image",
           "state":   "/joint_states",
           "action":  "/policy/action",
           "instruction": "/policy/instruction"}
      target_rate_hz: frame rate at which to resample the multi-topic
        streams into Scenes. Default 10 Hz (typical VLA prediction rate).
      profile: RobotProfile describing the robot's state + action
        conventions. Required for diagnostics that consult the profile.
      static_instruction: fallback instruction if the recording doesn't
        publish one on a topic. Useful when the operator typed the prompt
        out-of-band.
    """

    def __init__(
        self,
        path: str,
        *,
        topic_map: dict[str, str],
        target_rate_hz: float = 10.0,
        profile: Optional[RobotProfile] = None,
        static_instruction: Optional[str] = None,
    ):
        self.path = Path(path)
        if "primary" not in topic_map:
            raise KeyError(
                "MCAPRecording.topic_map must include a 'primary' camera "
                "entry. Real-robot logs put cameras under arbitrary topic "
                "names; the adapter never auto-aliases the first topic."
            )
        self.topic_map = dict(topic_map)
        self.target_rate_hz = float(target_rate_hz)
        self.profile = profile
        self.static_instruction = static_instruction
        if self.path.is_dir():
            self._files = sorted(self.path.glob("*.mcap"))
            if not self._files:
                raise FileNotFoundError(
                    f"MCAPRecording: directory '{self.path}' contains no .mcap files"
                )
        else:
            if not self.path.exists():
                raise FileNotFoundError(f"MCAPRecording: '{self.path}' does not exist")
            self._files = [self.path]
        self.name = f"mcap:{self.path.name}"

    # ── EpisodeSource interface ────────────────────────────────────

    def list_episodes(self) -> list[str]:
        return [str(i) for i in range(len(self._files))]

    def load_episode(self, episode_id: str) -> list[Scene]:
        idx = int(episode_id)
        if idx < 0 or idx >= len(self._files):
            raise IndexError(
                f"MCAPRecording: episode {idx} out of range "
                f"(have {len(self._files)} mcap file(s))"
            )
        return self._mcap_to_scenes(self._files[idx], idx)

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        return {i: self.load_episode(str(i)) for i in episode_indices}

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=self.target_rate_hz,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={
                "mcap_file": str(self._files[episode_idx]),
                "has_recorded_expert_action": False,
            },
        )

    def all_instructions(self) -> list[str]:
        if self.static_instruction:
            return [self.static_instruction]
        # We could mcap-scan for /instruction topic strings, but that's
        # expensive — only do it on explicit request via the diagnostics.
        return []

    # ── internals ──────────────────────────────────────────────────

    def _mcap_to_scenes(self, mcap_path: Path, ep_idx: int) -> list[Scene]:
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
        except ImportError as e:
            raise ImportError(
                "MCAPRecording needs the `mcap` extra. Install with: "
                "pip install 'emboviz[mcap]'. Underlying error: " + str(e)
            ) from e

        wanted_topics = set(self.topic_map.values())

        # Per-topic time-sorted list of (timestamp_ns, decoded_message)
        # Also remember the schema name for type dispatch on decode.
        streams: dict[str, list[tuple[int, Any]]] = {}
        schema_for_topic: dict[str, str] = {}
        with open(mcap_path, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[Ros2DecoderFactory()])
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=list(wanted_topics),
            ):
                if channel.topic not in wanted_topics:
                    continue
                streams.setdefault(channel.topic, []).append(
                    (message.log_time, decoded)
                )
                schema_for_topic[channel.topic] = schema.name if schema else ""

        primary_topic = self.topic_map["primary"]
        primary_stream = streams.get(primary_topic, [])
        if not primary_stream:
            raise ValueError(
                f"MCAPRecording: primary camera topic '{primary_topic}' "
                f"yielded zero messages in {mcap_path}. Topics found in "
                f"the file: {sorted(streams)}."
            )

        # Sort each stream by timestamp (MCAP should already be ordered,
        # but be defensive).
        for s in streams.values():
            s.sort(key=lambda kv: kv[0])

        # Resample at target rate. Use the primary camera's first
        # timestamp as t0; emit one frame per 1/target_rate_hz.
        t0_ns = primary_stream[0][0]
        tEnd_ns = primary_stream[-1][0]
        dt_ns = int(1e9 / max(self.target_rate_hz, 1e-3))
        frame_times = list(range(t0_ns, tEnd_ns + 1, dt_ns))

        scenes: list[Scene] = []
        for fi, ts in enumerate(frame_times):
            # Per-topic sample-and-hold.
            sample: dict[str, Any] = {}
            for role, topic in self.topic_map.items():
                msg = _nearest_at_or_before(streams.get(topic, []), ts)
                if msg is not None:
                    sample[role] = (msg, schema_for_topic.get(topic, ""))

            # Decode images.
            images: dict[str, RGBImage] = {}
            for role, (msg, schema_name) in sample.items():
                if role not in self.topic_map or role in ("state", "action", "instruction"):
                    continue
                pil = _decode_image(msg, schema_name)
                if pil is not None:
                    images[role] = RGBImage(data=pil, camera_id=role)

            if "primary" not in images:
                # Primary at this frame couldn't be decoded — skip the
                # frame (don't synthesize). The user gets a smaller
                # window honestly rather than fake images.
                continue

            # State (proprio + optional gripper).
            proprio: Optional[Proprioception] = None
            gripper: Optional[GripperState] = None
            if "state" in sample:
                msg, schema_name = sample["state"]
                values, gripper_val = _decode_state(msg, schema_name)
                if values is not None:
                    state_convention = (
                        self.profile.state.convention
                        if self.profile and self.profile.state else "joint_angles"
                    )
                    proprio = Proprioception(values=values, convention=state_convention)
                if gripper_val is not None and self.profile and self.profile.gripper:
                    gripper = GripperState(
                        value=float(gripper_val),
                        kind=self.profile.gripper.kind,
                        units=self.profile.gripper.units,
                    )

            # Instruction.
            instruction = ""
            if "instruction" in sample:
                msg, _ = sample["instruction"]
                instruction = _decode_string(msg) or ""
            if not instruction and self.static_instruction:
                instruction = self.static_instruction

            # Per-frame metadata. Crucially: deployment recordings do
            # NOT have a recorded expert action. We log the model's
            # PREDICTED action (from the policy topic) separately so
            # the user can compare our re-run to it; but we never
            # populate ``expert_action``, so the runner's
            # --show-imitation gate correctly hides imitation L2.
            metadata: dict = {
                "fps":            self.target_rate_hz,
                "frame_index":    fi,
                "episode_index":  ep_idx,
                "log_time_ns":    int(ts),
                "dataset":        str(mcap_path),
                "has_recorded_expert_action": False,
            }
            if "action" in sample:
                msg, schema_name = sample["action"]
                pred = _decode_action(msg, schema_name)
                if pred is not None:
                    metadata["policy_predicted_action"] = pred.tolist()

            scenes.append(Scene(
                observations=Observations(images=images, state=proprio, gripper=gripper),
                instruction=instruction,
                profile=self.profile,
                metadata=metadata,
                scene_id=f"{self.name}:{ep_idx}:{fi}",
            ))
        return scenes


# ────────────────────────────────────────────────────────────────────
# ROS 2 message → emboviz type decoders
# ────────────────────────────────────────────────────────────────────

def _decode_image(msg: Any, schema_name: str) -> Optional[Image.Image]:
    """Decode sensor_msgs/Image or sensor_msgs/CompressedImage → PIL.Image."""
    if "CompressedImage" in schema_name:
        data = bytes(msg.data)
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            warnings.warn(f"CompressedImage decode failed: {e}")
            return None
    if "Image" in schema_name:
        # sensor_msgs/Image: encoding, height, width, step, data
        encoding = getattr(msg, "encoding", "rgb8")
        h, w = int(msg.height), int(msg.width)
        raw = bytes(msg.data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if encoding in ("rgb8", "8UC3"):
            arr = arr.reshape(h, w, 3)
        elif encoding == "bgr8":
            arr = arr.reshape(h, w, 3)[..., ::-1]
        elif encoding in ("rgba8", "8UC4"):
            arr = arr.reshape(h, w, 4)[..., :3]
        elif encoding == "bgra8":
            arr = arr.reshape(h, w, 4)[..., :3][..., ::-1]
        elif encoding in ("mono8", "8UC1"):
            arr = arr.reshape(h, w)
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            warnings.warn(
                f"Unsupported sensor_msgs/Image encoding '{encoding}' — "
                f"add a decoder branch in emboviz/recordings/mcap.py."
            )
            return None
        return Image.fromarray(arr)
    return None


def _decode_state(msg: Any, schema_name: str) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Decode sensor_msgs/JointState or geometry_msgs/PoseStamped → (vec, gripper?).

    For JointState we use position[]. For PoseStamped we concatenate
    xyz + xyzw quaternion. Gripper extraction is delegated to the
    profile/gripper_extractor in the runner — this decoder returns None
    for gripper unless the message itself carries it.
    """
    if "JointState" in schema_name:
        pos = np.asarray(getattr(msg, "position", []), dtype=np.float32).reshape(-1)
        return pos, None
    if "PoseStamped" in schema_name:
        p = msg.pose.position
        q = msg.pose.orientation
        vec = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        return vec, None
    if "Pose" in schema_name:
        p = msg.position
        q = msg.orientation
        vec = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        return vec, None
    return None, None


def _decode_string(msg: Any) -> Optional[str]:
    """Decode std_msgs/String → str. Falls back to None on anything else."""
    s = getattr(msg, "data", None)
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    return None


def _decode_action(msg: Any, schema_name: str) -> Optional[np.ndarray]:
    """Decode the policy's predicted action.

    Supported schemas:
      • ``std_msgs/Float32MultiArray`` — ``msg.data`` is the vector
      • ``geometry_msgs/Twist`` — concatenates linear + angular
      • ``geometry_msgs/TwistStamped`` — same, from .twist
    """
    if "Float32MultiArray" in schema_name or "Float64MultiArray" in schema_name:
        return np.asarray(msg.data, dtype=np.float32).reshape(-1)
    if "TwistStamped" in schema_name:
        t = msg.twist
        return np.array([t.linear.x, t.linear.y, t.linear.z,
                         t.angular.x, t.angular.y, t.angular.z], dtype=np.float32)
    if "Twist" in schema_name:
        return np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                         msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32)
    return None


def _nearest_at_or_before(stream: list[tuple[int, Any]], ts: int) -> Optional[Any]:
    """Return the latest message in ``stream`` whose timestamp ≤ ts.

    The stream is presumed sorted ascending by timestamp. We binary-
    search; on a stream of 30Hz over a 30-second recording that's
    ~900 entries — linear scan would also be fine but binary keeps
    the loop body honest.
    """
    if not stream:
        return None
    lo, hi = 0, len(stream) - 1
    if stream[0][0] > ts:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if stream[mid][0] <= ts:
            lo = mid
        else:
            hi = mid - 1
    return stream[lo][1]
