"""DROID episode source via the lerobot-format conversion.

DROID (the 76-task Stanford / Berkeley / TRI manipulator dataset) is the
canonical π0 / GR00T benchmark dataset. The lerobot conversion is at
``IPEC-COMMUNITY/droid_100`` (a 100-episode subset) or the full
``IPEC-COMMUNITY/droid`` (76k episodes; multi-TB). Each frame has:

  • two exterior cameras (``exterior_image_1_left`` / ``exterior_image_2_left``)
  • one wrist camera (``wrist_image_left``)
  • 7-DOF joint position (``joint_position``)
  • 1-DOF gripper (``gripper_position``)
  • a language instruction

The wrist-camera and bimanual-exterior layout matches what
``Pi0Adapter(config_name="pi0_fast_droid")`` consumes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)
from emboviz.datasets.lerobot import LeRobotEpisodeSource


DROID_PROFILE = RobotProfile(
    name="droid",
    cameras=[
        CameraSpec(name="primary"),       # exterior_image_1_left
        CameraSpec(name="exterior_2"),    # exterior_image_2_left
        CameraSpec(name="wrist_left"),    # wrist_image_left
    ],
    state=StateSpec(
        dim=7,
        convention="joint_angles",
        joint_names=[f"q{i}" for i in range(7)],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(dim=7),
)


def _droid_gripper_extractor(state: np.ndarray) -> tuple:
    """DROID state layout: [q0..q6, gripper] when concatenated.

    Some lerobot DROID conversions store the gripper as a separate column;
    in that case the dataset's state vector is already 7-dim and the
    gripper has its own key — handle both.
    """
    if state.size == 7:
        return state.copy(), None   # gripper read from separate key
    if state.size == 8:
        return state[:7].copy(), float(state[7])
    raise ValueError(
        f"DROID state vector has size {state.size}; expected 7 or 8."
    )


class Droid100Source(LeRobotEpisodeSource):
    """100-episode DROID subset — light download (~5 GB) for quick experiments."""

    def __init__(self, repo_id: str = "lerobot/droid_100"):
        super().__init__(
            repo_id=repo_id,
            profile=DROID_PROFILE,
            image_keys={
                "primary":    "observation.images.exterior_image_1_left",
                "exterior_2": "observation.images.exterior_image_2_left",
                "wrist_left": "observation.images.wrist_image_left",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_droid_gripper_extractor,
            n_episodes=100,
        )
        self.name = "droid_100"


class DroidFullSource(LeRobotEpisodeSource):
    """Full DROID dataset (76k episodes; many TB). Use only if you've
    pre-downloaded with ``lerobot.download_dataset``."""

    def __init__(self, repo_id: str = "lerobot/droid_1.0.1"):
        super().__init__(
            repo_id=repo_id,
            profile=DROID_PROFILE,
            image_keys={
                "primary":    "observation.images.exterior_image_1_left",
                "exterior_2": "observation.images.exterior_image_2_left",
                "wrist_left": "observation.images.wrist_image_left",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_droid_gripper_extractor,
            n_episodes=76000,
        )
        self.name = "droid_full"


# ---------------------------------------------------------------------------
# OFFICIAL NVIDIA GR00T DROID demo dataset
# ---------------------------------------------------------------------------
# NVIDIA ships a 3-episode demo dataset with Isaac-GR00T at
# ``demo_data/droid_sample/`` formatted EXACTLY as the GR00T-N1.7-DROID
# embodiment was trained on. The state and action are 17-dim with three
# named segments (eef_9d + gripper_position + joint_position) that match
# the embodiment's declared state/action modality keys.

GR00T_DROID_PROFILE = RobotProfile(
    name="gr00t_droid",
    cameras=[
        CameraSpec(name="primary"),       # exterior_image_1_left
        CameraSpec(name="wrist_left"),    # wrist_image_left
    ],
    state=StateSpec(
        dim=17,
        convention="ee_pose",
        joint_names=(
            [f"eef_{i}" for i in range(9)]
            + ["gripper"]
            + [f"q{i}" for i in range(7)]
        ),
        segment_layout={
            "eef_9d":           slice(0, 9),
            "gripper_position": slice(9, 10),
            "joint_position":   slice(10, 17),
        },
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(
        dim=17,
        dim_names=(
            [f"eef_{i}" for i in range(9)]
            + ["gripper"]
            + [f"q{i}" for i in range(7)]
        ),
    ),
)


def _gr00t_droid_gripper_extractor(state: np.ndarray) -> tuple:
    """17-dim layout: eef_9d (0:9), gripper_position (9:10), joint_position (10:17)."""
    if state.size != 17:
        raise ValueError(
            f"GR00T DROID state size {state.size}; expected 17 "
            "(eef_9d 9 + gripper 1 + joint 7)."
        )
    return state.copy(), float(state[9])


class GR00TDroidSampleSource:
    """NVIDIA's official ``demo_data/droid_sample`` (3 episodes, 844 frames).

    Format matches the GR00T-N1.7-DROID embodiment exactly: 2 cameras
    (exterior_1_left → primary, wrist_left → wrist_left), 17-dim state
    and action with named segments (eef_9d + gripper_position + joint_position).

    Loaded from local disk using gr00t's own ``LeRobotEpisodeLoader`` — the
    huggingface ``LeRobotDataset`` class refuses to load from local paths
    without a hub repo_id (it tries to validate against the hub even with
    ``root=``). gr00t's loader is the right tool because NVIDIA shipped
    the demo dataset alongside the model and wrote a loader that just
    reads parquet + decodes mp4 frames locally.
    """

    def __init__(self, local_dir: str = "/root/repos/Isaac-GR00T/demo_data/droid_sample"):
        self.local_dir = local_dir
        self.profile = GR00T_DROID_PROFILE
        self.name = "gr00t_droid_sample"

    def load_episode(self, episode_id: str) -> list:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list) -> dict:
        """Load via gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader."""
        from PIL import Image
        from pathlib import Path
        import json

        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.types import ModalityConfig

        from emboviz.core.observations import (
            GripperState, Proprioception, RGBImage,
        )
        from emboviz.core.types import Observations, Scene

        # Read tasks.jsonl + episodes.jsonl + info.json so we can label scenes
        meta_dir = Path(self.local_dir) / "meta"
        info = json.loads((meta_dir / "info.json").read_text())
        fps = float(info.get("fps", 15.0))

        # Build ModalityConfig matching the dataset's modality.json. The
        # loader expects this so it knows which video/state keys to read.
        modality = json.loads((meta_dir / "modality.json").read_text())
        # gr00t's loader expects:
        #   • state/action keys = segment names (e.g. "eef_9d")
        #   • language keys = "annotation.<inner_key>" form (with prefix)
        #   • video keys = bare key names
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(modality["video"].keys()),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(modality["state"].keys()),
            ),
            "action": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(modality["action"].keys()),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=[f"annotation.{k}" for k in modality["annotation"].keys()],
            ),
        }

        loader = LeRobotEpisodeLoader(
            dataset_path=Path(self.local_dir),
            modality_configs=modality_configs,
        )

        out: dict[int, list[Scene]] = {ep: [] for ep in episode_indices}
        for ep_i in episode_indices:
            df = loader[ep_i]   # pandas DataFrame: one row per frame
            for frame_i in range(len(df)):
                row = df.iloc[frame_i]

                # Images live in "video.<key>" columns as PIL Images per row
                images = {}
                key_map = {
                    "exterior_1_left": "primary",
                    "wrist_left":      "wrist_left",
                }
                for gk, sk in key_map.items():
                    col = f"video.{gk}"
                    if col not in row:
                        continue
                    img = row[col]
                    # PIL image or ndarray
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
                    images[sk] = RGBImage(data=img, camera_id=sk)

                # The gr00t loader emits per-segment columns named
                # "state.<segment>" / "action.<segment>". Reconstruct the
                # full 17-dim vector in declared order.
                state_parts = []
                for k in ["state.eef_9d", "state.gripper_position", "state.joint_position"]:
                    if k in row:
                        state_parts.append(np.asarray(row[k], dtype=np.float32).reshape(-1))
                if not state_parts:
                    raise ValueError(
                        f"Could not find state segments in droid_sample row "
                        f"(columns present: {list(row.index)[:12]})"
                    )
                state_arr = np.concatenate(state_parts)

                proprio = Proprioception(
                    values=state_arr.reshape(-1), convention="ee_pose",
                )
                gripper_val = float(state_arr.reshape(-1)[9]) if state_arr.size >= 10 else 0.0
                gripper = GripperState(
                    value=gripper_val,
                    kind=self.profile.gripper.kind,
                    units=self.profile.gripper.units,
                )

                # Action: same per-segment layout
                act_parts = []
                for k in ["action.eef_9d", "action.gripper_position", "action.joint_position"]:
                    if k in row:
                        act_parts.append(np.asarray(row[k], dtype=np.float32).reshape(-1))
                expert_action = (
                    np.concatenate(act_parts).astype(np.float32).tolist()
                    if act_parts else None
                )

                # Instruction: "language.<key>" column. DROID is teleop data
                # where language was added in a post-hoc crowdsourced
                # annotation pass; about 20% of episodes never got labeled.
                # We honestly propagate the missing-label state here:
                # instruction="" + metadata["has_recorded_instruction"]=False.
                # We do NOT fabricate a string.
                #
                # ``load_trajectory`` (which loads the trajectory under test)
                # checks this flag and refuses, because diagnostics on the
                # tested episode require a real instruction. Pool samplers
                # tolerate missing instructions (they just skip that episode
                # for the instruction modality but still use its images/state).
                instruction = None
                lang_cols_tried: list[str] = []
                for col in row.index:
                    if col.startswith("language."):
                        lang_cols_tried.append(col)
                        instruction = row[col]
                        if isinstance(instruction, list):
                            instruction = instruction[0] if instruction else None
                        if instruction:
                            break
                has_instr = bool(instruction)
                instruction_str = str(instruction) if has_instr else ""

                scene = Scene(
                    observations=Observations(
                        images=images, state=proprio, gripper=gripper,
                    ),
                    instruction=instruction_str,
                    profile=self.profile,
                    metadata={
                        "fps": fps,
                        "frame_index": frame_i,
                        "episode_index": ep_i,
                        "dataset": self.local_dir,
                        "expert_action": expert_action,
                        "has_recorded_instruction": has_instr,
                        "language_columns_tried": lang_cols_tried,
                    },
                    scene_id=f"{self.name}:{ep_i}:{frame_i}",
                )
                out[ep_i].append(scene)
        return out

    def load_trajectory(self, episode_idx: int):
        from emboviz.core.types import Trajectory
        scenes = self.load_episode(str(episode_idx))
        if scenes and not scenes[0].metadata.get("has_recorded_instruction", True):
            tried = scenes[0].metadata.get("language_columns_tried", [])
            raise ValueError(
                f"GR00TDroidSampleSource: episode {episode_idx} has no "
                f"recorded language instruction (tried columns {tried}). "
                f"DROID is teleop data with ~20% unlabeled episodes — "
                f"pick a labeled episode for the trajectory under test, "
                f"or extend the loader to read instructions from tasks "
                f"metadata. We never fabricate an instruction."
            )
        fps = float(scenes[0].metadata.get("fps", 15.0)) if scenes else 15.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.local_dir},
        )

    def list_episodes(self) -> list[str]:
        return ["0", "1", "2"]
