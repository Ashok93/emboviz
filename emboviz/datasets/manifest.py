"""Manifest-driven EpisodeSource construction.

``build_source`` turns the ``dataset`` section of a run config (see
:mod:`emboviz.config`) into a concrete :class:`EpisodeSource`. It is the
single entry the runner resolves for any config-driven run, so the user
writes one uniform manifest regardless of format — only the reader
behind each key changes.

What the manifest declares (uniform across formats):
  • ``cameras``  — model camera role → the dataset's image/source key
  • ``state``    — {key, convention}; convention is the thing no format
                   encodes, so the user always states it
  • ``action``   — {key}
  • ``gripper``  — optional {source, kind, units, range}
  • ``instruction`` — {from: tasks} | {key: ...} | {text: ...}

The dataset's own schema (dims + per-dim names) is read from the source —
never hand-typed, never guessed:
  • LeRobot — ``meta/info.json`` ``features``.
  • HDF5    — the first demo's array shapes.
  • RLDS    — the TFDS feature spec (``builder.info.features``), without
    materializing data.

Format coverage — the three real "saved episode" dataset formats, each
self-describing, all fully manifest-driven:
  • ``lerobot`` — schema from info.json.
  • ``hdf5``    — schema from the first demo's shapes.
  • ``rlds``    — schema from the TFDS feature spec.

Rerun (.rrd) and MCAP/rosbag2 are *recording / debugging-viz* formats,
not dataset formats — they are deliberately not input formats here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)
from emboviz.datasets.base import EpisodeSource


def build_source(
    *,
    format: str,
    path: str,
    cameras: Optional[dict[str, str]] = None,
    state: Optional[dict] = None,
    action: Optional[dict] = None,
    gripper: Optional[dict] = None,
    instruction: Optional[dict] = None,
    extra: Optional[dict] = None,
    n_episodes: Optional[int] = None,
) -> EpisodeSource:
    """Build an EpisodeSource from a run config's ``dataset`` section."""
    cameras = cameras or {}
    extra = extra or {}
    if "primary" not in cameras:
        raise KeyError(
            "dataset.cameras must include a 'primary' role (the main "
            f"exterior camera). Got roles {sorted(cameras)}. We never "
            "auto-pick a primary camera."
        )

    if format == "lerobot":
        return _build_lerobot(path, cameras, state, action, gripper,
                              instruction, n_episodes)
    if format == "hdf5":
        return _build_hdf5(path, cameras, state, action, gripper,
                           instruction, extra)
    if format == "rlds":
        return _build_rlds(path, cameras, state, action, gripper,
                           instruction, extra)
    raise ValueError(
        f"unknown dataset.format={format!r} — emboviz reads the three "
        "self-describing dataset formats: 'lerobot', 'hdf5', 'rlds'. "
        "(Rerun/MCAP are recording-viz formats, not dataset inputs.)"
    )


# ── shared profile + gripper construction ────────────────────────────

def _parse_names(names_field: Any) -> Optional[list[str]]:
    """LeRobot ``names`` is one of: {"motors": [...]} / a flat list / null.
    Return a clean list[str] or None — never fabricate."""
    if names_field is None:
        return None
    if isinstance(names_field, dict):
        # take the single declared list (motors / axes / ...)
        for v in names_field.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        return None
    if isinstance(names_field, (list, tuple)):
        return [str(x) for x in names_field]
    return None


def _build_profile(
    *, name: str, cameras: dict[str, str],
    state_dim: Optional[int], state_names: Optional[list[str]], convention: Optional[str],
    action_dim: Optional[int], action_names: Optional[list[str]],
    gripper: Optional[dict],
) -> RobotProfile:
    state_spec = None
    if state_dim is not None:
        if not convention:
            raise ValueError(
                "dataset.state is present but state.convention is missing — "
                "the format does not encode joint-angles vs ee-pose, so you "
                "must state it (we refuse to guess)."
            )
        state_spec = StateSpec(dim=int(state_dim), convention=convention,
                               joint_names=state_names)
    action_spec = (
        ActionSpec(dim=int(action_dim), dim_names=action_names)
        if action_dim is not None else None
    )
    gripper_spec = None
    if gripper is not None:
        gripper_spec = GripperSpec(
            kind=gripper.get("kind", "parallel_jaw"),
            units=gripper.get("units", "unit"),
            range=tuple(gripper.get("range", (0.0, 1.0))),
        )
    return RobotProfile(
        name=name,
        cameras=[CameraSpec(name=role) for role in cameras],
        state=state_spec,
        gripper=gripper_spec,
        action=action_spec,
    )


def _make_gripper_extractor(
    gripper: Optional[dict], state_names: Optional[list[str]],
) -> Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]:
    """Return an extractor (state) → (proprio, gripper_value).

    The proprio is the FULL state vector (models consume the whole state
    they were trained on); the gripper value is pulled from the declared
    dim. ``gripper.source`` is an int index or a per-dim name resolved
    against the state names. No gripper → (state, None)."""
    if gripper is None:
        return lambda s: (s, None)
    src = gripper["source"]
    if isinstance(src, str):
        if not state_names or src not in state_names:
            raise ValueError(
                f"gripper.source={src!r} is a name but it is not in the "
                f"state's per-dim names ({state_names}). Use the integer "
                "index instead, or fix the name."
            )
        idx = state_names.index(src)
    else:
        idx = int(src)

    def extractor(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
        if idx >= state.size:
            raise ValueError(
                f"gripper.source index {idx} is out of range for a "
                f"{state.size}-dim state vector."
            )
        return state.copy(), float(state[idx])

    return extractor


# ── LeRobot ───────────────────────────────────────────────────────────

def _read_lerobot_info(path: str) -> dict:
    """Read ``meta/info.json`` for a LeRobot dataset — local dir or HF repo.
    Uses single-file ``hf_hub_download`` (no full-tree enumeration)."""
    if os.path.isdir(path):
        info_path = Path(path) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"{info_path} not found in local dataset")
        return json.loads(info_path.read_text())
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=path, filename="meta/info.json", repo_type="dataset")
    return json.loads(Path(p).read_text())


def _build_lerobot(path, cameras, state, action, gripper, instruction, n_episodes):
    from emboviz.datasets.lerobot import LeRobotEpisodeSource

    info = _read_lerobot_info(path)
    features = info.get("features", {})

    state_key = state["key"] if state else None
    action_key = action["key"] if action else "action"
    state_dim = state_names = None
    if state_key is not None:
        feat = features.get(state_key)
        if feat is None:
            raise KeyError(
                f"dataset.state.key={state_key!r} is not a feature in "
                f"{path}'s info.json. Available: {sorted(features)}."
            )
        state_dim = feat["shape"][0]
        state_names = _parse_names(feat.get("names"))
    action_dim = action_names = None
    if action_key in features:
        action_dim = features[action_key]["shape"][0]
        action_names = _parse_names(features[action_key].get("names"))

    profile = _build_profile(
        name=info.get("robot_type") or path,
        cameras=cameras,
        state_dim=state_dim, state_names=state_names,
        convention=(state or {}).get("convention"),
        action_dim=action_dim, action_names=action_names,
        gripper=gripper,
    )
    return LeRobotEpisodeSource(
        repo_id=path,
        profile=profile,
        image_keys=dict(cameras),
        state_key=state_key,
        action_key=action_key,
        gripper_extractor=_make_gripper_extractor(gripper, state_names),
        n_episodes=int(n_episodes or info.get("total_episodes", 1_000_000)),
    )


# ── HDF5 ──────────────────────────────────────────────────────────────

def _build_hdf5(path, cameras, state, action, gripper, instruction, extra):
    from emboviz.datasets.hdf5 import HDF5EpisodeSource
    import h5py

    demo_group = extra.get("demo_group", "data")
    state_key = state["key"] if state else None
    action_key = action["key"] if action else "actions"

    # Read dims from the FIRST demo's array shapes (never fabricated).
    state_dim = action_dim = None
    with h5py.File(path, "r") as f:
        if demo_group not in f:
            raise KeyError(
                f"HDF5 file {path} has no group '{demo_group}'. "
                f"Available: {list(f.keys())}. Set dataset.extra.demo_group."
            )
        demos = sorted(f[demo_group].keys())
        if not demos:
            raise ValueError(f"HDF5 group '{demo_group}' has no demos")
        demo0 = f[demo_group][demos[0]]
        if state_key is not None:
            if state_key not in demo0:
                raise KeyError(f"state.key {state_key!r} not in demo "
                               f"'{demos[0]}' (keys: {list(demo0.keys())})")
            state_dim = int(demo0[state_key].shape[-1])
        if action_key in demo0:
            action_dim = int(demo0[action_key].shape[-1])

    profile = _build_profile(
        name=extra.get("robot_type", path),
        cameras=cameras,
        state_dim=state_dim, state_names=None,   # HDF5 carries no per-dim names
        convention=(state or {}).get("convention"),
        action_dim=action_dim, action_names=None,
        gripper=gripper,
    )

    instr = (instruction or {}).get("text")
    instr_attr = (instruction or {}).get("key")   # for hdf5, `key` = an h5 attr name
    return HDF5EpisodeSource(
        path=path,
        camera_keys=dict(cameras),
        state_key=state_key,
        action_key=action_key,
        instruction=instr,
        instruction_attr=instr_attr,
        demo_group=demo_group,
        profile=profile,
        gripper_extractor=_make_gripper_extractor(gripper, None),
    )


# ── RLDS / TFDS (Open-X-Embodiment) ───────────────────────────────────

def _build_rlds(path, cameras, state, action, gripper, instruction, extra):
    """Build an RLDSEpisodeSource. ``path`` is the TFDS builder name (the
    dataset's identity, e.g. ``"bridge"``); ``extra.data_dir`` overrides the
    TFDS data root and ``extra.split`` the split.

    Dims are read from the TFDS feature spec — no data materialized. RLDS
    nests per-step fields under a ``steps`` Sequence: the action is a
    top-level step key, the proprio state lives under ``observation``."""
    from emboviz.datasets.rlds import RLDSEpisodeSource
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf  # noqa: F401  — TFDS needs it imported
    except ImportError as e:
        raise ImportError(
            "dataset.format='rlds' needs the `rlds` extra. Install with: "
            "pip install 'emboviz[rlds]'. Underlying error: " + str(e)
        ) from e

    data_dir = extra.get("data_dir")
    split = extra.get("split", "train")

    # builder.info reads the feature spec from code (registered builders) or
    # from the prepared dataset's dataset_info.json — never downloads data.
    builder = tfds.builder(path, data_dir=data_dir)
    top = builder.info.features
    if "steps" not in top:
        raise KeyError(
            f"RLDS dataset {path!r} has no 'steps' feature (top-level keys: "
            f"{sorted(top.keys())}). emboviz reads Open-X-Embodiment-style "
            "RLDS where per-step fields live under a 'steps' Sequence."
        )
    step_dict = top["steps"].feature          # inner per-step FeaturesDict
    if "observation" not in step_dict:
        raise KeyError(
            f"RLDS dataset {path!r} step features have no 'observation' "
            f"(step keys: {sorted(step_dict.keys())})."
        )
    obs_dict = step_dict["observation"]

    state_key = state["key"] if state else None
    action_key = action["key"] if action else "action"

    state_dim = None
    if state_key is not None:
        if state_key not in obs_dict:
            raise KeyError(
                f"dataset.state.key={state_key!r} is not an RLDS observation "
                f"feature. Available: {sorted(obs_dict.keys())}."
            )
        state_dim = int(obs_dict[state_key].shape[-1])

    action_dim = None
    if action_key in step_dict:
        action_dim = int(step_dict[action_key].shape[-1])

    profile = _build_profile(
        name=path,
        cameras=cameras,
        state_dim=state_dim, state_names=None,   # TFDS Tensor features carry no per-dim names
        convention=(state or {}).get("convention"),
        action_dim=action_dim, action_names=None,
        gripper=gripper,
    )

    # `instruction.key` selects the per-step instruction field; default to
    # the OXE convention ("language_instruction"), which RLDSEpisodeSource
    # also falls back to episode_metadata for when absent per step.
    instruction_key = (instruction or {}).get("key") or "language_instruction"
    return RLDSEpisodeSource(
        builder_name=path,
        data_dir=data_dir,
        split=split,
        camera_keys=dict(cameras),
        state_key=state_key,
        action_key=action_key,
        instruction_key=instruction_key,
        profile=profile,
        gripper_extractor=_make_gripper_extractor(gripper, None),
    )
