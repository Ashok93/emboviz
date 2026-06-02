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
never hand-typed, never guessed.

Format coverage — the "saved episode" dataset formats:
  • ``lerobot`` — LeRobot v3.0, read by the ISOLATED ``emboviz-lerobot``
    worker (its venv pins the latest lerobot; core never imports lerobot).
    ``build_source`` connects to that worker and returns its
    :class:`ZMQReaderClient`, which IS an EpisodeSource over the wire.
  • ``gr00t``   — LeRobot v2.1 + ``meta/modality.json`` (NVIDIA Isaac-GR00T),
    read by the ISOLATED ``emboviz-reader-gr00t`` worker (its venv pins the
    last v2.1-capable lerobot, 0.3.x). Same wire mechanism as ``lerobot``.
  • ``hdf5``    — read in-process (h5py is light + conflict-free).
  • ``rlds``    — read in-process (the ``rlds`` extra pulls tensorflow).

The shared profile / gripper-extractor construction lives in
``emboviz_wire.dataset_build`` so the in-process readers here and the
isolated lerobot worker build profiles from the SAME code.
"""

from __future__ import annotations

from typing import Optional

from emboviz_wire.dataset_build import build_profile, make_gripper_extractor
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
    if format == "gr00t":
        return _build_gr00t(path, cameras, state, action, gripper,
                            instruction, n_episodes)
    if format == "hdf5":
        return _build_hdf5(path, cameras, state, action, gripper,
                           instruction, extra)
    if format == "rlds":
        return _build_rlds(path, cameras, state, action, gripper,
                           instruction, extra)
    raise ValueError(
        f"unknown dataset.format={format!r} — emboviz reads these "
        "self-describing dataset formats: 'lerobot' (v3.0), 'gr00t' "
        "(LeRobot v2.1 + modality.json), 'hdf5', 'rlds'. (Rerun/MCAP are "
        "recording-viz formats, not dataset inputs.)"
    )


# ── LeRobot — isolated reader worker ──────────────────────────────────

def _build_lerobot(path, cameras, state, action, gripper, instruction, n_episodes):
    """Connect to the isolated ``emboviz-lerobot`` reader worker.

    Core does NOT read LeRobot data in-process (lerobot's transitive
    pins — notably ``rerun-sdk<0.27`` — would collide with core's own
    .rrd exporter). Instead we spawn the reader worker in its own venv
    and return its :class:`ZMQReaderClient`, an ``EpisodeSource`` whose
    methods round-trip over the wire. The worker reads ``info.json`` and
    builds the RobotProfile itself; all the dataset config travels as the
    worker's construction kwargs.
    """
    from emboviz.adapters import connect_reader

    reader_kwargs = {
        "path": path,
        "cameras": dict(cameras),
        "state": state,
        "action": action,
        "gripper": gripper,
        "instruction": instruction,
        "n_episodes": n_episodes,
    }
    return connect_reader("lerobot", reader_kwargs=reader_kwargs)


# ── GR00T — isolated reader worker (LeRobot v2.1 + modality.json) ─────

def _build_gr00t(path, cameras, state, action, gripper, instruction, n_episodes):
    """Connect to the isolated ``emboviz-reader-gr00t`` reader worker.

    A GR00T dataset is a LeRobot **v2.1** dataset plus ``meta/modality.json``.
    lerobot >=0.4 cannot read v2.x, so this reader's venv pins the last
    v2.1-capable lerobot (0.3.x); it is a SEPARATE reader from the v3.0
    ``lerobot`` one and from the GR00T *model* adapter. The reader's spec
    name is ``reader-gr00t`` (its own venv); the user-facing format is
    ``gr00t``. Same lifecycle as the lerobot reader: spawn the worker in
    its own venv and return its :class:`ZMQReaderClient` (an
    ``EpisodeSource`` over the wire). The worker reads ``info.json`` +
    ``modality.json`` and builds the RobotProfile itself.
    """
    from emboviz.adapters import connect_reader

    reader_kwargs = {
        "path": path,
        "cameras": dict(cameras),
        "state": state,
        "action": action,
        "gripper": gripper,
        "instruction": instruction,
        "n_episodes": n_episodes,
    }
    return connect_reader("reader-gr00t", reader_kwargs=reader_kwargs)


# ── HDF5 — in-process (h5py is light, conflict-free) ──────────────────

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

    profile = build_profile(
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
        gripper_extractor=make_gripper_extractor(gripper, None),
    )


# ── RLDS / TFDS (Open-X-Embodiment) — in-process ──────────────────────

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
            "uv sync --extra rlds. Underlying error: " + str(e)
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

    profile = build_profile(
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
        gripper_extractor=make_gripper_extractor(gripper, None),
    )
