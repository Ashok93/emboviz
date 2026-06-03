"""Shared construction helpers for dataset readers.

Turning a run config's ``dataset`` section into a :class:`RobotProfile`
and a gripper-extraction function is identical across every dataset
format (LeRobot / HDF5 / RLDS). The helpers live HERE — in the wire
package — so both:

  * emboviz core's in-process readers (HDF5, RLDS), and
  * the isolated ``emboviz-lerobot`` reader worker (which has the wire
    package but NOT emboviz core),

build profiles the same way from the same code, with no duplication and
no drift. They are pure (no I/O, no heavy deps) and depend only on the
profile types already defined in this package.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from emboviz_wire.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)


def parse_lerobot_names(names_field: Any) -> Optional[list[str]]:
    """Normalize a LeRobot ``names`` field to a clean ``list[str]`` or None.

    LeRobot's per-feature ``names`` is one of: ``{"motors": [...]}`` (a
    dict wrapping a single list), a flat list, or ``null``. We return the
    contained list of strings, or None — never fabricate names.
    """
    if names_field is None:
        return None
    if isinstance(names_field, dict):
        for v in names_field.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        return None
    if isinstance(names_field, (list, tuple)):
        return [str(x) for x in names_field]
    return None


def build_profile(
    *,
    name: str,
    cameras: dict[str, str],
    state_dim: Optional[int],
    state_names: Optional[list[str]],
    convention: Optional[str],
    action_dim: Optional[int],
    action_names: Optional[list[str]],
    gripper: Optional[dict],
    segment_layout: Optional[dict] = None,
) -> RobotProfile:
    """Build a :class:`RobotProfile` from declared + read-from-dataset info.

    ``cameras`` is the role→key mapping (only its KEYS — the role names —
    become :class:`CameraSpec`s). ``state_dim`` / ``action_dim`` and the
    per-dim names are read from the dataset's own schema by the caller;
    ``convention`` and ``gripper`` come from the run config. We refuse to
    guess the state convention — a format never encodes joint-angles vs
    ee-pose, so the user must state it.

    ``segment_layout`` (optional ``{field: slice}``) names slices of the
    state vector. The GR00T reader fills it from ``modality.json`` so the
    model can route each declared state key to its exact slice; most readers
    leave it None.
    """
    state_spec = None
    if state_dim is not None:
        if not convention:
            raise ValueError(
                "dataset.state is present but state.convention is missing — "
                "the format does not encode joint-angles vs ee-pose, so you "
                "must state it (we refuse to guess)."
            )
        state_spec = StateSpec(
            dim=int(state_dim), convention=convention, joint_names=state_names,
            segment_layout=segment_layout,
        )
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


def make_gripper_extractor(
    gripper: Optional[dict], state_names: Optional[list[str]],
) -> Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]:
    """Return an extractor ``(state) -> (proprio, gripper_value)``.

    The proprio is the FULL state vector (models consume the whole state
    they were trained on); the gripper value is pulled from the declared
    dim. ``gripper.source`` is an int index or a per-dim name resolved
    against ``state_names``. No gripper → ``(state, None)``.

    A gripper declared by a SEPARATE feature key (``gripper.key``) is not
    sliced from the state at all — the reader reads it from its own column —
    so this extractor leaves the state whole and yields no gripper value.
    """
    if gripper is None:
        return lambda s: (s, None)
    src = gripper.get("source")
    if src is None:
        if gripper.get("key"):
            # Separate-feature gripper: the reader supplies it from its own
            # column; the state vector is returned unsplit.
            return lambda s: (s, None)
        raise ValueError(
            "dataset.gripper is set but neither dataset.gripper.source nor "
            "dataset.gripper.key is given. Provide the gripper's index within "
            "the state vector (`source`) OR a separate gripper feature key "
            "(`key`). (Only the 'gr00t' reader derives it from meta/modality.json.)"
        )
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
