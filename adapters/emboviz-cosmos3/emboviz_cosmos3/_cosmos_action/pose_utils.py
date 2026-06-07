# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rotation/pose utilities — VENDORED from NVIDIA cosmos-framework.

Source: ``cosmos_framework/data/vfm/action/pose_utils.py`` (OpenMDW-1.1). This is
the subset of functions emboviz needs to reproduce Cosmos's action encoding for
the LeRobot robot domains, vendored verbatim except that the torch in/out
wrapping is removed (we only ever pass NumPy) so the module depends on numpy +
scipy alone and never pulls torch into the worker.

Vendored rather than imported because the full ``cosmos_framework`` package pulls
torch and the entire Cosmos training/serving stack; this is the same approach the
emboviz LaMa adapter takes with simple-lama. The algorithm is NVIDIA's, unchanged
— do not "improve" it; keeping it byte-faithful is what guarantees the encoding
matches the model's training distribution. Validated bit-equal (< 1e-5) to the
cosmos-framework reference on the DROID encode path (euler→rot6d, rot6d→matrix,
build_abs_pose_from_components, pose_abs_to_rel rot6d backward_framewise).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

PoseConvention = Literal["absolute", "backward_anchored", "backward_framewise"]
RotationConvention = Literal[
    "matrix", "euler_xyz", "quat_xyzw", "quat_wxyz", "rot6d", "axisangle", "rot9d"
]


def _normalize_rotation_matrices(rot_matrices: np.ndarray) -> np.ndarray:
    """Project approximate matrices onto valid rotation matrices via SVD (SO(3))."""
    matrices = np.asarray(rot_matrices, dtype=np.float32)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must have shape (..., 3, 3), got {matrices.shape}")

    original_shape = matrices.shape[:-2]
    matrices_flat = matrices.reshape(-1, 3, 3)

    U, _, Vt = np.linalg.svd(matrices_flat)
    normalized = U @ Vt

    det = np.linalg.det(normalized)
    reflection_mask = det < 0
    if np.any(reflection_mask):
        U_reflect = U.copy()
        U_reflect[reflection_mask, :, -1] *= -1
        normalized[reflection_mask] = U_reflect[reflection_mask] @ Vt[reflection_mask]

    return normalized.astype(np.float32, copy=False).reshape(*original_shape, 3, 3)


def convert_rotation(
    rotation: np.ndarray,
    input_format: RotationConvention,
    output_format: RotationConvention,
    normalize_matrix: bool = False,
) -> np.ndarray:
    """Convert rotations between the conventions used by the action datasets.

    Maps the input representation to rotation matrices, then emits the requested
    output convention. NumPy in, NumPy out.
    """
    rotation_np = np.asarray(rotation, dtype=np.float32)

    if input_format == "matrix":
        if rotation_np.ndim < 2 or rotation_np.shape[-2:] != (3, 3):
            raise ValueError(f"matrix rotation must have shape (..., 3, 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-2]
        matrices_flat = rotation_np.reshape(-1, 3, 3)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "euler_xyz":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 3:
            raise ValueError(f"{input_format} rotation must have shape (..., 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = R.from_euler("xyz", rotation_np.reshape(-1, 3), degrees=False).as_matrix().astype(np.float32)
    elif input_format in ("quat_xyzw", "quat_wxyz"):
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 4:
            raise ValueError(f"{input_format} rotation must have shape (..., 4), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        quaternions = rotation_np.reshape(-1, 4)
        if input_format == "quat_wxyz":
            quaternions = quaternions[:, [1, 2, 3, 0]]
        norms = np.linalg.norm(quaternions, axis=-1)
        if np.any(norms < 1e-8):
            raise ValueError(f"Found zero-norm quaternion(s) (min norm={norms.min():.2e}).")
        if normalize_matrix:
            quaternions = quaternions / norms[:, None]
        matrices_flat = R.from_quat(quaternions).as_matrix().astype(np.float32)
    elif input_format == "rot6d":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 6:
            raise ValueError(f"{input_format} rotation must have shape (..., 6), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        rot6d_flat = rotation_np.reshape(-1, 6)
        col0 = rot6d_flat[:, :3]
        col1 = rot6d_flat[:, 3:]
        col2 = np.cross(col0, col1, axis=-1)
        matrices_flat = np.stack((col0, col1, col2), axis=-1).astype(np.float32)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "rot9d":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 9:
            raise ValueError(f"rot9d rotation must have shape (..., 9), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = rotation_np.reshape(-1, 3, 3)
        if normalize_matrix:
            matrices_flat = _normalize_rotation_matrices(matrices_flat).reshape(-1, 3, 3)
    elif input_format == "axisangle":
        if rotation_np.ndim < 1 or rotation_np.shape[-1] != 3:
            raise ValueError(f"axisangle rotation must have shape (..., 3), got {rotation_np.shape}")
        original_shape = rotation_np.shape[:-1]
        matrices_flat = R.from_rotvec(rotation_np.reshape(-1, 3)).as_matrix().astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_format: {input_format!r}")

    if output_format == "matrix":
        converted = matrices_flat.reshape(*original_shape, 3, 3).astype(np.float32)
    elif output_format == "rot9d":
        converted = matrices_flat.reshape(-1, 9)
    elif output_format == "rot6d":
        converted = matrices_flat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    elif output_format == "quat_xyzw":
        converted = R.from_matrix(matrices_flat).as_quat().astype(np.float32)
    elif output_format == "quat_wxyz":
        converted = R.from_matrix(matrices_flat).as_quat().astype(np.float32)
        converted = converted[:, [3, 0, 1, 2]]
    elif output_format == "euler_xyz":
        converted = R.from_matrix(matrices_flat).as_euler("xyz", degrees=False).astype(np.float32)
    elif output_format == "axisangle":
        converted = R.from_matrix(matrices_flat).as_rotvec().astype(np.float32)
    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")

    if output_format != "matrix":
        converted = converted.reshape(*original_shape, converted.shape[-1])
    return converted


def build_abs_pose_from_components(
    xyz: np.ndarray,
    rotation: np.ndarray,
    rotation_input_format: Literal["euler_xyz", "quat_xyzw", "quat_wxyz", "axisangle"],
    translation_scale: float | None = None,
) -> np.ndarray:
    """Build absolute homogeneous poses ``(T, 4, 4)`` from per-frame xyz + rotation."""
    xyz_np = np.asarray(xyz, dtype=np.float32)
    rotation_np = np.asarray(rotation, dtype=np.float32)

    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        raise ValueError(f"xyz must have shape (T, 3), got {xyz_np.shape}")
    if rotation_np.ndim != 2:
        raise ValueError(f"rotation must be 2D, got {rotation_np.shape}")
    if rotation_np.shape[0] != xyz_np.shape[0]:
        raise ValueError(
            f"xyz and rotation must have the same length, got {xyz_np.shape[0]} and {rotation_np.shape[0]}"
        )

    rot_mats = np.asarray(
        convert_rotation(rotation_np, input_format=rotation_input_format, output_format="matrix"),
        dtype=np.float32,
    )

    if translation_scale is not None:
        if translation_scale == 0:
            raise ValueError("translation_scale must be non-zero")
        xyz_np = xyz_np / float(translation_scale)

    poses_abs = np.eye(4, dtype=np.float32)[None].repeat(xyz_np.shape[0], axis=0)
    poses_abs[:, :3, :3] = rot_mats.astype(np.float32)
    poses_abs[:, :3, 3] = xyz_np
    return poses_abs


def _delta_transform_to_pose_vector(
    delta_T: np.ndarray,
    rotation_output_format: RotationConvention,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Encode a relative transform as an action vector ``[translation(3), rotation(...)]``."""
    delta_np = np.asarray(delta_T, dtype=np.float32)
    if delta_np.shape != (4, 4):
        raise ValueError(f"delta_T must have shape (4, 4), got {delta_np.shape}")

    translation = delta_np[:3, 3] * translation_scale
    rotation = np.asarray(
        convert_rotation(delta_np[:3, :3], input_format="matrix", output_format=rotation_output_format),
        dtype=np.float32,
    )
    rotation = rotation * rotation_scale
    return np.concatenate([translation, rotation]).astype(np.float32)


def _get_relative_delta_transform(
    poses_abs: np.ndarray,
    inv_poses_abs: np.ndarray,
    frame_idx: int,
    pose_convention: PoseConvention,
) -> np.ndarray:
    """Compute one relative transform from an absolute-pose trajectory."""
    if pose_convention == "backward_framewise":
        return inv_poses_abs[frame_idx] @ poses_abs[frame_idx + 1]
    if pose_convention == "backward_anchored":
        return inv_poses_abs[0] @ poses_abs[frame_idx + 1]
    raise ValueError(
        f"Unsupported pose_convention={pose_convention!r}. Expected one of: "
        "backward_framewise, backward_anchored."
    )


def pose_abs_to_rel(
    poses_abs: np.ndarray,
    rotation_format: RotationConvention = "rot9d",
    pose_convention: PoseConvention = "backward_framewise",
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Convert an absolute-pose trajectory ``(T, 4, 4)`` into ``(T-1, 3 + rot_dim)``
    relative-pose action vectors."""
    num_frames = len(poses_abs)
    assert num_frames > 1, "At least 2 frames are required to compute relative poses"

    inv_poses_abs = np.linalg.inv(poses_abs)

    poses_rel = []
    for i in range(num_frames - 1):
        delta_T = _get_relative_delta_transform(poses_abs, inv_poses_abs, i, pose_convention)
        poses_rel.append(
            _delta_transform_to_pose_vector(
                delta_T,
                rotation_output_format=rotation_format,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
        )

    return np.stack(poses_rel).astype(np.float32)  # [T-1, D]
