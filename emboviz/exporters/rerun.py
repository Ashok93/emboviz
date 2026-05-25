"""Rerun `.rrd` exporter — the killer integration.

Emits diagnostic outputs as toggleable timeline tracks in a Rerun
recording. A roboticist who already uses Rerun for rollout playback
now sees Emboviz diagnostics overlaid on their existing camera streams —
zero context switch.

What gets logged:
  • `cameras/<camera_name>/image`           per-frame RGB image
  • `diagnostics/<axis>/score`              per-frame scalar over time
  • `diagnostics/<axis>/severity`           per-frame severity (color-coded)
  • `diagnostics/<axis>/explanation`        per-frame text annotation
  • `predictions/action/<dim_name>`         per-frame action vector dims
  • `predictions/expert/<dim_name>`         expert action (if metadata has it)

Lazy imports `rerun-sdk`. Install with: uv add rerun-sdk
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Trajectory
from emboviz.diagnostics.trajectory import TrajectoryDiagnosticResult


_SEVERITY_RGB = {
    Severity.CRITICAL: (201, 42, 42),
    Severity.MODERATE: (230, 119, 0),
    Severity.INFO:     (25, 113, 194),
    Severity.PASS:     (47, 158, 68),
    Severity.UNKNOWN:  (134, 142, 150),
}


def export_rerun(
    trajectory: Trajectory,
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    out_path: Path,
    *,
    application_id: str = "emboviz",
) -> Path:
    """Emit an .rrd recording containing camera streams + diagnostic tracks.

    Args:
        trajectory: source rollout (provides cameras + frame timing)
        per_axis_results: {axis_name → TrajectoryDiagnosticResult}
        out_path: where to write the .rrd
        application_id: shown in the Rerun viewer title bar
    """
    try:
        import rerun as rr
    except ImportError as e:
        raise ImportError(
            "Rerun export requires the `rerun-sdk` package. "
            "Install with: uv add rerun-sdk"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recording_id = trajectory.source or trajectory.episode_id or "rollout"
    rec = rr.new_recording(application_id=application_id, recording_id=recording_id)
    fps = trajectory.fps if trajectory.fps > 0 else 5.0

    # 1. Camera streams (one timeline per recording, frames at trajectory.fps).
    for i, scene in enumerate(trajectory.frames):
        rr.set_time_seconds("frame_time", i / fps, recording=rec)
        rr.set_time_sequence("frame_index", trajectory.frame_indices[i], recording=rec)
        for cam_name, rgb in scene.observations.images.items():
            arr = np.asarray(rgb.data)
            rr.log(f"cameras/{cam_name}/image", rr.Image(arr), recording=rec)

        # Action vector (predicted) — if any diagnostic recorded a baseline action.
        baseline_action = _find_baseline_action(scene, per_axis_results, i)
        if baseline_action is not None:
            dim_names = _action_dim_names(scene, len(baseline_action))
            for d, name in enumerate(dim_names):
                rr.log(
                    f"predictions/action/{name}",
                    rr.Scalar(float(baseline_action[d])),
                    recording=rec,
                )
        expert = scene.metadata.get("expert_action")
        if expert is not None:
            dim_names = _action_dim_names(scene, len(expert))
            for d, name in enumerate(dim_names):
                rr.log(
                    f"predictions/expert/{name}",
                    rr.Scalar(float(expert[d])),
                    recording=rec,
                )

    # 2. Diagnostic tracks — one set of channels per axis.
    for axis, traj_result in per_axis_results.items():
        for i, r in enumerate(traj_result.per_frame):
            rr.set_time_seconds("frame_time", i / fps, recording=rec)
            rr.set_time_sequence(
                "frame_index", traj_result.frame_indices[i], recording=rec,
            )
            score = r.scalar_score
            if score == score:  # not NaN
                rr.log(f"diagnostics/{axis}/score", rr.Scalar(float(score)),
                       recording=rec)
            color = _SEVERITY_RGB.get(r.severity, (128, 128, 128))
            rr.log(
                f"diagnostics/{axis}/severity",
                rr.TextLog(r.severity.value, color=color),
                recording=rec,
            )
            if r.explanation:
                rr.log(
                    f"diagnostics/{axis}/explanation",
                    rr.TextDocument(r.explanation),
                    recording=rec,
                )

    rr.save(str(out_path), recording=rec)
    return out_path


def _find_baseline_action(
    scene, per_axis_results: dict[str, TrajectoryDiagnosticResult], frame_idx: int,
) -> Optional[np.ndarray]:
    """Pull the baseline action out of any diagnostic that recorded one."""
    for traj in per_axis_results.values():
        if frame_idx >= len(traj.per_frame):
            continue
        raw = traj.per_frame[frame_idx].raw or {}
        action = raw.get("baseline_action")
        if action is None:
            variants = raw.get("variants")
            if variants and isinstance(variants, list) and variants:
                action = variants[0].get("baseline_action")
        if action is not None:
            return np.asarray(action, dtype=np.float32)
    return None


def _action_dim_names(scene, dim: int) -> list[str]:
    """Use RobotProfile.action.dim_names if available, else d0..dN."""
    if (
        scene.profile is not None
        and scene.profile.action is not None
        and scene.profile.action.dim_names is not None
        and len(scene.profile.action.dim_names) >= dim
    ):
        return list(scene.profile.action.dim_names[:dim])
    return [f"d{i}" for i in range(dim)]
