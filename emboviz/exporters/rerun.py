"""Rerun `.rrd` exporter — the killer integration.

Emits diagnostic outputs as toggleable timeline tracks in a Rerun
recording. A roboticist who already uses Rerun for rollout playback
now sees Emboviz diagnostics overlaid on their existing camera streams —
zero context switch.

What gets logged:
  • `cameras/<camera_name>/image`           per-frame RGB image
  • `attention/<camera_name>/heatmap`       per-frame attention overlay (if supplied)
  • `sensitivity/<camera_name>/heatmap`     per-frame sensitivity overlay (if supplied)
  • `target/<camera_name>/mask`             per-frame target detection mask (if supplied)
  • `diagnostics/<axis>/score`              per-frame scalar over time
  • `diagnostics/<axis>/severity`           per-frame severity (color-coded)
  • `diagnostics/<axis>/explanation`        per-frame text annotation
  • `predictions/action/<dim_name>`         per-frame predicted action vector
  • `predictions/expert/<dim_name>`         per-frame expert action (from metadata)
  • `predictions/delta_to_expert`           per-frame ‖predicted − expert‖
  • `predictions/delta_per_dim/<dim_name>`  per-frame per-dimension delta

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


PerFrameByCamera = dict[int, dict[str, np.ndarray]]
PerFrameDetection = dict[int, dict[str, dict]]
PerFrameMaskedImage = dict[int, dict[str, dict[str, np.ndarray]]]
PerFrameModalityResponse = dict[int, dict[str, float]]


def export_rerun(
    trajectory: Trajectory,
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    out_path: Path,
    *,
    application_id: str = "emboviz",
    attention_per_frame: Optional[PerFrameByCamera] = None,
    sensitivity_per_frame: Optional[PerFrameByCamera] = None,
    target_mask_per_frame: Optional[PerFrameByCamera] = None,
    target_detection_per_frame: Optional[PerFrameDetection] = None,
    masked_image_per_frame: Optional[PerFrameMaskedImage] = None,
    modality_response_per_frame: Optional[PerFrameModalityResponse] = None,
) -> Path:
    """Emit an .rrd recording containing camera streams + diagnostic tracks.

    Args:
        trajectory: source rollout (provides cameras + frame timing).
        per_axis_results: {axis_name → TrajectoryDiagnosticResult}.
        out_path: where to write the .rrd.
        application_id: shown in the Rerun viewer title bar.
        attention_per_frame: optional ``{frame_idx → {camera_name → (H, W)
            float heatmap}}`` per-camera attention overlays. Logged at
            ``attention/<camera>/heatmap``.
        sensitivity_per_frame: same nested shape — per-camera occlusion
            grids, logged at ``sensitivity/<camera>/heatmap``.
        target_mask_per_frame: same shape — per-camera target masks,
            logged at ``target/<camera>/mask``.
        target_detection_per_frame: ``{frame_idx → {camera → {bbox,
            confidence, label}}}`` — GroundingDINO detections logged as
            ``target/<camera>/bbox`` (with confidence + label text), so
            users can verify the diagnostic targeted the right object
            before trusting any memorization verdict on that frame.
        masked_image_per_frame: ``{frame_idx → {camera → {fill_mode →
            HxWxC uint8 image}}}`` — the masked-with-each-fill images
            the memorization diagnostic actually fed to the model.
            Logged at ``target/<camera>/masked_<fill_mode>`` so users
            can see at a glance whether the fill made the target
            visually disappear.
        modality_response_per_frame: ``{frame_idx → {modality →
            normalized Δaction}}`` — per-modality response magnitudes
            from modality_dropout. Logged as scalar timelines at
            ``modality/<modality>/response_normalized`` so users get a
            line plot showing which input drives the action over time.
    """
    try:
        import rerun as rr
    except ImportError as e:
        raise ImportError(
            "Rerun export requires the `rerun-sdk` package (>=0.23). "
            "Install with: uv pip install 'rerun-sdk>=0.23'"
        ) from e

    # ── rerun >= 0.23 API ─────────────────────────────────────────────
    # 0.23 unified the time-setter functions and renamed Scalar →
    # Scalars; 0.23 also replaced ``new_recording(...)`` with the
    # public ``RecordingStream(...)`` constructor. We target 0.23+ only.
    if not hasattr(rr, "RecordingStream") or not hasattr(rr, "set_time"):
        raise RuntimeError(
            "rerun-sdk too old. Install >=0.23: "
            "uv pip install 'rerun-sdk>=0.23'"
        )
    Scalars = rr.Scalars  # archetype rename was 0.23

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recording_id = trajectory.source or trajectory.episode_id or "rollout"
    rec = rr.RecordingStream(
        application_id=application_id, recording_id=recording_id,
    )
    fps = trajectory.fps if trajectory.fps > 0 else 5.0

    attention_per_frame          = attention_per_frame          or {}
    sensitivity_per_frame        = sensitivity_per_frame        or {}
    target_mask_per_frame        = target_mask_per_frame        or {}
    target_detection_per_frame   = target_detection_per_frame   or {}
    masked_image_per_frame       = masked_image_per_frame       or {}
    modality_response_per_frame  = modality_response_per_frame  or {}

    def _per_camera_for_frame(
        store: PerFrameByCamera, frame_idx: int, fallback_i: int,
    ) -> dict[str, np.ndarray]:
        """Look up frame data, preferring dataset frame_idx then enumeration idx."""
        if frame_idx in store:
            return store[frame_idx]
        if fallback_i in store:
            return store[fallback_i]
        return {}

    # 1. Camera streams + visual overlays per frame.
    for i, scene in enumerate(trajectory.frames):
        frame_idx = trajectory.frame_indices[i]
        rr.set_time("frame_time", duration=i / fps, recording=rec)
        rr.set_time("frame_index", sequence=frame_idx, recording=rec)
        scene_cameras = set(scene.observations.images.keys())
        for cam_name, rgb in scene.observations.images.items():
            arr = np.asarray(rgb.data)
            rr.log(f"cameras/{cam_name}/image", rr.Image(arr), recording=rec)

        # Per-camera attention heatmap overlays.
        for cam, attn in _per_camera_for_frame(
            attention_per_frame, frame_idx, i,
        ).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"attention_per_frame logs camera '{cam}' at frame "
                    f"{frame_idx} but the scene only has {sorted(scene_cameras)}. "
                    "Either fix the diagnostic that produced this overlay or "
                    "load that camera in the dataset adapter — we will not "
                    "silently attach an overlay to the wrong camera."
                )
            rr.log(
                f"attention/{cam}/heatmap",
                rr.Image(_colorize_heatmap(attn)), recording=rec,
            )

        # Per-camera sensitivity heatmap overlays.
        for cam, sens in _per_camera_for_frame(
            sensitivity_per_frame, frame_idx, i,
        ).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"sensitivity_per_frame logs camera '{cam}' at frame "
                    f"{frame_idx} but the scene only has {sorted(scene_cameras)}."
                )
            rr.log(
                f"sensitivity/{cam}/heatmap",
                rr.Image(_colorize_heatmap(sens)), recording=rec,
            )

        # Per-camera target detection masks.
        for cam, tmask in _per_camera_for_frame(
            target_mask_per_frame, frame_idx, i,
        ).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"target_mask_per_frame logs camera '{cam}' at frame "
                    f"{frame_idx} but the scene only has {sorted(scene_cameras)}."
                )
            rr.log(
                f"target/{cam}/mask",
                rr.Image(_mask_to_rgb(tmask)), recording=rec,
            )

        # Per-camera GroundingDINO detection metadata (bbox + label +
        # confidence). Lets the user verify the diagnostic targeted the
        # RIGHT object before trusting any memorization verdict.
        for cam, det in _per_camera_for_frame(
            target_detection_per_frame, frame_idx, i,
        ).items():
            if cam not in scene_cameras:
                continue
            x0, y0, x1, y1 = det.get("bbox", (0, 0, 0, 0))
            label = det.get("label", "")
            conf  = float(det.get("confidence", 0.0))
            try:
                rr.log(
                    f"target/{cam}/bbox",
                    rr.Boxes2D(
                        array=[[x0, y0, x1, y1]],
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"{label} ({conf:.2f})"],
                    ),
                    recording=rec,
                )
            except Exception:
                # Older rerun-sdk versions had a different Boxes2D API.
                # Fall back to a text annotation; bbox geometry is
                # still in the mask logged above.
                rr.log(
                    f"target/{cam}/label",
                    rr.TextDocument(
                        f"detected '{label}' "
                        f"conf {conf:.2f} bbox {(x0, y0, x1, y1)}"
                    ),
                    recording=rec,
                )

        # Per-fill masked images — what the diagnostic actually fed to
        # the model when computing Δaction for memorization.
        for cam, fills in _per_camera_for_frame(
            masked_image_per_frame, frame_idx, i,
        ).items():
            if cam not in scene_cameras:
                continue
            for fill_mode, masked_arr in fills.items():
                rr.log(
                    f"target/{cam}/masked_{fill_mode}",
                    rr.Image(np.asarray(masked_arr)),
                    recording=rec,
                )

        # Per-modality response scalars (modality attribution line plot).
        mod_resp = (
            modality_response_per_frame.get(frame_idx)
            or modality_response_per_frame.get(i)
            or {}
        )
        for modality, value in mod_resp.items():
            # Sanitize modality name for Rerun's path namespace.
            safe = modality.replace(":", "_").replace("/", "_")
            rr.log(
                f"modality/{safe}/response_normalized",
                Scalars(float(value)),
                recording=rec,
            )

        # Action vectors + per-frame action delta to expert + frame-to-
        # frame action shift magnitude. The shift magnitude highlights
        # "abrupt change moments" — common right before deployment
        # failures (the policy lurched into the wrong direction).
        baseline_action = _find_baseline_action(scene, per_axis_results, i)
        if i > 0 and baseline_action is not None:
            prev_baseline = _find_baseline_action(
                trajectory.frames[i - 1], per_axis_results, i - 1,
            )
            if prev_baseline is not None and len(prev_baseline) == len(baseline_action):
                shift = float(np.linalg.norm(baseline_action - prev_baseline))
                rr.log("predictions/frame_to_frame_shift",
                       Scalars(shift), recording=rec)
        expert = scene.metadata.get("expert_action")
        if baseline_action is not None:
            dim_names = _action_dim_names(scene, len(baseline_action))
            for d, name in enumerate(dim_names):
                rr.log(f"predictions/action/{name}",
                       Scalars(float(baseline_action[d])), recording=rec)
        if expert is not None:
            expert_arr = np.asarray(expert, dtype=np.float32)
            dim_names = _action_dim_names(scene, len(expert_arr))
            for d, name in enumerate(dim_names):
                rr.log(f"predictions/expert/{name}",
                       Scalars(float(expert_arr[d])), recording=rec)
            if baseline_action is not None:
                n = min(len(baseline_action), len(expert_arr))
                delta = baseline_action[:n] - expert_arr[:n]
                rr.log("predictions/delta_to_expert",
                       Scalars(float(np.linalg.norm(delta))), recording=rec)
                per_dim_names = _action_dim_names(scene, n)
                for d, name in enumerate(per_dim_names):
                    rr.log(f"predictions/delta_per_dim/{name}",
                           Scalars(float(delta[d])), recording=rec)

    # 2. Diagnostic tracks — one set of channels per axis.
    for axis, traj_result in per_axis_results.items():
        for i, r in enumerate(traj_result.per_frame):
            rr.set_time("frame_time", duration=i / fps, recording=rec)
            rr.set_time(
                "frame_index", sequence=traj_result.frame_indices[i], recording=rec,
            )
            score = r.scalar_score
            if score == score:  # not NaN
                rr.log(f"diagnostics/{axis}/score",
                       Scalars(float(score)), recording=rec)
            # Severity is conveyed only by colour. We never log the
            # severity word as text — the Finding below carries the
            # plain-English verdict.
            color = _SEVERITY_RGB.get(r.severity, (128, 128, 128))
            if r.finding is not None:
                f = r.finding
                text = (
                    f"OBSERVED:\n{f.observed}\n\n"
                    f"MEANING:\n{f.meaning}\n\n"
                    f"NEXT STEP:\n{f.next_step}"
                )
                rr.log(
                    f"diagnostics/{axis}/finding",
                    rr.TextDocument(text),
                    recording=rec,
                )
                rr.log(
                    f"diagnostics/{axis}/headline",
                    rr.TextLog(f.observed[:120], color=color),
                    recording=rec,
                )
            elif r.explanation:
                rr.log(
                    f"diagnostics/{axis}/explanation",
                    rr.TextDocument(r.explanation),
                    recording=rec,
                )

    rr.save(str(out_path), recording=rec)
    return out_path


def _colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min-max normalize + map to a viridis-ish RGB uint8 image."""
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim == 3 and h.shape[-1] == 1:
        h = h[..., 0]
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 1e-9:
        normalized = np.zeros_like(h)
    else:
        normalized = (h - lo) / (hi - lo)
    # Simple viridis-ish ramp: low=purple, mid=teal, high=yellow.
    r = (np.clip(normalized * 1.4 - 0.2, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(normalized * 1.0 + 0.05, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.0 - normalized, 0, 1) * 180).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Bool mask → red overlay RGB."""
    m = np.asarray(mask).astype(bool)
    out = np.zeros((*m.shape, 3), dtype=np.uint8)
    out[m] = (220, 50, 50)
    return out


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
