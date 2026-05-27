"""Rerun ``.rrd`` exporter — the killer integration.

A roboticist who already uses Rerun for rollout playback opens the
``.rrd`` and sees Emboviz diagnostics laid out for them: each camera with
its attention / sensitivity / target overlays *composited on the image*
(not as separate panels), diagnostic scores + predicted action + modality
response as time-series plots, and the plain-English findings (including
the attention-drift / trajectory-level axes) in a text panel.

Design notes (why it looks the way it does):

  • Overlays are logged as **RGBA images under the camera's own entity
    subtree** (``world/camera/<cam>/attention`` etc.), so a single
    ``Spatial2DView`` rooted at the camera composites them on top of the
    RGB by ``draw_order`` (higher = on top; Rerun docs). The alpha channel
    encodes magnitude, so "no signal" is *transparent*, never a flat blue
    square. A heatmap that is entirely flat/zero is **not logged at all**.

  • The layout is a curated ``rrb.Blueprint`` (sent via
    ``rr.save(default_blueprint=...)``). Without it Rerun auto-arranges
    ~50 entity paths into an unreadable grid.

  • A camera with no measurable response still shows its raw RGB (useful
    context) but gets no overlay — absence reads as absence.

Targets rerun-sdk >= 0.23 (unified ``set_time``; ``Scalars`` archetype;
``RecordingStream`` constructor). Tested against 0.32.x.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Trajectory
from emboviz.diagnostics.trajectory import TrajectoryDiagnosticResult


# Severity → RGB, used only for the time-series line / headline colour.
_SEVERITY_RGB = {
    Severity.CRITICAL: (201, 42, 42),
    Severity.MODERATE: (230, 119, 0),
    Severity.INFO:     (25, 113, 194),
    Severity.PASS:     (47, 158, 68),
    Severity.UNKNOWN:  (134, 142, 150),
}

# Draw-order layers within a camera's 2D view (higher = on top).
_DRAW_RGB         = 0.0
_DRAW_SENSITIVITY = 1.5
_DRAW_ATTENTION   = 2.0
_DRAW_MASK        = 2.5
_DRAW_BOX         = 3.0

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
    trajectory_axis_results: Optional[dict[str, dict]] = None,
) -> Path:
    """Emit an .rrd with composited camera overlays + diagnostic plots + a
    curated blueprint.

    Args mostly as before; new:
        trajectory_axis_results: ``{axis_name → {severity, scalar_score,
            explanation}}`` for trajectory-level axes (e.g. attention_drift)
            so they appear in the findings panel — they have no per-frame
            track but DID produce a verdict.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as e:
        raise ImportError(
            "Rerun export requires the `rerun-sdk` package (>=0.23). "
            "Install with: uv pip install 'rerun-sdk>=0.23'"
        ) from e

    if not hasattr(rr, "RecordingStream") or not hasattr(rr, "set_time"):
        raise RuntimeError(
            "rerun-sdk too old. Install >=0.23: uv pip install 'rerun-sdk>=0.23'"
        )
    Scalars = rr.Scalars

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recording_id = trajectory.source or trajectory.episode_id or "rollout"
    rec = rr.RecordingStream(application_id=application_id, recording_id=recording_id)
    fps = trajectory.fps if trajectory.fps > 0 else 5.0

    attention_per_frame          = attention_per_frame          or {}
    sensitivity_per_frame        = sensitivity_per_frame        or {}
    target_mask_per_frame        = target_mask_per_frame        or {}
    target_detection_per_frame   = target_detection_per_frame   or {}
    masked_image_per_frame       = masked_image_per_frame       or {}
    modality_response_per_frame  = modality_response_per_frame  or {}
    trajectory_axis_results      = trajectory_axis_results      or {}

    # Track what actually got logged, to build an accurate blueprint.
    cams_all: list[str] = []
    cam_overlays: dict[str, set[str]] = {}   # cam → {"attention","sensitivity","dino_mask"}
    cams_with_masked: set[str] = set()
    dead_cams: set[str] = set()              # blank/placeholder feeds we skip

    def _frame_data(store: PerFrameByCamera, frame_idx: int, i: int) -> dict[str, np.ndarray]:
        if frame_idx in store:
            return store[frame_idx]
        if i in store:
            return store[i]
        return {}

    # ── 1. Camera streams + composited overlays ───────────────────────────
    for i, scene in enumerate(trajectory.frames):
        frame_idx = trajectory.frame_indices[i]
        rr.set_time("frame_time", duration=i / fps, recording=rec)
        rr.set_time("frame_index", sequence=frame_idx, recording=rec)

        scene_cameras = set(scene.observations.images.keys())
        cam_hw: dict[str, tuple[int, int]] = {}
        for cam_name, rgb in scene.observations.images.items():
            arr = np.asarray(rgb.data)
            if _is_dead_feed(arr):
                # Blank / placeholder camera stream (some dataset episodes
                # carry inactive secondary cameras as all-black frames).
                # Don't show a dead black panel — skip it entirely.
                dead_cams.add(cam_name)
                continue
            cam_hw[cam_name] = (arr.shape[0], arr.shape[1])
            if cam_name not in cams_all:
                cams_all.append(cam_name)
            rr.log(
                f"world/camera/{cam_name}/rgb",
                rr.Image(arr, draw_order=_DRAW_RGB),
                recording=rec,
            )

        # Attention overlay (RGBA, magnitude→alpha; skip if flat).
        for cam, attn in _frame_data(attention_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"attention_per_frame logs camera '{cam}' at frame {frame_idx} "
                    f"but the scene only has {sorted(scene_cameras)}."
                )
            if cam not in cam_hw:
                continue  # dead/skipped camera — no base image to overlay on
            rgba = _heatmap_rgba(attn, cam_hw[cam], cmap_name="turbo")
            if rgba is not None:
                rr.log(
                    f"world/camera/{cam}/attention",
                    rr.Image(rgba, draw_order=_DRAW_ATTENTION, opacity=0.85),
                    recording=rec,
                )
                cam_overlays.setdefault(cam, set()).add("attention")

        # Sensitivity overlay.
        for cam, sens in _frame_data(sensitivity_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"sensitivity_per_frame logs camera '{cam}' at frame {frame_idx} "
                    f"but the scene only has {sorted(scene_cameras)}."
                )
            if cam not in cam_hw:
                continue
            rgba = _heatmap_rgba(sens, cam_hw[cam], cmap_name="viridis")
            if rgba is not None:
                rr.log(
                    f"world/camera/{cam}/sensitivity",
                    rr.Image(rgba, draw_order=_DRAW_SENSITIVITY, opacity=0.8),
                    recording=rec,
                )
                cam_overlays.setdefault(cam, set()).add("sensitivity")

        # GroundingDINO mask (red, semi-transparent) + per-instance boxes.
        for cam, tmask in _frame_data(target_mask_per_frame, frame_idx, i).items():
            if cam not in cam_hw:
                continue
            rgba = _mask_rgba(tmask, cam_hw[cam])
            if rgba is not None:
                rr.log(
                    f"world/camera/{cam}/dino_mask/mask",
                    rr.Image(rgba, draw_order=_DRAW_MASK),
                    recording=rec,
                )
                cam_overlays.setdefault(cam, set()).add("dino_mask")
        for cam, det in _frame_data(target_detection_per_frame, frame_idx, i).items():
            if cam not in cam_hw:
                continue
            label = det.get("label", "")
            # Draw every detected instance box (all spoons), not just the union.
            all_boxes = det.get("all_boxes") or [det.get("bbox", (0, 0, 0, 0))]
            all_scores = det.get("all_scores") or [float(det.get("confidence", 0.0))]
            try:
                rr.log(
                    f"world/camera/{cam}/dino_mask/boxes",
                    rr.Boxes2D(
                        array=[list(b) for b in all_boxes],
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"{label} ({s:.2f})" for s in all_scores],
                        draw_order=_DRAW_BOX,
                    ),
                    recording=rec,
                )
                cam_overlays.setdefault(cam, set()).add("dino_mask")
            except Exception:
                pass

        # Masked-input full frames (what the model saw) — separate subtree
        # so they don't composite onto the live camera.
        for cam, fills in _frame_data(masked_image_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                continue
            for fill_mode, masked_arr in fills.items():
                rr.log(
                    f"memorization/{cam}/{fill_mode}",
                    rr.Image(np.asarray(masked_arr)),
                    recording=rec,
                )
                cams_with_masked.add(cam)

        # Per-modality response scalars.
        mod_resp = (
            modality_response_per_frame.get(frame_idx)
            or modality_response_per_frame.get(i) or {}
        )
        for modality, value in mod_resp.items():
            safe = modality.replace(":", "_").replace("/", "_")
            rr.log(f"plots/modality/{safe}", Scalars(float(value)), recording=rec)

        # Predicted action + expert + per-frame shift.
        baseline_action = _find_baseline_action(scene, per_axis_results, i)
        if i > 0 and baseline_action is not None:
            prev = _find_baseline_action(trajectory.frames[i - 1], per_axis_results, i - 1)
            if prev is not None and len(prev) == len(baseline_action):
                rr.log("plots/action_shift",
                       Scalars(float(np.linalg.norm(baseline_action - prev))), recording=rec)
        expert = scene.metadata.get("expert_action")
        if baseline_action is not None:
            for d, name in enumerate(_action_dim_names(scene, len(baseline_action))):
                rr.log(f"plots/action/{name}", Scalars(float(baseline_action[d])), recording=rec)
        if expert is not None:
            expert_arr = np.asarray(expert, dtype=np.float32)
            for d, name in enumerate(_action_dim_names(scene, len(expert_arr))):
                rr.log(f"plots/expert/{name}", Scalars(float(expert_arr[d])), recording=rec)

    # ── 2. Diagnostic per-frame score tracks + per-frame finding text ─────
    for axis, traj_result in per_axis_results.items():
        for i, r in enumerate(traj_result.per_frame):
            rr.set_time("frame_time", duration=i / fps, recording=rec)
            rr.set_time("frame_index", sequence=traj_result.frame_indices[i], recording=rec)
            score = r.scalar_score
            if score == score:  # not NaN
                rr.log(f"plots/diagnostics/{axis}", Scalars(float(score)), recording=rec)
            if r.finding is not None:
                f = r.finding
                rr.log(
                    f"findings_per_frame/{axis}",
                    rr.TextDocument(
                        f"**{axis}**\n\n"
                        f"**Observed:** {f.observed}\n\n"
                        f"**Meaning:** {f.meaning}\n\n"
                        f"**Next step:** {f.next_step}",
                        media_type=rr.MediaType.MARKDOWN,
                    ),
                    recording=rec,
                )

    # ── 3. Findings summary (static, all axes incl. trajectory-level) ─────
    summary_md = _build_findings_markdown(
        trajectory, per_axis_results, trajectory_axis_results,
    )
    if dead_cams:
        summary_md += (
            "\n\n---\n*Cameras with no live feed (blank/placeholder streams, "
            f"skipped): {sorted(dead_cams)}*\n"
        )
    rr.log("findings", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
           recording=rec, static=True)

    # ── 4. Blueprint ──────────────────────────────────────────────────────
    blueprint = _build_blueprint(rrb, cams_all, cam_overlays, sorted(cams_with_masked))

    # Force the curated blueprint ACTIVE (not just default). rr.save's
    # default_blueprint only sets the default — the viewer keeps any
    # previously-cached *active* blueprint for this application_id, which
    # (after a schema change like ours) points at entity paths that no
    # longer exist → empty views. send_blueprint(make_active=True) makes
    # the viewer apply our layout on load regardless of cache.
    rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=rec)
    rr.save(str(out_path), recording=rec)
    return out_path


# ── helpers ────────────────────────────────────────────────────────────────

def _heatmap_rgba(
    heatmap: np.ndarray, target_hw: tuple[int, int],
    *, cmap_name: str = "turbo", gamma: float = 0.7,
) -> Optional[np.ndarray]:
    """Normalize → colormap → RGBA, alpha encodes magnitude. Upsampled to the
    camera resolution. Returns None for a flat/zero heatmap (so we never log a
    misleading solid-colour overlay for a camera with no measurable response)."""
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim == 3 and h.shape[-1] == 1:
        h = h[..., 0]
    if h.ndim != 2:
        return None
    lo, hi = float(np.nanmin(h)), float(np.nanmax(h))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9:
        return None  # no signal → don't log
    norm = (h - lo) / (hi - lo)

    # Upsample to camera HxW (bilinear) so the overlay aligns with the image.
    from PIL import Image as PILImage
    H, W = target_hw
    norm_u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    norm = (
        np.asarray(PILImage.fromarray(norm_u8).resize((W, H), PILImage.BILINEAR))
        .astype(np.float32) / 255.0
    )

    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    rgba = (cmap(norm) * 255).astype(np.uint8)        # (H, W, 4)
    rgba[..., 3] = (np.clip(norm, 0, 1) ** gamma * 255).astype(np.uint8)  # alpha = magnitude
    return rgba


def _mask_rgba(mask: np.ndarray, target_hw: tuple[int, int]) -> Optional[np.ndarray]:
    """Bool mask → red RGBA (transparent off-mask). Upsampled. None if empty."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(bool)
    if not m.any():
        return None
    from PIL import Image as PILImage
    H, W = target_hw
    if m.shape != (H, W):
        m = np.asarray(
            PILImage.fromarray(m.astype(np.uint8) * 255).resize((W, H), PILImage.NEAREST)
        ) > 127
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[m] = (220, 50, 50, 150)
    return out


def _build_findings_markdown(
    trajectory: Trajectory,
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    trajectory_axis_results: dict[str, dict],
) -> str:
    """Plain-English findings for every axis — per-axis (per-frame Finding
    aggregated) AND trajectory-level axes (attention_drift etc.), so the
    headline diagnostics are never silently dropped from the viewer."""
    lines = ["# Emboviz findings", ""]
    instr = getattr(trajectory, "instruction", None) or (
        trajectory.frames[0].instruction if trajectory.frames else None
    )
    if instr:
        lines += [f"**Instruction:** {instr}", ""]

    for axis, traj in per_axis_results.items():
        finding = None
        for r in traj.per_frame:
            if r.finding is not None:
                finding = r.finding
                break
        lines.append(f"## {axis}")
        if finding is not None:
            lines += [
                f"- **Observed:** {finding.observed}",
                f"- **Meaning:** {finding.meaning}",
                f"- **Next step:** {finding.next_step}",
                "",
            ]
        else:
            lines += ["- (no per-frame finding)", ""]

    for axis, info in trajectory_axis_results.items():
        lines.append(f"## {axis}")
        expl = info.get("explanation", "")
        score = info.get("scalar_score")
        lines.append(f"- {expl}" if expl else "- (trajectory-level axis)")
        if score is not None:
            lines.append(f"- score: {score}")
        lines.append("")
    return "\n".join(lines)


def _is_dead_feed(arr: np.ndarray) -> bool:
    """True if a camera frame carries no information (all-black / near-constant)
    — some dataset episodes ship inactive secondary cameras as blank frames.
    We skip those rather than show a dead black panel."""
    a = np.asarray(arr, dtype=np.float32)
    return float(a.std()) < 1.0


def _build_blueprint(rrb, cams_all, cam_overlays, cams_with_masked):
    """Curated layout: cameras (each a tabbed RGB / overlay view) on the left,
    the plain-English findings panel on the right.

    ``auto_views=False`` is essential: it tells the viewer to show ONLY the
    views we define. Otherwise Rerun auto-adds a view per un-placed entity
    (per-frame findings, masked-input frames, action scalars, …) on top of
    ours — which is what made the first version look like a wall of panels.
    """
    camera_views = []
    for cam in cams_all:
        origin = f"world/camera/{cam}"
        overlays = cam_overlays.get(cam, set())
        tabs = [rrb.Spatial2DView(origin=origin, contents=["+ $origin/rgb"], name="RGB")]
        if "attention" in overlays:
            tabs.append(rrb.Spatial2DView(
                origin=origin, contents=["+ $origin/rgb", "+ $origin/attention"],
                name="Attention"))
        if "sensitivity" in overlays:
            tabs.append(rrb.Spatial2DView(
                origin=origin, contents=["+ $origin/rgb", "+ $origin/sensitivity"],
                name="Sensitivity"))
        if "dino_mask" in overlays:
            tabs.append(rrb.Spatial2DView(
                origin=origin, contents=["+ $origin/rgb", "+ $origin/dino_mask/**"],
                name="DINO mask"))
        camera_views.append(rrb.Tabs(*tabs, name=cam) if len(tabs) > 1 else tabs[0])

    if cams_with_masked:
        masked_views = [
            rrb.Spatial2DView(origin=f"memorization/{cam}", name=f"masked:{cam}")
            for cam in cams_with_masked
        ]
        camera_views.append(rrb.Tabs(*masked_views, name="masked inputs")
                            if len(masked_views) > 1 else masked_views[0])

    left = rrb.Grid(*camera_views) if camera_views else rrb.Spatial2DView(origin="world")
    right = rrb.TextDocumentView(origin="findings", name="Findings (plain English)")

    return rrb.Blueprint(
        rrb.Horizontal(left, right, column_shares=[3, 2]),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
        collapse_panels=True,
        auto_views=False,
    )


def _find_baseline_action(
    scene, per_axis_results: dict[str, TrajectoryDiagnosticResult], frame_idx: int,
) -> Optional[np.ndarray]:
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
    if (
        scene.profile is not None
        and scene.profile.action is not None
        and scene.profile.action.dim_names is not None
        and len(scene.profile.action.dim_names) >= dim
    ):
        return list(scene.profile.action.dim_names[:dim])
    return [f"d{i}" for i in range(dim)]
