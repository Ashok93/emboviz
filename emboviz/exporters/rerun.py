"""Rerun ``.rrd`` exporter — user-centric tabbed dashboard, one tab per metric.

The user opens the ``.rrd`` and sees one tab per diagnostic. Every tab
follows the same skeleton — predictability over per-metric cleverness:

    ┌─ What this measures (markdown) ─┐
    │  README-quality 2–3 sentences   │
    ├─ How to read this tab ──────────┤
    │  Green = X.  Yellow = Y.        │
    │  Red = Z.   Hot = ... etc.      │
    ├─ Main visualization ────────────┤
    │  (heatmaps + per-camera         │
    │   overlays, all cameras shown)  │
    ├─ Per-frame details ─────────────┤
    │  Numbers for the cursor frame.  │
    └─────────────────────────────────┘

Design rules (locked):

  • Three-tier verdict labels everywhere — **OK / MIXED / WORTH A LOOK**.
    No "CRITICAL". No new invented words.
  • Three colors — green / yellow / red — used consistently for verdict
    ribbons and the categorical modality-dropout heatmap.
  • "couldn't test" is a per-frame state only (gray cell), shown when
    detection failed on that frame or the intervention was too weak.
    It is NEVER an outcome for an entire metric — diagnostics that
    structurally don't apply to the model don't appear as tabs at all.
  • All cameras are shown on every tab that has per-camera signal
    (attention, sensitivity, memorization). No exceptions.
  • Continuous magnitude signals (attention heatmap, sensitivity
    heatmap) use ``turbo`` because the underlying signal is continuous
    and the user reads "hot region = used by model" naturally.

Targets the pinned rerun-sdk >= 0.33, < 0.34 (unified ``set_time``;
``Scalars`` / ``BarChart`` archetypes; ``RecordingStream`` constructor;
blueprint ``Tabs``/``Vertical``/``Grid``). The .rrd on-disk format is tied
to rerun's minor version, so the pin is exact-minor on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Trajectory
from emboviz.diagnostics.trajectory import TrajectoryDiagnosticResult


# ─── Verdict palette ──────────────────────────────────────────────────────
# Three-tier system that surfaces to the user. Maps from internal Severity
# (5-tier sort key, used for failure-moment detection and JSON output) to
# the 3-tier dashboard label + color.
#
# Severity.PASS / Severity.INFO → "OK" (green)
# Severity.MODERATE             → "MIXED" (yellow)
# Severity.CRITICAL             → "WORTH A LOOK" (red)
# Severity.UNKNOWN              → "couldn't test" (gray, frame-only)

VERDICT_OK_RGB         = (47, 158, 68)
VERDICT_MIXED_RGB      = (230, 119, 0)
VERDICT_LOOK_RGB       = (201, 42, 42)
VERDICT_NOTEST_RGB     = (134, 142, 150)

# Severity → (label, RGB) lookup. Kept internal — never rendered as text
# anywhere in the dashboard. Labels are user-facing strings.
_VERDICT_BY_SEVERITY: dict[Severity, tuple[str, tuple[int, int, int]]] = {
    Severity.PASS:     ("OK",            VERDICT_OK_RGB),
    Severity.INFO:     ("OK",            VERDICT_OK_RGB),
    Severity.MODERATE: ("MIXED",         VERDICT_MIXED_RGB),
    Severity.CRITICAL: ("WORTH A LOOK",  VERDICT_LOOK_RGB),
    Severity.UNKNOWN:  ("couldn't test", VERDICT_NOTEST_RGB),
}


def _verdict(sev: Severity) -> tuple[str, tuple[int, int, int]]:
    return _VERDICT_BY_SEVERITY.get(sev, ("couldn't test", VERDICT_NOTEST_RGB))


# Draw-order layers within a camera's 2D view (higher = on top).
_DRAW_RGB         = 0.0
_DRAW_SENSITIVITY = 1.5
_DRAW_ATTENTION   = 2.0
_DRAW_MASK        = 2.5
_DRAW_BOX         = 3.0

# Type aliases for the per-frame artifact dicts the runner passes in.
PerFrameByCamera         = dict[int, dict[str, np.ndarray]]
PerFrameDetection        = dict[int, dict[str, dict]]
PerFrameMaskedImage      = dict[int, dict[str, dict[str, np.ndarray]]]
PerFrameModalityResponse = dict[int, dict[str, float]]


# ──────────────────────────────────────────────────────────────────────────
# Tab description text — kept in one place so wording is consistent and
# editable. Markdown, rendered by the TextDocumentView at the top of each
# tab. README-quality language; no jargon; explains how to read the
# visualizations + what the three verdict colors mean for THIS metric.
# ──────────────────────────────────────────────────────────────────────────

_DOC_MEMORIZATION = """## Memorization — is the model looking at the target?

We located the manipulated target on every camera (SAM 3 or the per-frame
annotations you provided), masked it with two independent fills (channel
mean + Gaussian blur), and measured how much the model's action changed.

**How to read this tab**

- **Timeline strip** colors every frame by the verdict on that frame.
  - 🟢 **OK** — masking the target moved the action substantially. The
    model is reading the scene.
  - 🟡 **MIXED** — partial response, between sampling noise and a strong
    signal.
  - 🔴 **WORTH A LOOK** — action barely moved. Memorized signature: the
    policy may be predicting from state + instruction + history without
    visually consuming the target.
  - ⚪ **couldn't test** — target not detected on that frame, or the mask
    didn't actually change the image (fill ≈ target color).
- **Original | Masked** per camera lets you visually verify the mask
  covered the right object. If SAM 3 (or your annotation) caught the
  wrong thing, the verdict is meaningless — that's why we show it.
- **Δaction over frames** is the magnitude of the action shift,
  normalized to "fraction of a typical action". Higher = more visually
  grounded.

Numbers are normalized by the model's typical action magnitude (set by
the per-trajectory calibration), so the same threshold has the same
meaning across models.
"""

_DOC_MODALITY = """## Modality dropout — which inputs is the model using?

For each declared input modality (every camera, the instruction, robot
state, gripper, action history), we substitute it with K real values
sampled from OTHER episodes in your dataset, and measure how much the
action changes. If the action barely moves, that input isn't being used.

**How to read this tab**

- **Verdict heatmap** — rows are modalities, columns are frames, color
  is the verdict on that (modality, frame) cell.
  - 🟢 **OK** — strong response. The model is consuming this input.
  - 🟡 **MIXED** — partial response (between noise and the "strong"
    threshold).
  - 🔴 **WORTH A LOOK** — response below the noise floor. The model
    appears to ignore this input on this frame.
  - ⚪ **couldn't test** — substitute too similar to the current value
    (the pool didn't have enough variety) or pool empty for this modality.
- **Mean response per modality** (across the whole episode) — the
  episode-level signal at a glance. A row that's red across most frames
  is a candidate for "the model isn't using this input."
- A modality that's *supposed* to matter (instruction for a
  language-conditioned task, the wrist camera for fine manipulation) but
  reads consistently 🔴 is the strongest signal in this dataset.
"""

_DOC_ATTENTION = """## Attention — where is the model looking?

We extract the model's internal attention map at each frame and overlay
the cleaned per-camera version on the live image (mid-layer
literature-backed filter + sink masking, per LITERATURE.md §4).

**How to read this tab**

- **Per-camera overlays** — hotter = more attention. Should land on the
  target object / gripper / task-relevant region. If attention sits on
  the background or off-frame, the policy isn't visually grounded.
- **Attention centroid drift (px)** — pixel distance the attention
  centroid moves between consecutive frames. Stable focus = small
  numbers; wandering = large jumps. Spikes correlate with the few frames
  before a failure (model loses its anchor).
- Drift is measured on the **primary** camera (configurable). Use the
  per-camera overlay to see what's drifting.

The clean map already discounts softmax routing artifacts (RoPE / BOS
sinks); for the unfiltered raw, see `attn.image_weights(cam)` in code.
"""

_DOC_SENSITIVITY = """## Scene sensitivity — which regions drive the action?

We sweep an N×N grid of patches across every camera. For each cell we
mask it with the channel mean and measure how much the action changes.
The resulting heatmap shows which pixels actually drive the model's
decision.

**How to read this tab**

- **Per-camera sensitivity heatmap** overlaid on the live image:
  - **Hot** cells — the model reacts when those pixels are hidden. Used.
  - **Cold** cells — the model doesn't care about those pixels.
- Hot regions should align with the manipulated object / gripper / task
  area. If the hot regions are spread thinly across the background, the
  policy is relying on scene gist (brittle).
- **Concentration per camera** — a single number from 0 (totally diffuse)
  to 1 (one cell carries all the signal). Higher = more focused.

We subtract the per-call sampling noise floor before drawing the
heatmap; flat-zero cameras don't get an overlay (absence reads as
absence, not as a misleading blue square).
"""

_DOC_CHUNK = """## Chunk consistency — can you trust the multi-step plan?

Models that emit an action *chunk* (π0, OpenVLA-OFT, GR00T, ACT,
Diffusion Policy) predict the next N steps at every frame. We compare
the model's prediction for step *k* made at frame *t* against what it
actually emits at frame *t+k*. If they disagree, the chunk beyond
step 0 is best-guess and you must re-plan every step.

**How to read this tab**

- **Per-frame disagreement** is the normalized L2 distance between
  ``chunk[t][k]`` and ``chunk[t+k][0]``. Lower = more consistent
  lookahead.
- **Safely-committable horizon** — how many steps into the chunk you
  can trust before disagreement crosses the threshold. Higher is
  better.

This metric is hidden from the dashboard when your model is
single-step (no ``action_chunk``).
"""

_DOC_FINDINGS = """## Findings — plain English for every metric

A single page summarizing every diagnostic for every axis the framework
could compute. For each axis you get:

- **Observed** — what the test actually measured (numbers + the most
  representative per-frame example).
- **Meaning** — what the result tells you about the model.
- **Next step** — a concrete action: what to try, what to check, where
  to drill.

If a metric was inconclusive on every frame of this episode, the
diagnostic says so — we do not pretend a number when one isn't there.
"""


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────

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
    """Emit an .rrd with one tab per metric, plus a Findings summary.

    Tabs are only included when the underlying diagnostic produced a
    result for this episode. Metrics that don't apply to the model
    (e.g. chunk consistency on a single-step policy) are silently
    absent — they never appear as empty placeholders.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as e:
        raise ImportError(
            "Rerun export requires the `rerun-sdk` package. It ships with "
            "emboviz core — reinstall with: uv pip install 'emboviz' "
            "(rerun-sdk>=0.33,<0.34)."
        ) from e

    if not hasattr(rr, "RecordingStream"):
        raise RuntimeError(
            "rerun-sdk too old for the export API. Install the pinned range: "
            "uv pip install 'rerun-sdk>=0.33,<0.34'"
        )

    # rerun-sdk 0.23 renamed Scalar → Scalars (plural) and unified the
    # per-axis ``set_time_sequence`` / ``set_time_seconds`` calls into a
    # single ``set_time(name, *, sequence|duration)``. Core pins
    # rerun-sdk>=0.33,<0.34 (see pyproject), which always has the unified
    # ``set_time`` and the ``Scalars`` archetype — so no version shim, and
    # no support for the <0.33 the pin forbids.
    Scalars = rr.Scalars

    def _set_time(name: str, *, sequence=None, duration=None) -> None:
        if sequence is not None:
            rr.set_time(name, sequence=sequence, recording=rec)
        elif duration is not None:
            rr.set_time(name, duration=duration, recording=rec)

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

    # ── 0. Static descriptions ────────────────────────────────────────────
    # One markdown doc per metric tab. Always logged so the blueprint can
    # reference them; the dashboard only shows the ones whose tab is built.
    for path, md in (
        ("docs/memorization", _DOC_MEMORIZATION),
        ("docs/modality",     _DOC_MODALITY),
        ("docs/attention",    _DOC_ATTENTION),
        ("docs/sensitivity",  _DOC_SENSITIVITY),
        ("docs/chunk",        _DOC_CHUNK),
        ("docs/findings",     _DOC_FINDINGS),
    ):
        rr.log(path, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN),
               recording=rec, static=True)

    # Track which cameras are live (skip blank/placeholder feeds) and
    # which tabs ended up populated, so the blueprint only references
    # views that have data.
    cams_all: list[str] = []
    dead_cams: set[str] = set()
    cams_with_attention: set[str] = set()
    cams_with_sensitivity: set[str] = set()
    cams_with_mask: set[str] = set()
    cams_with_masked_image: set[str] = set()

    def _frame_data(store: PerFrameByCamera, frame_idx: int, i: int) -> dict[str, np.ndarray]:
        if frame_idx in store:
            return store[frame_idx]
        if i in store:
            return store[i]
        return {}

    # ── 1. Per-frame camera streams + overlays ────────────────────────────
    for i, scene in enumerate(trajectory.frames):
        frame_idx = trajectory.frame_indices[i]
        _set_time("frame_time", duration=i / fps)
        _set_time("frame_index", sequence=frame_idx)

        scene_cameras = set(scene.observations.images.keys())
        cam_hw: dict[str, tuple[int, int]] = {}
        for cam_name, rgb in scene.observations.images.items():
            arr = np.asarray(rgb.data)
            if _is_dead_feed(arr):
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
            # Memorization tab uses its own per-camera subtree so the
            # mask overlay there can co-exist with the attention overlay
            # in the Attention tab on the same underlying RGB.
            rr.log(
                f"memorization/{cam_name}/original",
                rr.Image(arr, draw_order=_DRAW_RGB),
                recording=rec,
            )

        # Attention overlay (RGBA, magnitude → alpha; skip if flat).
        for cam, attn in _frame_data(attention_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"attention_per_frame logs camera '{cam}' at frame "
                    f"{frame_idx} but the scene only has "
                    f"{sorted(scene_cameras)}."
                )
            if cam not in cam_hw:
                continue
            rgba = _heatmap_rgba(attn, cam_hw[cam], cmap_name="turbo")
            if rgba is not None:
                rr.log(
                    f"world/camera/{cam}/attention",
                    rr.Image(rgba, draw_order=_DRAW_ATTENTION, opacity=0.85),
                    recording=rec,
                )
                cams_with_attention.add(cam)

        # Sensitivity overlay.
        for cam, sens in _frame_data(sensitivity_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                raise ValueError(
                    f"sensitivity_per_frame logs camera '{cam}' at frame "
                    f"{frame_idx} but the scene only has "
                    f"{sorted(scene_cameras)}."
                )
            if cam not in cam_hw:
                continue
            rgba = _heatmap_rgba(sens, cam_hw[cam], cmap_name="turbo")
            if rgba is not None:
                rr.log(
                    f"world/camera/{cam}/sensitivity",
                    rr.Image(rgba, draw_order=_DRAW_SENSITIVITY, opacity=0.85),
                    recording=rec,
                )
                cams_with_sensitivity.add(cam)

        # Target mask overlay (red, semi-transparent) on the Memorization
        # tab's per-camera RGB. Plus per-instance bounding boxes for
        # transparency about what the detector caught.
        for cam, tmask in _frame_data(target_mask_per_frame, frame_idx, i).items():
            if cam not in cam_hw:
                continue
            rgba = _mask_rgba(tmask, cam_hw[cam])
            if rgba is not None:
                rr.log(
                    f"memorization/{cam}/mask",
                    rr.Image(rgba, draw_order=_DRAW_MASK),
                    recording=rec,
                )
                cams_with_mask.add(cam)
        for cam, det in _frame_data(target_detection_per_frame, frame_idx, i).items():
            if cam not in cam_hw:
                continue
            label = det.get("label", "")
            all_boxes = det.get("all_boxes") or [det.get("bbox", (0, 0, 0, 0))]
            all_scores = det.get("all_scores") or [float(det.get("confidence", 0.0))]
            try:
                rr.log(
                    f"memorization/{cam}/boxes",
                    rr.Boxes2D(
                        array=[list(b) for b in all_boxes],
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"{label} ({s:.2f})" for s in all_scores],
                        draw_order=_DRAW_BOX,
                    ),
                    recording=rec,
                )
                cams_with_mask.add(cam)
            except Exception:
                pass

        # Masked images (what the model actually saw, per fill mode).
        for cam, fills in _frame_data(masked_image_per_frame, frame_idx, i).items():
            if cam not in scene_cameras:
                continue
            for fill_mode, masked_arr in fills.items():
                rr.log(
                    f"memorization/{cam}/masked_{fill_mode}",
                    rr.Image(np.asarray(masked_arr)),
                    recording=rec,
                )
                cams_with_masked_image.add(cam)

        # Per-modality response time-series — one line per modality on
        # the Modality tab's chart.
        mod_resp = (
            modality_response_per_frame.get(frame_idx)
            or modality_response_per_frame.get(i)
            or {}
        )
        for modality, value in mod_resp.items():
            safe = modality.replace(":", "_").replace("/", "_")
            rr.log(f"plots/modality/{safe}", Scalars(float(value)), recording=rec)

    # ── 2. Per-axis time-series + per-frame finding text + verdict ────────
    # For every per-axis diagnostic, log the scalar score over time AND
    # the current-frame Finding markdown (cursor-driven on the dashboard).
    for axis, traj_result in per_axis_results.items():
        for i, r in enumerate(traj_result.per_frame):
            _set_time("frame_time", duration=i / fps)
            _set_time("frame_index", sequence=traj_result.frame_indices[i])
            score = r.scalar_score
            if score == score:  # not NaN
                rr.log(
                    f"plots/diagnostics/{axis}",
                    Scalars(float(score)),
                    recording=rec,
                )
            if r.finding is not None:
                f = r.finding
                rr.log(
                    f"current_frame/{axis}",
                    rr.TextDocument(
                        _current_frame_markdown(axis, r, f),
                        media_type=rr.MediaType.MARKDOWN,
                    ),
                    recording=rec,
                )

    # ── 3. Per-axis verdict ribbons (RGBA strips, color = verdict) ────────
    # One RGBA strip per axis, 1 row × N_frames, drawn from the per-frame
    # severities. Sits at the top of each metric tab so the user sees
    # "where in the episode is the verdict different" at a glance.
    for axis, traj_result in per_axis_results.items():
        ribbon = _verdict_ribbon(traj_result)
        rr.log(
            f"ribbon/{axis}",
            rr.Image(ribbon, draw_order=_DRAW_BOX),
            recording=rec, static=True,
        )

    # ── 4. Modality dropout — categorical verdict heatmap ─────────────────
    # M-row × N-col RGBA matrix where row = modality, col = frame, color =
    # 3-tier verdict for that (modality, frame). The single most informative
    # view on this metric: one glance shows "instruction row is solid red".
    modality_axis = "input.modality_dropout"
    if modality_axis in per_axis_results:
        heatmap, modality_order = _modality_verdict_heatmap(
            per_axis_results[modality_axis],
        )
        if heatmap is not None:
            rr.log(
                "modality/heatmap",
                rr.Image(heatmap, draw_order=_DRAW_BOX),
                recording=rec, static=True,
            )
            # Episode-summary bar: mean Δ_out per modality across all frames.
            means = _modality_episode_means(
                per_axis_results[modality_axis], modality_order,
            )
            # Rerun's BarChart wants ≥2 values; for a single modality
            # the per-modality text card alone is the right view.
            if means is not None and len(means) >= 2:
                rr.log(
                    "modality/episode_summary",
                    rr.BarChart(np.asarray(means, dtype=np.float32)),
                    recording=rec, static=True,
                )
            # Always log the values as a text doc so the single-modality
            # case still shows the number; the bar chart is just a
            # multi-modality visual.
            if means is not None:
                rr.log(
                    "modality/episode_summary_labels",
                    rr.TextDocument(
                        "**Mean Δaction per modality (whole episode):**\n\n"
                        + "\n".join(
                            f"- `{m}` — {v:.3f}"
                            for m, v in zip(modality_order, means)
                        ),
                        media_type=rr.MediaType.MARKDOWN,
                    ),
                    recording=rec, static=True,
                )

    # ── 5. Sensitivity — per-camera concentration bar ─────────────────────
    sensitivity_axis = "vision.scene_sensitivity"
    if sensitivity_axis in per_axis_results:
        cams_order, conc = _sensitivity_concentration_per_camera(
            per_axis_results[sensitivity_axis],
        )
        if conc and len(conc) >= 2:
            rr.log(
                "sensitivity/concentration",
                rr.BarChart(np.asarray(conc, dtype=np.float32)),
                recording=rec, static=True,
            )
        # The text-doc card below always logs so the single-camera case
        # still tells the user the number; the bar chart is just a
        # multi-camera visual.
        if conc:
            rr.log(
                "sensitivity/concentration_labels",
                rr.TextDocument(
                    "**Concentration per camera "
                    "(0 = totally diffuse, 1 = single-cell focus):**\n\n"
                    + "\n".join(f"- `{c}` — {v:.2f}" for c, v in zip(cams_order, conc)),
                    media_type=rr.MediaType.MARKDOWN,
                ),
                recording=rec, static=True,
            )

    # ── 6. Trajectory-level axes (attention drift + chunk consistency) ────
    # These don't have per-frame entries in per_axis_results; they live in
    # trajectory_axis_results as a one-shot summary per episode.
    chunk_axis_present = "internal.chunk_consistency" in trajectory_axis_results
    attention_drift_axis_present = "internal.attention_drift" in trajectory_axis_results
    if chunk_axis_present:
        info = trajectory_axis_results["internal.chunk_consistency"]
        rr.log(
            "chunk/summary",
            rr.TextDocument(
                _trajectory_axis_markdown(
                    "Chunk consistency", info,
                    extra=(
                        f"\n\n*Mean disagreement (normalized): "
                        f"{info.get('scalar_score', float('nan')):.3f}* "
                        f"(raw L2: "
                        f"{info.get('raw_mean_delta', float('nan')):.3f})"
                    ),
                ),
                media_type=rr.MediaType.MARKDOWN,
            ),
            recording=rec, static=True,
        )
    if attention_drift_axis_present:
        info = trajectory_axis_results["internal.attention_drift"]
        rr.log(
            "attention/summary",
            rr.TextDocument(
                _trajectory_axis_markdown(
                    "Attention drift", info,
                    extra=(
                        f"\n\n*Mean centroid drift: "
                        f"{info.get('scalar_score', float('nan')):.1f} px*"
                    ),
                ),
                media_type=rr.MediaType.MARKDOWN,
            ),
            recording=rec, static=True,
        )

    # Per-frame-pair series for the trajectory-level axes. These axes carry
    # one episode-level verdict, but the underlying signal (chunk-step
    # disagreement, attention-centroid drift) is naturally per-frame-pair.
    # Logging it keyed to the originating frame populates each tab's
    # time-series view — without it those views referenced an entity path
    # that was never written.
    for axis, info in trajectory_axis_results.items():
        for frame_idx, value in info.get("per_frame_series", []):
            _set_time("frame_index", sequence=int(frame_idx))
            rr.log(f"plots/diagnostics/{axis}", Scalars(float(value)), recording=rec)

    # ── 7. Findings (one big plain-English page across all axes) ──────────
    findings_md = _build_findings_markdown(
        trajectory, per_axis_results, trajectory_axis_results,
    )
    if dead_cams:
        findings_md += (
            "\n\n---\n*Cameras with no live feed (blank/placeholder "
            f"streams, skipped): {sorted(dead_cams)}*\n"
        )
    rr.log("findings", rr.TextDocument(findings_md, media_type=rr.MediaType.MARKDOWN),
           recording=rec, static=True)

    # ── 8. Blueprint ──────────────────────────────────────────────────────
    blueprint = _build_blueprint(
        rrb,
        cams_all=cams_all,
        per_axis_results=per_axis_results,
        trajectory_axis_results=trajectory_axis_results,
        cams_with_attention=cams_with_attention,
        cams_with_sensitivity=cams_with_sensitivity,
        cams_with_mask=cams_with_mask,
        cams_with_masked_image=sorted(cams_with_masked_image),
    )
    # send_blueprint(make_active=True) overrides any cached viewer layout
    # tied to the application_id. Without this, a schema change leaves
    # the viewer pointing at entity paths that no longer exist → empty
    # views.
    rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=rec)
    rr.save(str(out_path), recording=rec)
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# Heatmap / mask helpers (continuous-magnitude → RGBA upsampled to image)
# ──────────────────────────────────────────────────────────────────────────

def _heatmap_rgba(
    heatmap: np.ndarray, target_hw: tuple[int, int],
    *, cmap_name: str = "turbo", gamma: float = 0.7,
) -> Optional[np.ndarray]:
    """Normalize → colormap → RGBA, alpha encodes magnitude. Upsampled to
    the camera resolution. Returns None for a flat / zero heatmap so we
    never log a misleading solid-colour overlay for a camera with no
    measurable response."""
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim == 3 and h.shape[-1] == 1:
        h = h[..., 0]
    if h.ndim != 2:
        return None
    lo, hi = float(np.nanmin(h)), float(np.nanmax(h))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9:
        return None
    norm = (h - lo) / (hi - lo)

    from PIL import Image as PILImage
    H, W = target_hw
    norm_u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    norm = (
        np.asarray(PILImage.fromarray(norm_u8).resize((W, H), PILImage.BILINEAR))
        .astype(np.float32) / 255.0
    )

    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(norm, 0, 1) ** gamma * 255).astype(np.uint8)
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


def _is_dead_feed(arr: np.ndarray) -> bool:
    """True if a camera frame is all-black / near-constant — some dataset
    episodes ship inactive secondary cameras as blank frames."""
    a = np.asarray(arr, dtype=np.float32)
    return float(a.std()) < 1.0


# ──────────────────────────────────────────────────────────────────────────
# Verdict ribbon / categorical heatmap builders
# ──────────────────────────────────────────────────────────────────────────

# Visual height of a verdict ribbon (1 row of frames painted at this many
# image pixels so it's visible as a strip, not a 1-pixel line).
_RIBBON_PX_PER_ROW = 24
_RIBBON_PX_PER_COL = 16   # each frame is this many image pixels wide
_HEATMAP_PX_PER_CELL = 24


def _verdict_ribbon(traj_result: TrajectoryDiagnosticResult) -> np.ndarray:
    """Per-axis horizontal ribbon, one column per frame, color = verdict.

    Returned as an upsampled RGB image so it reads as a clean strip in
    Rerun's Spatial2DView. Sits at the top of each metric tab so the
    user sees "where the verdict changes" at a glance.
    """
    n = len(traj_result.per_frame)
    row = np.zeros((1, max(1, n), 3), dtype=np.uint8)
    for i, r in enumerate(traj_result.per_frame):
        _, rgb = _verdict(r.severity)
        row[0, i] = rgb
    # Upsample so the strip is visible (NEAREST so colors stay crisp).
    from PIL import Image as PILImage
    H = _RIBBON_PX_PER_ROW
    W = max(_RIBBON_PX_PER_COL, n * _RIBBON_PX_PER_COL)
    out = np.asarray(
        PILImage.fromarray(row).resize((W, H), PILImage.NEAREST)
    )
    return out


def _modality_verdict_heatmap(
    traj_result: TrajectoryDiagnosticResult,
) -> tuple[Optional[np.ndarray], list[str]]:
    """M × N RGBA matrix of per-(modality, frame) verdicts.

    Categorical encoding — not a continuous magnitude colormap. The
    palette is the 3-tier verdict palette + gray for "couldn't test."
    Returns ``(heatmap, modality_order)`` or ``(None, [])`` if the
    diagnostic produced no per-modality data.
    """
    # Collect every modality that appears in any frame.
    modality_order: list[str] = []
    seen: set[str] = set()
    for r in traj_result.per_frame:
        per = (r.raw or {}).get("per_modality") or {}
        for m in per:
            if m not in seen:
                seen.add(m)
                modality_order.append(m)
    if not modality_order:
        return None, []
    M = len(modality_order)
    N = len(traj_result.per_frame)
    row = np.full((M, N, 3), VERDICT_NOTEST_RGB, dtype=np.uint8)
    for j, r in enumerate(traj_result.per_frame):
        per = (r.raw or {}).get("per_modality") or {}
        for i, m in enumerate(modality_order):
            sub = per.get(m)
            if not isinstance(sub, dict):
                continue
            verdict = sub.get("verdict")
            row[i, j] = _modality_cell_rgb(verdict)
    from PIL import Image as PILImage
    H = M * _HEATMAP_PX_PER_CELL
    W = N * _HEATMAP_PX_PER_CELL
    out = np.asarray(PILImage.fromarray(row).resize((W, H), PILImage.NEAREST))
    return out, modality_order


def _modality_cell_rgb(verdict: Optional[str]) -> tuple[int, int, int]:
    """ModalityDropout per-(modality, frame) verdict → ribbon color."""
    if verdict == "USED":        return VERDICT_OK_RGB
    if verdict == "PARTIAL":     return VERDICT_MIXED_RGB
    if verdict == "IGNORED":     return VERDICT_LOOK_RGB
    if verdict == "BELOW_NOISE": return VERDICT_NOTEST_RGB
    if verdict == "UNTESTABLE":  return VERDICT_NOTEST_RGB
    return VERDICT_NOTEST_RGB


def _modality_episode_means(
    traj_result: TrajectoryDiagnosticResult, modality_order: list[str],
) -> Optional[list[float]]:
    """Mean per-modality response magnitude across the episode.

    Returns one value per modality in the same order as
    ``modality_order``; missing/NaN entries collapse to 0 so the bar
    chart never has gaps.
    """
    if not modality_order:
        return None
    sums: dict[str, list[float]] = {m: [] for m in modality_order}
    for r in traj_result.per_frame:
        per = (r.raw or {}).get("per_modality") or {}
        for m in modality_order:
            sub = per.get(m)
            if not isinstance(sub, dict):
                continue
            val = sub.get("mean_response_normalized")
            if val is None or val != val:
                continue
            sums[m].append(float(val))
    return [float(np.mean(v)) if v else 0.0 for v in sums.values()]


def _sensitivity_concentration_per_camera(
    traj_result: TrajectoryDiagnosticResult,
) -> tuple[list[str], list[float]]:
    """Mean per-camera concentration across the episode for the
    sensitivity tab's summary bar chart.

    Higher = more focused; lower = more diffuse.
    """
    sums: dict[str, list[float]] = {}
    for r in traj_result.per_frame:
        per_cam = (r.raw or {}).get("top_k_concentration_per_camera") or {}
        for cam, v in per_cam.items():
            if v is None or v != v:
                continue
            sums.setdefault(cam, []).append(float(v))
    order = sorted(sums)
    return order, [float(np.mean(sums[c])) if sums[c] else 0.0 for c in order]


# ──────────────────────────────────────────────────────────────────────────
# Markdown builders (per-frame + trajectory-level + final findings page)
# ──────────────────────────────────────────────────────────────────────────

def _current_frame_markdown(axis: str, r: DiagnosticResult, f) -> str:
    """Per-frame current-cursor card. Big verdict label + observed +
    numbers. NO Severity word; uses the 3-tier label."""
    label, rgb = _verdict(r.severity)
    hex_color = "#%02x%02x%02x" % rgb
    raw = f.raw_numbers or {}
    keys = [k for k in raw if not k.startswith("_")][:8]   # top 8 numbers
    num_lines = "\n".join(
        f"- `{k}` = {raw[k]:.4f}" if isinstance(raw[k], (int, float))
        else f"- `{k}` = {raw[k]}"
        for k in keys
    )
    return (
        f"### {axis}\n\n"
        f"<span style='color:{hex_color};font-weight:bold'>{label}</span>\n\n"
        f"**Observed:** {f.observed}\n\n"
        + (f"**Numbers**\n{num_lines}\n" if num_lines else "")
    )


def _trajectory_axis_markdown(
    title: str, info: dict, *, extra: str = "",
) -> str:
    sev = info.get("severity", "unknown")
    sev_enum = _severity_from_string(sev)
    label, rgb = _verdict(sev_enum)
    hex_color = "#%02x%02x%02x" % rgb
    expl = info.get("explanation", "")
    return (
        f"### {title}\n\n"
        f"<span style='color:{hex_color};font-weight:bold'>{label}</span>"
        f"{extra}\n\n"
        f"{expl}\n"
    )


def _severity_from_string(s: str) -> Severity:
    try:
        return Severity(s)
    except (ValueError, KeyError):
        return Severity.UNKNOWN


def _build_findings_markdown(
    trajectory: Trajectory,
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    trajectory_axis_results: dict[str, dict],
) -> str:
    """Plain-English findings page for the Findings tab.

    Mixes per-axis (per-frame Finding rolled up) AND trajectory-level
    axes (attention drift, chunk consistency). Worst-first ordering so
    things that warrant a look surface at the top of the page.
    """
    lines = ["# Emboviz findings", ""]
    instr = getattr(trajectory, "instruction", None) or (
        trajectory.frames[0].instruction if trajectory.frames else None
    )
    if instr:
        lines += [f"**Instruction:** {instr}", ""]

    # Sort by worst severity first so the user sees the headlines.
    def _per_axis_severity(traj_result: TrajectoryDiagnosticResult) -> Severity:
        # Pick the highest-priority severity seen on any frame.
        rank = max(
            (r.severity for r in traj_result.per_frame),
            default=Severity.UNKNOWN,
            key=lambda s: s.sort_key,
        )
        return rank

    ordered_axes = sorted(
        per_axis_results.items(),
        key=lambda kv: -_per_axis_severity(kv[1]).sort_key,
    )
    for axis, traj_result in ordered_axes:
        finding = None
        for r in traj_result.per_frame:
            if r.finding is not None:
                finding = r.finding
                break
        worst_sev = _per_axis_severity(traj_result)
        label, _ = _verdict(worst_sev)
        lines.append(f"## {axis} — {label}")
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
        sev_enum = _severity_from_string(info.get("severity", "unknown"))
        label, _ = _verdict(sev_enum)
        lines.append(f"## {axis} — {label}")
        lines += [info.get("explanation", "") or "(no explanation)", ""]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Blueprint — one tab per applicable metric, identical skeleton per tab
# ──────────────────────────────────────────────────────────────────────────

def _build_blueprint(
    rrb,
    *,
    cams_all: list[str],
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    trajectory_axis_results: dict[str, dict],
    cams_with_attention: set[str],
    cams_with_sensitivity: set[str],
    cams_with_mask: set[str],
    cams_with_masked_image: list[str],
):
    """Construct the tabbed layout.

    Only tabs whose underlying diagnostic produced data appear. We never
    show an empty placeholder — if a metric structurally doesn't apply
    (e.g. chunk consistency on a single-step policy), its tab is absent
    from the dashboard entirely. The README + Findings tab still list
    it as "did not apply" so the user knows what wasn't measured.
    """
    tabs = []

    # ── Memorization tab ─────────────────────────────────────────────────
    if "vision.memorization" in per_axis_results:
        tabs.append(_memorization_tab(rrb, cams_all, cams_with_mask, cams_with_masked_image))

    # ── Modality tab ─────────────────────────────────────────────────────
    if "input.modality_dropout" in per_axis_results:
        tabs.append(_modality_tab(rrb))

    # ── Attention tab ────────────────────────────────────────────────────
    if cams_with_attention or "internal.attention_drift" in trajectory_axis_results:
        tabs.append(_attention_tab(rrb, cams_all, cams_with_attention,
                                   has_drift="internal.attention_drift" in trajectory_axis_results))

    # ── Sensitivity tab ──────────────────────────────────────────────────
    if "vision.scene_sensitivity" in per_axis_results:
        tabs.append(_sensitivity_tab(rrb, cams_all, cams_with_sensitivity))

    # ── Chunk tab ────────────────────────────────────────────────────────
    if "internal.chunk_consistency" in trajectory_axis_results:
        tabs.append(_chunk_tab(rrb))

    # ── Findings tab ─────────────────────────────────────────────────────
    tabs.append(_findings_tab(rrb))

    return rrb.Blueprint(
        rrb.Tabs(*tabs),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),     # show the timeline scrubber
        collapse_panels=True,
        auto_views=False,
    )


def _memorization_tab(
    rrb, cams_all: list[str], cams_with_mask: set[str],
    cams_with_masked_image: list[str],
):
    """Description → verdict ribbon → per-camera (original | masked)
    grid → Δaction time-series → current-frame card."""
    description = rrb.TextDocumentView(
        origin="docs/memorization", name="What this measures",
    )
    ribbon = rrb.Spatial2DView(
        origin="ribbon/vision.memorization",
        name="Per-frame verdict (green = OK, yellow = MIXED, red = WORTH A LOOK)",
    )

    # Per-camera triptychs. For each camera: original, masked
    # (per fill mode), bbox/mask overlay. We use Spatial2DView per camera
    # and stack them into a Grid. Always include every live camera —
    # cameras with no detection just show the raw frame.
    cam_views = []
    for cam in cams_all:
        cam_views.append(rrb.Spatial2DView(
            origin=f"memorization/{cam}",
            contents=[
                "+ $origin/original",
                "+ $origin/mask",
                "+ $origin/boxes",
            ],
            name=f"{cam} — original + mask overlay",
        ))
        # If we have masked-fill images for this camera, show them too.
        if cam in cams_with_masked_image:
            cam_views.append(rrb.Spatial2DView(
                origin=f"memorization/{cam}",
                contents=[
                    "+ $origin/masked_channel_mean",
                    "+ $origin/masked_gaussian_blur",
                    # on-manifold fill; only present when the run enabled
                    # the lama_inpaint fill (ignored by rerun if absent).
                    "+ $origin/masked_lama_inpaint",
                ],
                name=f"{cam} — what the model saw (masked)",
            ))
    cameras_block = rrb.Grid(*cam_views) if cam_views else rrb.TextDocumentView(
        origin="docs/memorization", name="(no live cameras)",
    )

    delta_plot = rrb.TimeSeriesView(
        origin="plots/diagnostics/vision.memorization",
        name="Action change when target is masked (higher = more grounded)",
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/vision.memorization",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        ribbon,
        cameras_block,
        delta_plot,
        current_frame,
        name="Memorization",
        row_shares=[2, 1, 6, 3, 3],
    )


def _modality_tab(rrb):
    """Description → verdict heatmap (M×N RGBA) → episode summary bar →
    per-modality time-series → current-frame card."""
    description = rrb.TextDocumentView(
        origin="docs/modality", name="What this measures",
    )
    heatmap = rrb.Spatial2DView(
        origin="modality/heatmap",
        name="Per-frame verdict (rows = input, cols = frame; "
             "green = OK, yellow = MIXED, red = WORTH A LOOK, gray = couldn't test)",
    )
    episode_summary = rrb.BarChartView(
        origin="modality/episode_summary",
        name="Mean Δaction per modality (whole episode)",
    )
    episode_labels = rrb.TextDocumentView(
        origin="modality/episode_summary_labels",
        name="Mean Δaction — values",
    )
    per_modality_series = rrb.TimeSeriesView(
        origin="plots/modality",
        name="Per-modality response over time",
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/input.modality_dropout",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        heatmap,
        rrb.Horizontal(episode_summary, episode_labels, column_shares=[3, 2]),
        per_modality_series,
        current_frame,
        name="Modality dropout",
        row_shares=[2, 5, 3, 3, 3],
    )


def _attention_tab(
    rrb, cams_all: list[str], cams_with_attention: set[str], *, has_drift: bool,
):
    """Description → per-camera attention overlay grid → drift plot →
    drift summary card. All live cameras are shown; cameras without an
    attention overlay simply show the bare RGB so the user has the
    visual context to compare."""
    description = rrb.TextDocumentView(
        origin="docs/attention", name="What this measures",
    )
    cam_views = [
        rrb.Spatial2DView(
            origin=f"world/camera/{cam}",
            contents=["+ $origin/rgb", "+ $origin/attention"],
            name=f"{cam}{' (no attention)' if cam not in cams_with_attention else ''}",
        )
        for cam in cams_all
    ]
    cameras_block = rrb.Grid(*cam_views) if cam_views else rrb.TextDocumentView(
        origin="docs/attention", name="(no live cameras)",
    )

    bottom = []
    drift_plot = rrb.TimeSeriesView(
        origin="plots/diagnostics/internal.attention_drift",
        name="Attention centroid drift over time (px)",
    )
    bottom.append(drift_plot)
    if has_drift:
        bottom.append(rrb.TextDocumentView(
            origin="attention/summary",
            name="Episode summary — drift",
        ))

    return rrb.Vertical(
        description,
        cameras_block,
        *bottom,
        name="Attention",
        row_shares=[2, 7, 3] + ([3] if has_drift else []),
    )


def _sensitivity_tab(
    rrb, cams_all: list[str], cams_with_sensitivity: set[str],
):
    """Description → per-camera sensitivity overlay grid → per-camera
    concentration bar → current-frame card."""
    description = rrb.TextDocumentView(
        origin="docs/sensitivity", name="What this measures",
    )
    cam_views = [
        rrb.Spatial2DView(
            origin=f"world/camera/{cam}",
            contents=["+ $origin/rgb", "+ $origin/sensitivity"],
            name=f"{cam}{' (no sensitivity signal)' if cam not in cams_with_sensitivity else ''}",
        )
        for cam in cams_all
    ]
    cameras_block = rrb.Grid(*cam_views) if cam_views else rrb.TextDocumentView(
        origin="docs/sensitivity", name="(no live cameras)",
    )

    concentration_bar = rrb.BarChartView(
        origin="sensitivity/concentration",
        name="Per-camera focus (0 = diffuse, 1 = sharp)",
    )
    concentration_labels = rrb.TextDocumentView(
        origin="sensitivity/concentration_labels",
        name="Per-camera focus — values",
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/vision.scene_sensitivity",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        cameras_block,
        rrb.Horizontal(concentration_bar, concentration_labels, column_shares=[3, 2]),
        current_frame,
        name="Sensitivity",
        row_shares=[2, 7, 3, 3],
    )


def _chunk_tab(rrb):
    """Description → per-frame disagreement plot → summary card."""
    description = rrb.TextDocumentView(
        origin="docs/chunk", name="What this measures",
    )
    disagreement_plot = rrb.TimeSeriesView(
        origin="plots/diagnostics/internal.chunk_consistency",
        name="Chunk-step disagreement over time (lower = more consistent)",
    )
    summary = rrb.TextDocumentView(
        origin="chunk/summary",
        name="Episode summary — chunk consistency",
    )
    return rrb.Vertical(
        description,
        disagreement_plot,
        summary,
        name="Chunk",
        row_shares=[2, 6, 3],
    )


def _findings_tab(rrb):
    description = rrb.TextDocumentView(
        origin="docs/findings", name="How to read this page",
    )
    all_findings = rrb.TextDocumentView(
        origin="findings", name="All findings in plain English",
    )
    return rrb.Vertical(
        description, all_findings,
        name="Findings",
        row_shares=[1, 8],
    )
