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


# Emoji per verdict. Rerun's Markdown renderer does NOT support inline HTML —
# a ``<span style='color:...'>`` shows up as literal text — so verdict colour
# in the text cards is carried by an emoji instead. Matches the green/yellow/
# red/gray legend in each tab's "What this measures" description.
_VERDICT_EMOJI: dict[Severity, str] = {
    Severity.PASS:     "🟢",
    Severity.INFO:     "🟢",
    Severity.MODERATE: "🟡",
    Severity.CRITICAL: "🔴",
    Severity.UNKNOWN:  "⚪",
}


def _verdict_emoji(sev: Severity) -> str:
    return _VERDICT_EMOJI.get(sev, "⚪")


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

We locate the manipulated target on every camera, mask it (channel-mean +
Gaussian-blur fills; LaMa inpaint when enabled), and measure how much the
model's action changes. A frame is scored only when the target is removed
from EVERY camera the model sees: if it can't be located on one camera it
stays visible there, so that frame is marked "couldn't test" rather than
scored on a partial mask.

**How to read this tab**

- **Original | Masked** per camera — verify the mask actually covered the
  target. If the detector caught the wrong thing the verdict is meaningless,
  which is why these come first.
- **Δaction plot** (seekable — scrub it and the images + card update):
  - The **line** is how much the action moved when the target was masked,
    normalized to "fraction of a typical action".
  - The **green band** is the "grounded" threshold, the **red band** the
    "memorized" one. Where the line sits is the reading: above green the
    model is reading the target; below red it barely reacted — the memorized
    signature (predicting from state + history, not vision).
  - Each frame's **point is coloured by its verdict** — 🟢 OK (grounded),
    🟡 MIXED, 🔴 WORTH A LOOK (memorized), ⚪ couldn't test.
- **This frame** card — verdict + raw numbers for the frame under the cursor.

Numbers are normalized by the model's typical action magnitude (per-trajectory
calibration), so the thresholds mean the same thing across models.
"""

_DOC_MODALITY = """## Modality dropout — which inputs is the model using?

For each declared input (every camera, the instruction, robot state, gripper,
action history) we substitute it with K real values sampled from OTHER
episodes in your dataset and measure how much the action changes. A big change
means the model relies on that input; barely any change means it's ignoring it
(or the dataset couldn't offer a different-enough value to test).

**How to read this tab**

- **Per-modality response plot** (seekable — scrub it and the per-frame card
  updates). One line per input, normalized to "fraction of a typical action":
  - the **green band** is the "used" threshold, the **red band** the "ignored"
    one.
  - a line riding **above green** = the model uses that input; pinned **below
    red** = it ignores it; in between = partial.
- **Per modality — mean Δaction + verdict** card: the whole-episode average
  for each input, tagged 🟢 USED / 🟡 PARTIAL / 🔴 IGNORED against the same
  scale, strongest first.
- **This frame** card: the verdict + numbers for the frame under the cursor.

A modality you EXPECT to matter (the instruction on a language task, the wrist
camera for fine manipulation) reading 🔴 IGNORED is the strongest signal here.
Caveat: on a single-task dataset the substitutes can be too similar to the
original to count as a real intervention — read a low `state` response with
that in mind.
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
  centroid moves between consecutive frames (seekable; scrub it against the
  overlays above). Low = stable focus; large jumps = wandering, which tends
  to spike in the frames before a failure.
  - the 🟡 **warning** and 🔴 **critical** bands are the thresholds: a line
    sitting well below the warning band means the gaze is anchored.
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

- **Per-frame disagreement** (seekable) is the normalized L2 distance
  between ``chunk[t][k]`` and ``chunk[t+k][0]``. Lower = more consistent
  lookahead.
  - the 🟢 **consistent** band (noise floor) and 🔴 **drifts** band
    (strong-disagreement threshold) mark the regions: a line below green is
    reliable lookahead; above red the multi-step plan drifts.
- **Episode summary** card gives the one-line verdict + the mean.

This metric is hidden from the dashboard when your model is
single-step (no ``action_chunk``).
"""

_DOC_FINDINGS = """## Findings — plain English for every metric

A single page summarizing every diagnostic, **most-concerning first**, each
tagged 🟢 OK / 🟡 MIXED / 🔴 WORTH A LOOK / ⚪ couldn't test. For each one:

- **Observed** — what the test actually measured (the per-frame distribution
  plus the most representative example).
- **Meaning** — what the result tells you about the model.
- **Next step** — a concrete action: what to try, what to check, where to
  drill.

Diagnostics that **could not run** are listed under "Not measured" with the
reason — we surface what wasn't tested rather than silently dropping it. If a
metric was inconclusive on every frame, it says so; we never invent a number.
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
    not_applicable: Optional[dict[str, str]] = None,
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
            "emboviz core — if it is missing your install is incomplete; "
            "reinstall from the repo root with: uv sync (rerun-sdk>=0.33,<0.34)."
        ) from e

    if not hasattr(rr, "RecordingStream"):
        raise RuntimeError(
            "rerun-sdk too old for the export API. Reinstall emboviz from the "
            "repo root with: uv sync (pins rerun-sdk>=0.33,<0.34)."
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
    not_applicable               = not_applicable               or {}

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
    fill_modes_seen: set[str] = set()

    def _frame_data(store: PerFrameByCamera, frame_idx: int, i: int) -> dict[str, np.ndarray]:
        if frame_idx in store:
            return store[frame_idx]
        if i in store:
            return store[i]
        return {}

    # ── 1. Per-frame camera streams + overlays ────────────────────────────
    # Conditional overlays (attention, sensitivity, mask, detection boxes,
    # masked images) are logged ONLY on frames that produce them. Rerun keeps
    # the latest logged value, so a frame WITHOUT an overlay would otherwise
    # display the previous frame's — e.g. a stale broccoli mask lingering on an
    # occluded frame. We record what we log each frame and explicitly Clear any
    # overlay that vanished, so absence reads as absence.
    prev_overlay_paths: set[str] = set()
    for i, scene in enumerate(trajectory.frames):
        frame_idx = trajectory.frame_indices[i]
        _set_time("frame_time", duration=i / fps)
        _set_time("frame_index", sequence=frame_idx)
        frame_overlay_paths: set[str] = set()

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
                frame_overlay_paths.add(f"world/camera/{cam}/attention")

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
                frame_overlay_paths.add(f"world/camera/{cam}/sensitivity")

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
                frame_overlay_paths.add(f"memorization/{cam}/mask")
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
                frame_overlay_paths.add(f"memorization/{cam}/boxes")
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
                fill_modes_seen.add(fill_mode)
                frame_overlay_paths.add(f"memorization/{cam}/masked_{fill_mode}")

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

        # Clear overlays shown on an earlier frame but absent here, so they do
        # not visually persist (Rerun keeps the latest value). Clearing on the
        # transition suffices — the cleared state then persists until the entity
        # is logged again.
        for stale_path in prev_overlay_paths - frame_overlay_paths:
            rr.log(stale_path, rr.Clear(recursive=False), recording=rec)
        prev_overlay_paths = frame_overlay_paths

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

    # ── 3. Memorization plot overlays — decision bands + verdict-coloured
    # points on the SEEKABLE Δaction time-series. The verdict now scrubs with
    # the time cursor instead of living in a frozen strip (the old static
    # ribbon, removed): the two flat bands are the diagnostic's own thresholds
    # — below "memorized" the action barely moved when the target was masked,
    # above "grounded" it moved substantially — and each frame's point is
    # coloured by its verdict, so where the line sits IS the reading.
    mem_axis = "vision.memorization"
    if mem_axis in per_axis_results:
        _log_memorization_overlays(
            rr, rec, Scalars, per_axis_results[mem_axis], fps, _set_time,
        )

    # ── 4. Modality dropout — decision bands on the seekable per-modality
    # plot + an episode-summary card that anchors each modality's mean response
    # to a verdict. (No static verdict heatmap, no unlabelled bar chart: the
    # per-modality lines ride the timeline, and where each sits relative to the
    # bands is the reading.)
    modality_axis = "input.modality_dropout"
    if modality_axis in per_axis_results:
        _log_modality_overlays(
            rr, rec, Scalars, per_axis_results[modality_axis], fps, _set_time,
        )

    # ── 5. Sensitivity — per-camera concentration values ──────────────────
    # The number already carries its own 0→1 scale, so a bare bar chart (no
    # axis labels) added nothing over the values card; only the card is kept.
    sensitivity_axis = "vision.scene_sensitivity"
    if sensitivity_axis in per_axis_results:
        cams_order, conc = _sensitivity_concentration_per_camera(
            per_axis_results[sensitivity_axis],
        )
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
    # Each point MUST be stamped on both the frame_index AND the frame_time
    # timeline: the plots display on frame_time, and setting only frame_index
    # left every point inheriting the last frame_time set elsewhere, collapsing
    # the whole series onto a single tick. We map the dataset frame index to
    # its sequential position to recover the frame_time coordinate.
    idx_to_pos = {int(fi): i for i, fi in enumerate(trajectory.frame_indices)}
    for axis, info in trajectory_axis_results.items():
        series = info.get("per_frame_series", [])
        if not series:
            continue
        for frame_idx, value in series:
            fi = int(frame_idx)
            _set_time("frame_index", sequence=fi)
            pos = idx_to_pos.get(fi)
            if pos is not None:
                _set_time("frame_time", duration=pos / fps)
            rr.log(f"plots/diagnostics/{axis}", Scalars(float(value)), recording=rec)
        _log_trajectory_axis_overlays(
            rr, rec, Scalars, axis, info, series, idx_to_pos, fps, _set_time,
        )

    # ── 7. Findings (one big plain-English page across all axes) ──────────
    findings_md = _build_findings_markdown(
        trajectory, per_axis_results, trajectory_axis_results,
        not_applicable=not_applicable,
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
        fill_modes=sorted(fill_modes_seen),
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

# The old verdict-ribbon and modality-heatmap raster constants are gone: every
# per-frame verdict now rides the seekable time-series, not a frozen image.


def _verdict_point_key(sev: Severity) -> str:
    """Per-frame severity → the point-series bucket it is drawn on in the
    memorization plot. Keys match the ``v_<key>`` child entity paths logged
    by :func:`_log_memorization_overlays`."""
    if sev in (Severity.PASS, Severity.INFO):
        return "ok"
    if sev == Severity.MODERATE:
        return "mixed"
    if sev == Severity.CRITICAL:
        return "look"
    return "untested"   # Severity.UNKNOWN / anything else


def _log_memorization_overlays(
    rr, rec, Scalars, traj_result: TrajectoryDiagnosticResult,
    fps: float, set_time,
) -> None:
    """Overlay the memorization decision bands + verdict-coloured points on the
    seekable Δaction series at ``plots/diagnostics/vision.memorization``.

    Everything is logged under that entity path (or children of it) so the
    tab's single TimeSeriesView shows the Δaction line, two flat threshold
    bands, and one coloured point per frame — all on the same timeline the
    images and the per-frame card scrub against. The bands are the diagnostic's
    OWN thresholds, read from its per-frame raw numbers (falling back to the
    documented defaults), never hard-coded blind.
    """
    base = "plots/diagnostics/vision.memorization"

    # Thresholds are constant across frames; take them from the first frame
    # that reported them (the diagnostic stamps them into finding.raw_numbers).
    memorized_thr, grounded_thr = 0.05, 0.30
    for r in traj_result.per_frame:
        rn = (r.finding.raw_numbers if r.finding is not None else None) or {}
        if "ignored_threshold" in rn:
            memorized_thr = float(rn["ignored_threshold"])
        if "grounded_threshold" in rn:
            grounded_thr = float(rn["grounded_threshold"])
        if rn:
            break

    # Static styling (logged once): the main Δaction line + the two bands.
    rr.log(base, rr.SeriesLines(
        colors=[120, 130, 140], names="Δaction (masked − baseline)", widths=1.5,
    ), recording=rec, static=True)
    rr.log(f"{base}/memorized_below", rr.SeriesLines(
        colors=list(VERDICT_LOOK_RGB),
        names=f"memorized below {memorized_thr:.2f}", widths=1.0,
    ), recording=rec, static=True)
    rr.log(f"{base}/grounded_above", rr.SeriesLines(
        colors=list(VERDICT_OK_RGB),
        names=f"grounded above {grounded_thr:.2f}", widths=1.0,
    ), recording=rec, static=True)

    # Verdict point series — one per category, static colour + circle marker.
    _VPOINTS = {
        "ok":       (VERDICT_OK_RGB,     "OK (grounded)"),
        "mixed":    (VERDICT_MIXED_RGB,  "MIXED"),
        "look":     (VERDICT_LOOK_RGB,   "WORTH A LOOK (memorized)"),
        "untested": (VERDICT_NOTEST_RGB, "couldn't test"),
    }
    for key, (rgb, name) in _VPOINTS.items():
        rr.log(f"{base}/v_{key}", rr.SeriesPoints(
            colors=list(rgb), names=name, markers="circle", marker_sizes=6.0,
        ), recording=rec, static=True)

    # Per-frame data: the two flat band values at every frame, and the frame's
    # Δaction drawn as a point on the series matching its verdict.
    for i, r in enumerate(traj_result.per_frame):
        set_time("frame_time", duration=i / fps)
        set_time("frame_index", sequence=traj_result.frame_indices[i])
        rr.log(f"{base}/memorized_below", Scalars(memorized_thr), recording=rec)
        rr.log(f"{base}/grounded_above", Scalars(grounded_thr), recording=rec)
        score = r.scalar_score
        if score == score:   # not NaN
            key = _verdict_point_key(r.severity)
            rr.log(f"{base}/v_{key}", Scalars(float(score)), recording=rec)


# Distinct line colours for the per-modality response plot, cycled in the
# order modalities first appear. Readable on Rerun's dark theme.
_MODALITY_LINE_RGB = [
    (77, 171, 247),   # blue
    (132, 94, 247),   # violet
    (32, 201, 151),   # teal
    (255, 146, 43),   # orange
    (240, 101, 149),  # pink
    (148, 216, 45),   # lime
]


def _modality_thresholds(
    traj_result: TrajectoryDiagnosticResult,
) -> tuple[float, float]:
    """``(ignored_below, used_above)`` — the modality diagnostic's own
    noise-floor and grounded thresholds, read from its per-frame raw output.
    Falls back to the documented defaults (0.05 / 0.30) if absent."""
    ignored_below, used_above = 0.05, 0.30
    for r in traj_result.per_frame:
        raw = r.raw or {}
        if "noise_floor_score" in raw:
            ignored_below = float(raw["noise_floor_score"])
        if "grounded_threshold" in raw:
            used_above = float(raw["grounded_threshold"])
        if raw:
            break
    return ignored_below, used_above


def _modality_order(traj_result: TrajectoryDiagnosticResult) -> list[str]:
    """Modalities in first-seen order across the episode."""
    order: list[str] = []
    seen: set[str] = set()
    for r in traj_result.per_frame:
        for m in ((r.raw or {}).get("per_modality") or {}):
            if m not in seen:
                seen.add(m)
                order.append(m)
    return order


def _log_modality_overlays(
    rr, rec, Scalars, traj_result: TrajectoryDiagnosticResult,
    fps: float, set_time,
) -> None:
    """Decision bands + named/coloured lines on the seekable per-modality plot
    at ``plots/modality``, plus an episode-summary card anchoring each
    modality's mean response to a USED / PARTIAL / IGNORED verdict.

    Replaces the static verdict heatmap (frozen, unseekable) and the
    unlabelled bar chart (redundant with the values card). The bands are the
    diagnostic's own thresholds, read from its raw output — never hard-coded.
    """
    ignored_below, used_above = _modality_thresholds(traj_result)
    order = _modality_order(traj_result)
    base = "plots/modality"

    # Name + colour each modality line (logged static; the per-frame loop logs
    # the data). The safe-path mirrors that loop's transform exactly.
    for idx, m in enumerate(order):
        safe = m.replace(":", "_").replace("/", "_")
        rgb = _MODALITY_LINE_RGB[idx % len(_MODALITY_LINE_RGB)]
        rr.log(f"{base}/{safe}", rr.SeriesLines(
            colors=list(rgb), names=m, widths=1.8,
        ), recording=rec, static=True)

    # Decision bands: above "used" the model responds to the input, below
    # "ignored" it does not. Where each modality line sits IS the verdict.
    rr.log(f"{base}/_ignored_below", rr.SeriesLines(
        colors=list(VERDICT_LOOK_RGB),
        names=f"ignored below {ignored_below:.2f}", widths=1.0,
    ), recording=rec, static=True)
    rr.log(f"{base}/_used_above", rr.SeriesLines(
        colors=list(VERDICT_OK_RGB),
        names=f"used above {used_above:.2f}", widths=1.0,
    ), recording=rec, static=True)
    for i in range(len(traj_result.per_frame)):
        set_time("frame_time", duration=i / fps)
        set_time("frame_index", sequence=traj_result.frame_indices[i])
        rr.log(f"{base}/_ignored_below", Scalars(ignored_below), recording=rec)
        rr.log(f"{base}/_used_above", Scalars(used_above), recording=rec)

    # Episode-summary card: mean Δaction per modality + its verdict + the scale.
    means = _modality_episode_means(traj_result, order)
    if means is None:
        return

    def _verdict_for(v: float) -> str:
        if v > used_above:
            return "🟢 USED"
        if v >= ignored_below:
            return "🟡 PARTIAL"
        return "🔴 IGNORED"

    lines = [
        "**Mean Δaction per modality (whole episode)**",
        "",
        f"_scale: 🟢 USED > {used_above:.2f}  ·  🟡 PARTIAL "
        f"{ignored_below:.2f}–{used_above:.2f}  ·  🔴 IGNORED < {ignored_below:.2f}_",
        "",
    ]
    # Strongest response first — the most-used input reads at the top.
    for m, v in sorted(zip(order, means), key=lambda kv: -kv[1]):
        lines.append(f"- `{m}` — {v:.3f}  {_verdict_for(v)}")
    rr.log("modality/episode_summary_labels", rr.TextDocument(
        "\n".join(lines), media_type=rr.MediaType.MARKDOWN,
    ), recording=rec, static=True)


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


# Main-series legend names for the trajectory-level axis plots.
_TRAJ_AXIS_SERIES_NAME = {
    "internal.attention_drift":   "centroid drift (px)",
    "internal.chunk_consistency": "chunk disagreement (normalized)",
}


def _log_trajectory_axis_overlays(
    rr, rec, Scalars, axis: str, info: dict, series: list,
    idx_to_pos: dict, fps: float, set_time,
) -> None:
    """Name the trajectory-axis line and draw any decision bands it declares,
    as flat reference lines on the same TimeSeriesView so the magnitude reads
    against them. Thresholds come from ``info`` (forwarded by the runner) —
    never hard-coded here. Band values span the same frames as the series, on
    both timelines, so they land on the frame_time axis the plot uses."""
    base = f"plots/diagnostics/{axis}"

    # Name the main line (its data was already logged by the caller).
    name = _TRAJ_AXIS_SERIES_NAME.get(axis)
    if name:
        rr.log(base, rr.SeriesLines(
            colors=[120, 130, 140], names=name, widths=1.8,
        ), recording=rec, static=True)

    # Decision bands per axis. Attention drift declares warn/critical px;
    # chunk consistency declares a noise floor (consistent below) and a
    # grounded threshold (strong disagreement above).
    bands: list[tuple[str, float, tuple[int, int, int], str]] = []
    if axis == "internal.attention_drift":
        warn = info.get("warn_px")
        crit = info.get("critical_px")
        if warn is not None:
            bands.append(("warn", float(warn), VERDICT_MIXED_RGB,
                          f"warning {float(warn):.0f}px"))
        if crit is not None:
            bands.append(("critical", float(crit), VERDICT_LOOK_RGB,
                          f"critical {float(crit):.0f}px"))
    elif axis == "internal.chunk_consistency":
        nf = info.get("noise_floor")
        gt = info.get("grounded_threshold")
        if nf is not None:
            bands.append(("consistent", float(nf), VERDICT_OK_RGB,
                          f"consistent below {float(nf):.2f}"))
        if gt is not None:
            bands.append(("drifts", float(gt), VERDICT_LOOK_RGB,
                          f"drifts above {float(gt):.2f}"))
    if not bands:
        return

    for key, _val, rgb, bname in bands:
        rr.log(f"{base}/band_{key}", rr.SeriesLines(
            colors=list(rgb), names=bname, widths=1.0,
        ), recording=rec, static=True)
    for frame_idx, _value in series:
        fi = int(frame_idx)
        set_time("frame_index", sequence=fi)
        pos = idx_to_pos.get(fi)
        if pos is not None:
            set_time("frame_time", duration=pos / fps)
        for key, val, _rgb, _bname in bands:
            rr.log(f"{base}/band_{key}", Scalars(val), recording=rec)


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

# Human-friendly titles for the dotted axis IDs (match the tab names). Used
# everywhere a verdict is shown to the user so no raw ``vision.memorization``
# style identifier leaks into the dashboard.
_AXIS_TITLE = {
    "vision.memorization":        "Memorization",
    "input.modality_dropout":     "Modality dropout",
    "vision.scene_sensitivity":   "Scene sensitivity",
    "internal.attention_drift":   "Attention drift",
    "internal.chunk_consistency": "Chunk consistency",
}


def _current_frame_markdown(axis: str, r: DiagnosticResult, f) -> str:
    """Per-frame current-cursor card. Big verdict label + observed +
    numbers. NO Severity word; uses the 3-tier label."""
    label, _ = _verdict(r.severity)
    emoji = _verdict_emoji(r.severity)
    raw = f.raw_numbers or {}
    keys = [k for k in raw if not k.startswith("_")][:8]   # top 8 numbers
    num_lines = "\n".join(
        f"- `{k}` = {raw[k]:.4f}" if isinstance(raw[k], (int, float))
        else f"- `{k}` = {raw[k]}"
        for k in keys
    )
    return (
        f"### {_AXIS_TITLE.get(axis, axis)}\n\n"
        f"**{emoji} {label}**\n\n"
        f"**Observed:** {f.observed}\n\n"
        + (f"**Numbers**\n{num_lines}\n" if num_lines else "")
    )


def _trajectory_axis_markdown(
    title: str, info: dict, *, extra: str = "",
) -> str:
    sev = info.get("severity", "unknown")
    sev_enum = _severity_from_string(sev)
    label, _ = _verdict(sev_enum)
    emoji = _verdict_emoji(sev_enum)
    expl = info.get("explanation", "")
    return (
        f"### {title}\n\n"
        f"**{emoji} {label}**{extra}\n\n"
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
    not_applicable: Optional[dict[str, str]] = None,
) -> str:
    """Plain-English findings page for the Findings tab.

    One entry per diagnostic — per-frame axes (rolled up) and trajectory-level
    axes (attention drift, chunk consistency) merged into a SINGLE list, most-
    concerning first, each with a human title + verdict emoji. Diagnostics that
    could not run are listed at the end so the user also sees what was NOT
    measured (and why), never a silent omission.
    """
    not_applicable = not_applicable or {}
    lines = ["# Emboviz findings", ""]
    instr = getattr(trajectory, "instruction", None) or (
        trajectory.frames[0].instruction if trajectory.frames else None
    )
    if instr:
        lines += [f"**Instruction:** {instr}", ""]

    # The verdict label + quoted text come from the trajectory aggregate
    # (trajectory_severity / trajectory_finding) — the SAME source report.md
    # uses — so label and text always agree. Per-frame and trajectory-level
    # axes are unified into one list and ordered most-concerning first, so the
    # ordering is genuinely worst-first across BOTH kinds.
    _CONCERN = {
        Severity.CRITICAL: 4, Severity.MODERATE: 3, Severity.INFO: 2,
        Severity.PASS: 1, Severity.UNKNOWN: 0,
    }
    entries: list[tuple[str, Severity, str, Optional[str], Optional[str]]] = []
    for axis, tr in per_axis_results.items():
        f = tr.trajectory_finding()
        entries.append((axis, tr.trajectory_severity(),
                        f.observed, f.meaning, f.next_step))
    for axis, info in trajectory_axis_results.items():
        sev = _severity_from_string(info.get("severity", "unknown"))
        entries.append((axis, sev,
                        info.get("explanation", "") or "(no explanation)",
                        None, None))
    entries.sort(key=lambda e: -_CONCERN.get(e[1], 0))

    # One-line verdict spread so the user gets the gist before reading.
    if entries:
        spread: dict[str, int] = {}
        for _axis, sev, *_rest in entries:
            lbl, _ = _verdict(sev)
            spread[lbl] = spread.get(lbl, 0) + 1
        summary = " · ".join(f"{n} {lbl}" for lbl, n in spread.items())
        lines += [f"**{len(entries)} diagnostics —** {summary}", ""]

    for axis, sev, observed, meaning, next_step in entries:
        title = _AXIS_TITLE.get(axis, axis)
        label, _ = _verdict(sev)
        emoji = _verdict_emoji(sev)
        lines.append(f"## {emoji} {title} — {label}")
        lines.append(f"- **Observed:** {observed}")
        if meaning:
            lines.append(f"- **Meaning:** {meaning}")
        if next_step:
            lines.append(f"- **Next step:** {next_step}")
        lines.append("")

    if not_applicable:
        lines += ["---", "", "### Not measured", ""]
        for axis, why in not_applicable.items():
            lines.append(f"- **{_AXIS_TITLE.get(axis, axis)}** — {why}")
        lines.append("")

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
    fill_modes: list[str],
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
        tabs.append(_memorization_tab(rrb, cams_all, cams_with_mask, cams_with_masked_image, fill_modes))

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
    cams_with_masked_image: list[str], fill_modes: list[str],
):
    """Description → per-camera (original | masked) grid → Δaction
    time-series (with decision bands + verdict-coloured points) →
    current-frame card. No static verdict strip — the verdict rides the
    seekable plot."""
    description = rrb.TextDocumentView(
        origin="docs/memorization", name="What this measures",
    )

    # Per-camera before/after. For each camera: the original frame with the
    # removal mask outlined, then the LaMa-inpaint result (target removed) —
    # the clean on-manifold removal the user reads the verdict against. The
    # OOD fills (channel_mean / gaussian_blur) still drive the verdict but
    # are not rendered. Always include every live camera — cameras with no
    # detection just show the raw frame.
    cam_views = []
    for cam in cams_all:
        cam_views.append(rrb.Spatial2DView(
            origin=f"memorization/{cam}",
            contents=[
                "+ $origin/original",
                "+ $origin/mask",
                "+ $origin/boxes",
            ],
            name=f"{cam} — original (mask outlined)",
        ))
        # The removal result. ``fill_modes`` here is the displayed fill set
        # (LaMa only when present — the runner logs just that one), so this
        # is a single "target removed" panel next to the original.
        if cam in cams_with_masked_image:
            for fm in fill_modes:
                label = (
                    "target removed (LaMa inpaint)"
                    if fm == "lama_inpaint" else f"masked: {fm}"
                )
                cam_views.append(rrb.Spatial2DView(
                    origin=f"memorization/{cam}",
                    contents=[f"+ $origin/masked_{fm}"],
                    name=f"{cam} — {label}",
                ))
    cameras_block = rrb.Grid(*cam_views) if cam_views else rrb.TextDocumentView(
        origin="docs/memorization", name="(no live cameras)",
    )

    delta_plot = rrb.TimeSeriesView(
        origin="plots/diagnostics/vision.memorization",
        name=(
            "Δaction when target masked — above green = grounded, "
            "below red = memorized; point colour = per-frame verdict"
        ),
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/vision.memorization",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        cameras_block,
        delta_plot,
        current_frame,
        name="Memorization",
        row_shares=[2, 6, 4, 3],
    )


def _modality_tab(rrb):
    """Description → seekable per-modality response plot (with used/ignored
    bands) → episode-summary card (mean Δaction + verdict per modality) →
    current-frame card. No static heatmap, no unlabelled bar chart."""
    description = rrb.TextDocumentView(
        origin="docs/modality", name="What this measures",
    )
    per_modality_series = rrb.TimeSeriesView(
        origin="plots/modality",
        name=(
            "Per-modality response over time — above green = used, "
            "below red = ignored (one line per input)"
        ),
    )
    episode_labels = rrb.TextDocumentView(
        origin="modality/episode_summary_labels",
        name="Per modality — mean Δaction + verdict",
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/input.modality_dropout",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        per_modality_series,
        episode_labels,
        current_frame,
        name="Modality dropout",
        row_shares=[2, 6, 3, 3],
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
        name=(
            "Attention centroid drift (px) — below yellow = anchored, "
            "above = wandering"
        ),
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
    concentration values → current-frame card. No bar chart — the values
    card carries the same number with its 0→1 scale."""
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

    concentration_labels = rrb.TextDocumentView(
        origin="sensitivity/concentration_labels",
        name="Per-camera focus — values (0 = diffuse, 1 = sharp)",
    )
    current_frame = rrb.TextDocumentView(
        origin="current_frame/vision.scene_sensitivity",
        name="This frame — verdict + numbers",
    )

    return rrb.Vertical(
        description,
        cameras_block,
        concentration_labels,
        current_frame,
        name="Sensitivity",
        row_shares=[2, 7, 2, 3],
    )


def _chunk_tab(rrb):
    """Description → per-frame disagreement plot → summary card."""
    description = rrb.TextDocumentView(
        origin="docs/chunk", name="What this measures",
    )
    disagreement_plot = rrb.TimeSeriesView(
        origin="plots/diagnostics/internal.chunk_consistency",
        name=(
            "Chunk-step disagreement over time — below green = consistent, "
            "above red = drifts (lower is better)"
        ),
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
