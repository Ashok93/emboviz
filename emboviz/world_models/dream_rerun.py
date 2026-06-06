"""Rerun ``.rrd`` exporter for a closed-loop dream clip — the side-by-side.

One clip = one ``.rrd``. The viewer opens it and sees, on a single shared
timeline, the **original recorded episode** (left) next to the **Cosmos dream
with the policy in the loop** (right), plus the conditioning seed the world model
started from. Scrubbing moves both panels together, so the difference between
reality and the simulated policy rollout is something you watch, not infer.

Time alignment is exact: the reactive loop commits one dreamed frame per real
timestep at the same fps, so dream frame ``i`` lines up with original frame
``i`` on the timeline — no resampling. Both panels show the same physical camera
(the dream's exterior view is split out of the concat to match the dataset's
single-camera frame), so the comparison is apples-to-apples.

This is the dream path's only visual output: it replaces the ad-hoc per-step
MP4s, and follows the same conventions as :mod:`emboviz.exporters.rerun` — the
unified ``set_time`` timeline, blueprint-driven layout, markdown context card.
Targets the pinned rerun-sdk >= 0.33, < 0.34 (see that module for the rationale
on the exact-minor pin and the on-disk format coupling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _validate_rgb(arr: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.uint8)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{what} must be (H, W, 3) uint8 RGB, got shape {a.shape}.")
    return a


def _header_markdown(
    *, policy_name: str, instruction: Optional[str], perturbation: Optional[str],
    cameras: list[str], fps: float, n_dream: int,
) -> str:
    lines = [
        f"# Cosmos dream — `{policy_name}` in the loop",
        "",
        f"**Task:** {instruction or '(none)'}",
        "",
    ]
    if perturbation:
        lines += [
            f"**Perturbation (counterfactual):** {perturbation}",
            "",
            "> The **right** column is a scene that never physically happened — Cosmos "
            f'edited the seed ("{perturbation}") and then simulated the policy acting '
            "in the result. The **left** column is the real recorded episode (unedited).",
            "",
        ]
    lines += [
        "**How to read:** scrub the timeline — every panel advances together.",
        "",
        f"- **Left** of each row — the original recorded episode, ground truth.",
        f"- **Right** of each row — Cosmos simulating the policy's actions, "
        f"{fps:g} fps, {n_dream} frames.",
        f"- One row per camera: {', '.join(f'`{c}`' for c in cameras)}.",
        "",
        "The **conditioning seed** below is the full concat (wrist on top, the two "
        "exterior cameras tiled beneath) the world model was actually given.",
    ]
    return "\n".join(lines)


def export_dream_rerun(
    out_path: Path,
    *,
    original_views: dict[str, list[np.ndarray]],
    dream_views: dict[str, list[np.ndarray]],
    seed_concat: np.ndarray,
    instruction: Optional[str],
    perturbation: Optional[str],
    fps: float,
    policy_name: str,
    application_id: str = "emboviz-dream",
    recording_id: Optional[str] = None,
) -> Path:
    """Write one clip's multi-camera side-by-side comparison to ``out_path``.

    ``original_views`` (left) and ``dream_views`` (right) map a camera name to its
    per-frame ``(H, W, 3)`` uint8 arrays, aligned frame-for-frame on a shared
    timeline. Both dicts must cover the same camera names and be non-empty; for
    each camera the two lists may differ in length (the recorded episode can run
    out before the dream does) and need not share a resolution. The viewer shows
    one row per camera — original on the left, dream on the right. ``seed_concat``
    is the full concat frame the world model conditioned on (already perturbed when
    ``perturbation`` is set), shown as a static reference. Raises on empty inputs or
    malformed frames — never writes a half-empty comparison.
    """
    if not dream_views or not original_views:
        raise ValueError("export_dream_rerun: original_views/dream_views must be non-empty.")
    cameras = list(dream_views.keys())
    if set(cameras) != set(original_views.keys()):
        raise ValueError(
            f"export_dream_rerun: camera sets differ — dream {sorted(dream_views)} "
            f"vs original {sorted(original_views)}."
        )
    for cam in cameras:
        if not dream_views[cam]:
            raise ValueError(f"export_dream_rerun: dream_views[{cam!r}] is empty.")
        if not original_views[cam]:
            raise ValueError(f"export_dream_rerun: original_views[{cam!r}] is empty.")

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as e:
        raise ImportError(
            "Rerun export requires the `rerun-sdk` package. It ships with emboviz "
            "core — if it is missing your install is incomplete; reinstall from the "
            "repo root with: uv sync (rerun-sdk>=0.33,<0.34)."
        ) from e
    if not hasattr(rr, "RecordingStream"):
        raise RuntimeError(
            "rerun-sdk too old for the export API. Reinstall emboviz from the repo "
            "root with: uv sync (pins rerun-sdk>=0.33,<0.34)."
        )

    seed = _validate_rgb(seed_concat, "seed_concat")
    left = {c: [_validate_rgb(f, f"original {c} frame") for f in original_views[c]] for c in cameras}
    right = {c: [_validate_rgb(f, f"dream {c} frame") for f in dream_views[c]] for c in cameras}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=recording_id or out_path.stem,
    )
    rate = fps if fps > 0 else 10.0
    n_dream = max(len(right[c]) for c in cameras)

    def _set_time(i: int) -> None:
        rr.set_time("frame_time", duration=i / rate, recording=rec)
        rr.set_time("frame_index", sequence=i, recording=rec)

    # Context card + conditioning seed: static, so they show at any cursor.
    rr.log(
        "header",
        rr.TextDocument(
            _header_markdown(
                policy_name=policy_name, instruction=instruction,
                perturbation=perturbation, cameras=cameras, fps=rate, n_dream=n_dream,
            ),
            media_type=rr.MediaType.MARKDOWN,
        ),
        recording=rec, static=True,
    )
    rr.log("seed/rgb", rr.Image(seed), recording=rec, static=True)

    # Per-frame panels on the shared timeline. The dream sets the length; the
    # original is logged only while real frames remain (Rerun holds the last
    # value, so a shorter real episode freezes on its final true frame rather
    # than inventing pixels).
    for i in range(n_dream):
        _set_time(i)
        for cam in cameras:
            if i < len(right[cam]):
                rr.log(f"compare/{cam}/dream/rgb", rr.Image(right[cam][i]), recording=rec)
            if i < len(left[cam]):
                rr.log(f"compare/{cam}/original/rgb", rr.Image(left[cam][i]), recording=rec)

    counterfactual = " (counterfactual)" if perturbation else ""
    camera_rows = [
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"compare/{cam}/original", name=f"Original — {cam}"),
            rrb.Spatial2DView(
                origin=f"compare/{cam}/dream",
                name=f"Dream ({policy_name}) — {cam}{counterfactual}",
            ),
        )
        for cam in cameras
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(origin="header", name="About this clip"),
            *camera_rows,
            rrb.Spatial2DView(origin="seed", name="Conditioning seed (concat)"),
            row_shares=[2] + [6] * len(camera_rows) + [3],
        ),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True,
        auto_views=False,
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=rec)
    rr.save(str(out_path), recording=rec)
    return out_path


__all__ = ["export_dream_rerun"]
