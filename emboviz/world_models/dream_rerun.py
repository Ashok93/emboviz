"""Rerun ``.rrd`` exporter for a closed-loop dream clip — the side-by-side.

One clip = one ``.rrd``. The viewer opens it and sees, on a single shared
timeline, the **original recorded episode** (leftmost column) next to one or more
**Cosmos dream** columns (the policy in the loop). The simplest run has one dream
column; a counterfactual run adds a second (e.g. *baseline* next to *spoon swap*)
so reality, the unperturbed dream, and the swapped dream all scrub together.

Time alignment is exact: the reactive loop commits one dreamed frame per real
timestep at the same fps, so dream frame ``i`` lines up with original frame ``i``
on the timeline — no resampling. Each row is one physical camera (the dream's
views are split out of the concat to match the dataset's single-camera frames),
so the comparison is apples-to-apples.

A dream column may be **not run** (``views=None``) — e.g. the masked swap when
SAM detected the target in no camera. That column renders as a constant black
panel carrying the reason, so the viewer always sees *why* nothing is there
rather than mistaking an empty column for a failed export.

Targets the pinned rerun-sdk >= 0.33, < 0.34 (see :mod:`emboviz.exporters.rerun`
for the rationale on the exact-minor pin and the on-disk format coupling).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DreamColumn:
    """One dream column in the side-by-side.

    ``views`` maps camera name -> per-frame ``(H, W, 3)`` uint8 arrays, aligned
    frame-for-frame with the original on the shared timeline. ``views=None`` means
    the column was **not run**; it renders as a black panel labelled with
    ``note``. ``note`` annotates the column in the context card either way (e.g.
    the per-camera swap status, or why the column is empty).
    """

    name: str
    views: Optional[dict[str, list[np.ndarray]]]
    note: Optional[str] = None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "col"


def _validate_rgb(arr: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.uint8)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{what} must be (H, W, 3) uint8 RGB, got shape {a.shape}.")
    return a


def _header_markdown(
    *, policy_name: str, instruction: Optional[str], columns: list[DreamColumn],
    cameras: list[str], fps: float, n_dream: int, seed_labels: list[str],
    context_note: Optional[str],
) -> str:
    lines = [
        f"# Cosmos dream — `{policy_name}` in the loop",
        "",
        f"**Task:** {instruction or '(none)'}",
        "",
        "**How to read:** scrub the timeline — every panel advances together.",
        "",
        "- **Leftmost** of each row — the original recorded episode, ground truth.",
    ]
    for col in columns:
        if col.views is None:
            lines.append(f"- **{col.name}** — not run ({col.note or 'no output'}); shown black.")
        else:
            extra = f" — {col.note}" if col.note else ""
            lines.append(
                f"- **{col.name}** — Cosmos simulating the policy, {fps:g} fps, "
                f"{n_dream} frames{extra}."
            )
    lines += [
        "",
        f"- One row per camera: {', '.join(f'`{c}`' for c in cameras)}.",
        "",
    ]
    if context_note:
        lines += [context_note, ""]
    lines += [
        f"The **conditioning seed(s)** below ({', '.join(seed_labels)}) are the full "
        "concats (wrist on top, the two exterior cameras tiled beneath) the world "
        "model was actually given.",
    ]
    return "\n".join(lines)


def export_dream_rerun(
    out_path: Path,
    *,
    original_views: dict[str, list[np.ndarray]],
    dream_columns: list[DreamColumn],
    seeds: list[tuple[str, np.ndarray]],
    instruction: Optional[str],
    fps: float,
    policy_name: str,
    context_note: Optional[str] = None,
    application_id: str = "emboviz-dream",
    recording_id: Optional[str] = None,
) -> Path:
    """Write one clip's multi-camera, multi-column side-by-side to ``out_path``.

    ``original_views`` (leftmost column) maps camera name -> per-frame
    ``(H, W, 3)`` uint8 arrays. ``dream_columns`` are the dream columns in display
    order; each column's ``views`` must cover the SAME camera names as
    ``original_views`` (or be ``None`` for a not-run column, rendered black).
    ``seeds`` are ``(label, concat)`` pairs shown as static reference panels (the
    seed(s) the world model conditioned on). Per camera the lists may differ in
    length (the recorded episode can run out before the dream does) and need not
    share a resolution. Raises on empty/malformed inputs — never writes a
    half-empty comparison.
    """
    if not original_views:
        raise ValueError("export_dream_rerun: original_views must be non-empty.")
    if not dream_columns:
        raise ValueError("export_dream_rerun: at least one dream column is required.")
    if not seeds:
        raise ValueError("export_dream_rerun: at least one seed panel is required.")
    cameras = list(original_views.keys())
    for cam in cameras:
        if not original_views[cam]:
            raise ValueError(f"export_dream_rerun: original_views[{cam!r}] is empty.")
    for col in dream_columns:
        if col.views is None:
            continue
        if set(col.views.keys()) != set(cameras):
            raise ValueError(
                f"export_dream_rerun: column {col.name!r} cameras "
                f"{sorted(col.views)} differ from original {sorted(cameras)}."
            )
        for cam in cameras:
            if not col.views[cam]:
                raise ValueError(
                    f"export_dream_rerun: column {col.name!r} views[{cam!r}] is empty."
                )

    # Unique, stable column keys for entity paths.
    keys: list[str] = []
    seen: set[str] = set()
    for col in dream_columns:
        base = _slug(col.name)
        key = base
        n = 1
        while key in seen:
            n += 1
            key = f"{base}_{n}"
        seen.add(key)
        keys.append(key)

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

    left = {c: [_validate_rgb(f, f"original {c} frame") for f in original_views[c]] for c in cameras}
    col_frames: list[Optional[dict[str, list[np.ndarray]]]] = []
    for col in dream_columns:
        if col.views is None:
            col_frames.append(None)
        else:
            col_frames.append(
                {c: [_validate_rgb(f, f"{col.name} {c} frame") for f in col.views[c]] for c in cameras}
            )
    seed_panels = [(label, _validate_rgb(arr, f"seed {label}")) for label, arr in seeds]

    # Timeline length: the longest of any real/dream column.
    n_dream = max(
        [len(left[c]) for c in cameras]
        + [len(cf[c]) for cf in col_frames if cf is not None for c in cameras]
    )
    # Black placeholder per camera (sized to the original) for not-run columns.
    black = {c: np.zeros_like(left[c][0]) for c in cameras}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=recording_id or out_path.stem,
    )
    rate = fps if fps > 0 else 10.0

    def _set_time(i: int) -> None:
        rr.set_time("frame_time", duration=i / rate, recording=rec)
        rr.set_time("frame_index", sequence=i, recording=rec)

    rr.log(
        "header",
        rr.TextDocument(
            _header_markdown(
                policy_name=policy_name, instruction=instruction,
                columns=dream_columns, cameras=cameras, fps=rate, n_dream=n_dream,
                seed_labels=[label for label, _ in seed_panels], context_note=context_note,
            ),
            media_type=rr.MediaType.MARKDOWN,
        ),
        recording=rec, static=True,
    )
    for label, arr in seed_panels:
        rr.log(f"seed/{_slug(label)}/rgb", rr.Image(arr), recording=rec, static=True)

    # Not-run columns log a single static black panel per camera (shown at any
    # cursor). Per-frame panels for the original + run columns advance on the
    # timeline; Rerun holds the last value, so a shorter stream freezes on its
    # final true frame rather than inventing pixels.
    for key, cf in zip(keys, col_frames):
        if cf is None:
            for cam in cameras:
                rr.log(f"compare/{cam}/{key}/rgb", rr.Image(black[cam]), recording=rec, static=True)
    for i in range(n_dream):
        _set_time(i)
        for cam in cameras:
            if i < len(left[cam]):
                rr.log(f"compare/{cam}/original/rgb", rr.Image(left[cam][i]), recording=rec)
        for key, cf in zip(keys, col_frames):
            if cf is None:
                continue
            for cam in cameras:
                if i < len(cf[cam]):
                    rr.log(f"compare/{cam}/{key}/rgb", rr.Image(cf[cam][i]), recording=rec)

    def _col_title(col: DreamColumn) -> str:
        suffix = " (not run)" if col.views is None else ""
        return f"{col.name}{suffix}"

    camera_rows = [
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"compare/{cam}/original", name=f"Original — {cam}"),
            *[
                rrb.Spatial2DView(
                    origin=f"compare/{cam}/{key}", name=f"{_col_title(col)} — {cam}",
                )
                for key, col in zip(keys, dream_columns)
            ],
        )
        for cam in cameras
    ]
    seed_row = rrb.Horizontal(
        *[
            rrb.Spatial2DView(origin=f"seed/{_slug(label)}", name=f"Seed — {label}")
            for label, _ in seed_panels
        ]
    )
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(origin="header", name="About this clip"),
            *camera_rows,
            seed_row,
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


__all__ = ["DreamColumn", "export_dream_rerun"]
