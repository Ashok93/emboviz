"""Rendering for a world-model trust run.

Two outputs: the trust curve (divergence vs horizon) and — the one that actually
shows the story — the predicted-vs-reality frames side by side, so the drift the
curve summarizes is something you can *see*. Both are written from a
:func:`emboviz.world_models.rollout.analyze_trust` result.

Pure Pillow + matplotlib (both in core); no torch, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emboviz_wire.types import Trajectory


def save_trust_curve(report: dict, path: Path) -> None:
    """Render the trust curve (divergence vs horizon) with the noise-floor band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(report["horizons"], report["divergence"], marker="o", label="prediction vs reality")
    ax.axhline(report["noise_floor"], ls="--", color="green", label="noise floor")
    ax.axhline(report["trust_band"], ls="--", color="orange", label="trust band")
    th = report["trust_horizon"]
    if th < len(report["horizons"]):
        ax.axvline(th, color="red", label=f"trust horizon = {th}")
    ax.set_xlabel("rollout horizon (frame)")
    ax.set_ylabel(f"{report['metric']} divergence")
    ax.set_title(f"World-model trust — {report['world_model']} / ep {report['episode_id']}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _frames_of(traj_or_list) -> list:
    """Accept a Trajectory or a plain list of Scenes; return the Scene list."""
    return traj_or_list.frames if isinstance(traj_or_list, Trajectory) else list(traj_or_list)


def _scene_image(scene, camera: str) -> np.ndarray:
    return np.asarray(scene.observations.images[camera].data, dtype=np.uint8)


def _resize_to_height(img: np.ndarray, height: int):
    from PIL import Image

    pil = Image.fromarray(img, mode="RGB")
    w = max(1, round(pil.width * height / pil.height))
    return pil.resize((w, height), Image.BILINEAR)


def save_frame_comparison(
    predicted,
    aligned_real,
    divergences,
    out_dir: Path,
    *,
    camera: str = "primary",
    trust_band: float = None,
    start_index: int = 0,
    panel_height: int = 256,
) -> int:
    """Write ``predicted | real`` side-by-side PNGs labelled with each frame's
    divergence and TRUSTED/DRIFT verdict. Returns the number written.

    ``predicted`` / ``aligned_real`` may be Trajectories or plain Scene lists, and
    ``start_index`` offsets the filenames (``compare_{start_index+i}.png``) — so
    this can be called once on a whole rollout OR incrementally, one segment at a
    time, to persist results as they are produced. ``trust_band`` (if given)
    colours the verdict: within band → TRUSTED, above → DRIFT.

    The two views may differ in size/content (Cosmos's concat view vs a single
    real camera); they are scaled to a common height and concatenated.
    """
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    pf, rf = _frames_of(predicted), _frames_of(aligned_real)
    n = min(len(pf), len(rf), len(divergences))

    gap, bar = 8, 22
    for i in range(n):
        left = _resize_to_height(_scene_image(pf[i], camera), panel_height)
        right = _resize_to_height(_scene_image(rf[i], camera), panel_height)
        w = left.width + gap + right.width
        canvas = Image.new("RGB", (w, panel_height + bar), (16, 16, 16))
        canvas.paste(left, (0, bar))
        canvas.paste(right, (left.width + gap, bar))

        div = float(divergences[i])
        trusted = trust_band is not None and div <= trust_band
        d = ImageDraw.Draw(canvas)
        d.text((4, 4), "predicted (Cosmos)", fill=(180, 180, 180))
        d.text((left.width + gap + 4, 4), "real episode", fill=(180, 180, 180))
        band_txt = f" (band {trust_band:.3f})" if trust_band is not None else ""
        verdict = "TRUSTED" if trusted else "DRIFT"
        color = (90, 200, 120) if trusted else (220, 90, 90)
        d.text(
            (w - 230, 4),
            f"frame {start_index + i}  div={div:.3f}{band_txt}  {verdict}",
            fill=color,
        )
        canvas.save(out_dir / f"compare_{start_index + i:03d}.png")
    return n
