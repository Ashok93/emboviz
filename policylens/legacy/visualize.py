"""Render attribution heatmaps as a side-by-side GIF and a frame-grid PNG.

We use matplotlib (not OpenCV) because it ships everywhere, handles colormaps
cleanly, and the output quality is fine for a hypothesis check. The GIF and
PNG are the *only* deliverables that need to look presentable — keep all
other complexity out of this module.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from policylens.attribute import FrameAttributions
from policylens.load import EpisodeFrames
from policylens.replay import ReplayResult


_METHOD_ORDER = ["raw", "ig", "saliency", "random"]
_METHOD_TITLES = {
    "raw": "Frame",
    "ig": "Integrated Gradients",
    "saliency": "Saliency",
    "random": "Random baseline",
}


def render_side_by_side_gif(
    episode: EpisodeFrames,
    attributions: dict[int, FrameAttributions],
    out_path: Path,
    failure_idx: int,
) -> None:
    """Render a GIF: each animated frame is a 4-panel row across methods.

    Frames without computed attribution are skipped (we only attribute the
    keyframes — full-episode IG would be wasteful for a hypothesis check).
    """
    keyframes = sorted(attributions.keys())
    images = []

    for idx in keyframes:
        frame = _to_hwc_uint8(episode.images[idx])
        panels = {
            "raw": frame,
            "ig": _overlay(frame, attributions[idx].ig),
            "saliency": _overlay(frame, attributions[idx].saliency),
            "random": _overlay(frame, attributions[idx].random),
        }
        fig = _compose_row(panels, title=f"t={idx}" + ("  ← failure frame" if idx == failure_idx else ""))
        images.append(_fig_to_rgb(fig))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ~2 fps — slow enough to actually look at each frame.
    imageio.mimsave(out_path, images, duration=0.5, loop=0)


def render_frame_grid_png(
    episode: EpisodeFrames,
    attributions: dict[int, FrameAttributions],
    out_path: Path,
    failure_idx: int,
) -> None:
    """Render an N×4 grid PNG: rows are timesteps, cols are methods."""
    keyframes = sorted(attributions.keys())
    n_rows = len(keyframes)
    n_cols = len(_METHOD_ORDER)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), squeeze=False
    )
    for col, method in enumerate(_METHOD_ORDER):
        axes[0, col].set_title(_METHOD_TITLES[method], fontsize=10)
    for row, idx in enumerate(keyframes):
        frame = _to_hwc_uint8(episode.images[idx])
        attr = attributions[idx]
        panels = {
            "raw": frame,
            "ig": _overlay(frame, attr.ig),
            "saliency": _overlay(frame, attr.saliency),
            "random": _overlay(frame, attr.random),
        }
        label = f"t={idx}" + ("\n(failure)" if idx == failure_idx else "")
        for col, method in enumerate(_METHOD_ORDER):
            ax = axes[row, col]
            ax.imshow(panels[method])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_deviation_plot(result: ReplayResult, out_path: Path) -> None:
    """Plot policy-vs-expert action deviation across the episode.

    Useful context for the failure frame: did the policy gradually drift, or
    blow up at one timestep? The shape of this curve frames the GIF.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(result.action_deviations.numpy())
    ax.axvline(result.failure_frame_idx, color="red", linestyle="--", label="failure frame")
    ax.set_xlabel("timestep")
    ax.set_ylabel("||policy − expert|| (L2)")
    ax.set_title("Per-timestep action deviation from expert")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- internals ---------------------------------------------------------------


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    a = image.detach().cpu().float().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = a.transpose(1, 2, 0)
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return a


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB uint8 frame using the 'jet' colormap."""
    # Resize heatmap to frame size if needed (image encoder may downsample).
    if heatmap.shape != frame.shape[:2]:
        heatmap = _resize_2d(heatmap, frame.shape[:2])
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = (frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _resize_2d(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour 2D resize without bringing in OpenCV."""
    from PIL import Image

    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _compose_row(panels: dict[str, np.ndarray], title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(_METHOD_ORDER), figsize=(3 * len(_METHOD_ORDER), 3.4))
    for ax, method in zip(axes, _METHOD_ORDER):
        ax.imshow(panels[method])
        ax.set_title(_METHOD_TITLES[method], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()
