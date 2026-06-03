"""Stage-2 driver — roll a real episode forward in a world model and report trust.

Wires the existing dataset-config machinery to the world-model worker and the
trust analysis: load an episode via any reader, condition the world model on its
first frame + real logged actions, and emit the trust curve + action-dependence
verdict.

Run (host side; needs the reader + world-model workers reachable)::

    uv run python -m emboviz.world_models.stage2_cli \
        --config configs/droid.yaml --episode 0 \
        --world-model cosmos3 \
        --server-url https://<podid>-8000.proxy.runpod.net \
        --domain droid_lerobot --action-dim 10 \
        --n-actions 16 --out report/cosmos_trust

The dataset (camera mapping, state/action keys, …) comes from the run config's
``dataset`` section — the same file the analyze CLI uses — so the episode is
loaded identically to a normal run. The world-model embodiment (``domain`` /
``action-dim``) is NOT inferred; the action encoding for that domain is owned by
the world-model adapter (``WorldModel.prepare_actions``). Currently implemented:
``droid_lerobot`` (10-D normalized pose deltas). The conditioning offset (which
real frame each predicted frame maps to) is exposed for the embodiment cadence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.adapters import connect_world_model
from emboviz.world_models.rollout import summarize, trust_report


def _save_plot(report: dict, path: Path) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="run config (its dataset section is reused)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode index (default: first in the config's analysis.episodes)")
    p.add_argument("--world-model", default="cosmos3")
    p.add_argument("--server-url", required=True)
    p.add_argument("--domain", required=True, help="world-model domain_name (embodiment)")
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--n-actions", type=int, default=16)
    p.add_argument("--camera", default="primary")
    p.add_argument("--metric", default="pixel_l2", choices=["pixel_l2", "ssim"])
    p.add_argument("--conditioning-offset", type=int, default=1)
    p.add_argument("--out", default="report/cosmos_trust")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[stage2] loading episode {episode} via {cfg.dataset.format} reader ...")
    source = build_source(**cfg.dataset_build_kwargs())
    real = source.load_trajectory(episode)
    print(f"[stage2] episode has {len(real.frames)} frames")

    print(f"[stage2] connecting world model '{args.world_model}' @ {args.server_url}")
    wm = connect_world_model(args.world_model, world_model_kwargs={
        "server_url": args.server_url,
        "domain_name": args.domain,
        "action_dim": args.action_dim,
        "conditioning_camera": args.camera,
    })

    report = trust_report(
        wm, real,
        frame_start=args.frame_start, n_actions=args.n_actions,
        camera=args.camera, metric=args.metric,
        conditioning_offset=args.conditioning_offset,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trust_report.json").write_text(json.dumps(report, indent=2))
    _save_plot(report, out / "trust_curve.png")

    print("\n" + summarize(report))
    print(f"\n[stage2] wrote {out}/trust_report.json and trust_curve.png")


if __name__ == "__main__":
    main()
