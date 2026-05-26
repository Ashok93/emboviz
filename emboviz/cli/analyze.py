"""`emboviz analyze` — the single user-facing analyze command.

The user gives:
  • Their model (an adapter alias or a full ``module:attr[:registry-key]`` spec)
  • Their dataset (an alias or full spec)
  • One or more episodes to analyze (``--episodes 0`` or ``"0,3,7"`` or ``"0-5"`` or ``"all"``)
  • Where to write the report
  • Optionally: the target object phrase for memorization

We produce, per episode:
  • ``<out>/episode_<idx>/summary.json``  — per-axis Findings + raw numbers
  • ``<out>/episode_<idx>/rollout.rrd``  — Rerun playback w/ overlays

And across all episodes:
  • ``<out>/aggregate.json``  — cross-episode patterns
  • ``<out>/aggregate.md``    — human-readable summary

Phase 5 scope: full-episode-by-default, multi-episode loops, cross-
episode aggregation. The actual diagnostic orchestration is unchanged
from Phase 4 (calls ``emboviz._internal.runner.run_story`` per episode).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import click


# Built-in model adapter shortcuts → ``module:attr:registry-key`` specs.
_MODEL_ALIASES: dict[str, str] = {
    "openvla":     "emboviz.models.registry:get_model:openvla",
    "openvla-7b":  "emboviz.models.registry:get_model:openvla",
    "oft":         "emboviz.models.registry:get_model:openvla-oft",
    "openvla-oft": "emboviz.models.registry:get_model:openvla-oft",
    "pi0":         "emboviz.models.registry:get_model:pi0",
    "pi05":        "emboviz.models.registry:get_model:pi0",
    "gr00t":       "emboviz.models.registry:get_model:gr00t",
    "gr00t-n1":    "emboviz.models.registry:get_model:gr00t",
    "mock":        "emboviz.models.registry:get_model:mock",
}

_DATASET_ALIASES: dict[str, str] = {
    "bridge":           "emboviz.datasets.lerobot_bridge:BridgeEpisodeSource",
    "libero-spatial":   "emboviz.datasets.lerobot_libero:LiberoSpatialSource",
    "libero-object":    "emboviz.datasets.lerobot_libero:LiberoObjectSource",
    "libero-goal":      "emboviz.datasets.lerobot_libero:LiberoGoalSource",
    "libero-10":        "emboviz.datasets.lerobot_libero:Libero10Source",
    "pi-libero":        "emboviz.datasets.lerobot_libero:PhysicalIntelligenceLiberoSource",
    "droid-100":        "emboviz.datasets.lerobot_droid:Droid100Source",
    "droid-full":       "emboviz.datasets.lerobot_droid:DroidFullSource",
    "droid-sample":     "emboviz.datasets.lerobot_droid:GR00TDroidSampleSource",
    "aloha-transfer":   "emboviz.datasets.lerobot_aloha:AlohaSimTransferCubeSource",
    "aloha-insertion":  "emboviz.datasets.lerobot_aloha:AlohaSimInsertionSource",
}


def _resolve_model_spec(model: str) -> str:
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if ":" in model:
        return model
    if "/" in model:
        raise click.UsageError(
            f"HuggingFace repo id '{model}' resolution is not implemented "
            "yet. Use an adapter alias from this list: "
            + ", ".join(sorted(_MODEL_ALIASES))
        )
    raise click.UsageError(
        f"Unknown model '{model}'. Choose one of: "
        + ", ".join(sorted(_MODEL_ALIASES))
    )


def _resolve_dataset_spec(dataset: str) -> str:
    if dataset in _DATASET_ALIASES:
        return _DATASET_ALIASES[dataset]
    if ":" in dataset:
        return dataset
    raise click.UsageError(
        f"Unknown dataset '{dataset}'. Choose one of: "
        + ", ".join(sorted(_DATASET_ALIASES))
    )


@click.command("analyze")
@click.option("--model", required=True,
              help="Model adapter alias (e.g. 'openvla', 'oft', 'pi0', 'gr00t').")
@click.option("--model-kwargs", "model_kwargs_json", default="",
              help="JSON dict of kwargs passed to the model adapter constructor.")
@click.option("--dataset", required=True,
              help="Dataset alias (e.g. 'bridge', 'libero-spatial', 'droid-sample').")
@click.option("--dataset-kwargs", "dataset_kwargs_json", default="",
              help="JSON dict of kwargs passed to the dataset adapter constructor.")
@click.option("--episodes", "episodes_spec", default="0",
              help="Episode(s) to analyze. Forms: '7' / '0,3,7' / '0-5' / 'all'. "
                   "Default: '0'.")
@click.option("--frame-start", type=int, default=0,
              help="First frame index in the analysis window (default 0).")
@click.option("--n-frames", type=int, default=-1,
              help="Number of frames in the window. Default -1 = ALL "
                   "frames from --frame-start to the end of each episode.")
@click.option("--frame-stride", type=int, default=1,
              help="Stride between sampled frames in the window. "
                   "Default 1 = every frame. Set to 5 or 10 to "
                   "subsample long episodes.")
@click.option("--target", "target_text", default="",
              help="Target object phrase (e.g. 'the red cup') passed to "
                   "GroundingDINO for the memorization diagnostic. "
                   "If empty, memorization is skipped.")
@click.option("--output", "out_dir", type=click.Path(), required=True,
              help="Output directory. Per-episode subdirs + aggregate "
                   "report are written here.")
@click.option("--sensitivity-grid-side", type=int, default=4,
              help="Side length of the occlusion grid for scene-sensitivity (default 4).")
@click.option("--modality-pool-size", type=int, default=20,
              help="Episodes sampled to build the SHAP-marginal pool (default 20).")
@click.option("--modality-k-samples", type=int, default=10,
              help="Substitution samples per modality per frame (default 10).")
@click.option("--modality-pool-seed", type=int, default=0,
              help="RNG seed for the modality pool sampler.")
@click.option("--modality-pool-cache-dir", type=click.Path(), default=None,
              help="Optional directory where the modality pool is cached on disk.")
@click.option("--show-imitation", is_flag=True, default=False,
              help="Compute and show imitation L2 vs recorded expert action.")
def analyze_cmd(
    model: str, model_kwargs_json: str,
    dataset: str, dataset_kwargs_json: str,
    episodes_spec: str, frame_start: int, n_frames: int, frame_stride: int,
    target_text: str, out_dir: str,
    sensitivity_grid_side: int,
    modality_pool_size: int, modality_k_samples: int,
    modality_pool_seed: int, modality_pool_cache_dir: Optional[str],
    show_imitation: bool,
) -> None:
    """Analyze a model on one or more episodes and write diagnostics.

    Examples:

    \b
        # Single episode, all frames
        emboviz analyze --model openvla --dataset bridge --episodes 0 \\
            --target "the spoon" --output ./report

    \b
        # Multiple episodes with stride
        emboviz analyze --model oft --dataset libero-spatial \\
            --episodes "0,3,7,12" --frame-stride 5 \\
            --target "the black bowl" --output ./report

    \b
        # All episodes (use with caution on big datasets)
        emboviz analyze --model pi0 --dataset pi-libero \\
            --episodes all --frame-stride 10 \\
            --target "the white mug" --output ./report
    """
    from emboviz._internal.multi_episode import (
        EpisodeReport,
        aggregate_episodes,
        parse_episode_spec,
        write_aggregate_html,
        write_aggregate_markdown,
    )
    from emboviz._internal.report import write_episode_reports

    model_spec = _resolve_model_spec(model)
    dataset_spec = _resolve_dataset_spec(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve --episodes. For "all" we need the dataset's episode count.
    n_available: Optional[int] = None
    if episodes_spec.strip() == "all":
        # Defer to dataset to enumerate. Heavy import, hence inline.
        from emboviz._internal.runner import _resolve as resolve_builder
        click.echo("[analyze] resolving dataset to count episodes for --episodes all ...")
        ds = resolve_builder(dataset_spec, dataset_kwargs_json)
        try:
            n_available = len(ds.list_episodes())
        except Exception as e:
            raise click.UsageError(
                f"--episodes all requires dataset.list_episodes() to work: {e}"
            )
    episode_indices = parse_episode_spec(episodes_spec, n_available)
    if not episode_indices:
        raise click.UsageError("--episodes resolved to an empty list")
    click.echo(f"[analyze] analyzing {len(episode_indices)} episode(s): "
               f"{episode_indices[:10]}{' ...' if len(episode_indices) > 10 else ''}")

    # Per-episode loop. We re-load the model + dataset between episodes
    # because run_story does so internally. (Phase 6+ will hoist model
    # load above the loop for amortization.)
    episode_reports: list[EpisodeReport] = []
    for ep_idx in episode_indices:
        ep_dir = out / f"episode_{ep_idx:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"\n[analyze] ===== episode {ep_idx} =====")

        args = argparse.Namespace(
            story_id=f"{model}:{dataset}:ep{ep_idx}",
            model_builder=model_spec,
            model_kwargs_json=model_kwargs_json,
            dataset_builder=dataset_spec,
            dataset_kwargs_json=dataset_kwargs_json,
            episode_idx=ep_idx,
            frame_start=frame_start,
            n_frames=n_frames,
            frame_stride=frame_stride,
            sensitivity_grid_side=sensitivity_grid_side,
            out_dir=str(ep_dir),
            modality_pool_size=modality_pool_size,
            modality_k_samples=modality_k_samples,
            modality_pool_seed=modality_pool_seed,
            modality_pool_cache_dir=modality_pool_cache_dir,
            target_text=target_text,
            show_imitation=show_imitation,
        )

        from emboviz._internal.runner import run_story
        try:
            run_story(args)
        except Exception as e:
            click.echo(f"[analyze] episode {ep_idx} FAILED: "
                       f"{type(e).__name__}: {e}", err=True)
            traceback.print_exc()
            # Don't abort the whole multi-episode run on one bad episode.
            continue

        summary_path = ep_dir / "summary.json"
        rrd_path = ep_dir / "rollout.rrd"
        if summary_path.exists():
            ep_report = EpisodeReport(
                episode_idx=ep_idx,
                out_dir=ep_dir,
                summary_path=summary_path,
                rollout_rrd_path=rrd_path if rrd_path.exists() else None,
            )
            episode_reports.append(ep_report)

            # Per-episode human-readable reports (md + html). HTML only
            # when the `viz` extra is installed; markdown always.
            try:
                summary_dict = json.loads(summary_path.read_text())
                paths = write_episode_reports(
                    summary_dict, ep_dir,
                    rrd_path=str(rrd_path) if rrd_path.exists() else None,
                )
                click.echo(f"[analyze] wrote {paths['md']}"
                           + (f" + {paths['html']}" if paths.get("html") else ""))
            except Exception as e:
                click.echo(f"[analyze] episode {ep_idx} report rendering FAILED: "
                           f"{type(e).__name__}: {e}", err=True)

    # Aggregate cross-episode patterns.
    if not episode_reports:
        click.echo("\n[analyze] no episodes produced summary.json — nothing to aggregate.", err=True)
        sys.exit(1)
    click.echo(f"\n[analyze] aggregating across {len(episode_reports)} episode(s) ...")
    aggregate = aggregate_episodes(episode_reports)
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str))
    md = write_aggregate_markdown(aggregate, model_id=model, out_path=out / "aggregate.md")
    html = write_aggregate_html(
        aggregate, model_id=model, episodes=episode_reports,
        out_path=out / "aggregate.html",
    )
    click.echo(f"[analyze] wrote {out / 'aggregate.json'}")
    click.echo(f"[analyze] wrote {md}")
    if html is not None:
        click.echo(f"[analyze] wrote {html}")
    else:
        click.echo("[analyze] (skipped aggregate.html — install 'emboviz[viz]' for HTML reports)")
    click.echo(f"[analyze] per-episode reports in {out}/episode_*/")
