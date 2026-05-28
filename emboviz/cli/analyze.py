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


# Legacy in-process adapter aliases. The new path is the entry-point
# adapter registry — ``--model openvla`` first looks for an installed
# ``emboviz-openvla`` package and routes through Ray. Aliases below
# are only used when (a) no matching adapter is installed AND (b) the
# legacy in-process module is still present (mock + lerobot). Once
# every VLA adapter has been migrated to a Ray-actor package, this
# table shrinks to ``mock`` and ``lerobot``.
_LEGACY_MODEL_ALIASES: dict[str, str] = {
    "openvla-7b":  "adapter:openvla",
    "openvla-oft": "adapter:oft",
    "oft":         "adapter:oft",
    "pi05":        "adapter:pi0",
    "gr00t-n1":    "adapter:gr00t",
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


# Generic data-format shortcuts for users with their own dataset / recording
# at a local path. Maps a short format name to (adapter_spec, path-kwarg-name).
# Selected via ``--dataset-format <fmt> --dataset-path <path>``; any other
# adapter-specific kwargs (camera_keys, topic_map, builder_name, ...) go in
# ``--dataset-kwargs '<JSON>'``.
#
# These are the formats whose ALL required adapter kwargs are JSON-friendly
# (dicts of strings, paths). LeRobot v2/v3 and generic HuggingFace datasets
# are intentionally NOT in this dict because their adapters need a
# ``RobotProfile`` instance (and, for HF, a ``row_to_scene`` callable) that
# can't be expressed in JSON. For those, use one of the pre-shipped
# ``--dataset <alias>`` (bridge / libero-* / droid-* / aloha-*) which bakes
# in the right profile, or subclass ``LeRobotEpisodeSource`` for your own
# robot and pass it via ``--dataset emboviz.your_module:YourSource``.
_DATASET_FORMATS: dict[str, tuple[str, str]] = {
    # name             (adapter spec,                                  path-kwarg)
    "hdf5":           ("emboviz.datasets:HDF5EpisodeSource",          "path"),
    "rlds":           ("emboviz.datasets:RLDSEpisodeSource",          "data_dir"),
    "mcap":           ("emboviz.recordings:MCAPRecording",            "path"),
    "rerun-rrd":      ("emboviz.recordings:RerunRecording",           "path"),
}


def _resolve_model_spec(model: str) -> str:
    """Resolve ``--model <X>`` to a spec the runner understands.

    Lookup order:

      1. Verbatim ``adapter:<name>`` (or any ``module:Class`` form) →
         passed through; runner handles each.
      2. Match against the installed adapter entry-point registry. If
         a package called ``emboviz-<model>`` is installed, route as
         ``adapter:<model>``. This is the new common path.
      3. Match against the legacy alias table — handles aliases like
         ``openvla-7b`` and the still-in-process ``mock`` adapter.
      4. Otherwise: raise a useful error listing what IS installed
         AND the legacy aliases, so the user always knows their next
         move (install a missing adapter package or fix a typo).
    """
    if ":" in model:
        return model

    # Prefer installed adapter packages over legacy aliases.
    from emboviz.adapters import list_adapters
    installed = list_adapters()
    if model in installed:
        return f"adapter:{model}"

    if model in _LEGACY_MODEL_ALIASES:
        return _LEGACY_MODEL_ALIASES[model]

    available_adapters = sorted(installed)
    legacy = sorted(_LEGACY_MODEL_ALIASES)
    raise click.UsageError(
        f"Unknown model '{model}'.\n"
        f"  Installed adapters (entry-point): {available_adapters or '(none)'}\n"
        f"  Legacy aliases:                   {legacy}\n"
        f"  Power-user form:                  --model <module>:<Class>\n"
        f"  To add '{model}' as a Ray-actor adapter, run:\n"
        f"      uv pip install emboviz-{model}\n"
        f"      emboviz install-{model}"
    )


def _resolve_dataset_spec(dataset: str) -> str:
    if dataset in _DATASET_ALIASES:
        return _DATASET_ALIASES[dataset]
    if ":" in dataset:
        return dataset
    raise click.UsageError(
        f"Unknown dataset '{dataset}'. Choose one of: "
        + ", ".join(sorted(_DATASET_ALIASES))
        + ". For generic local data, use --dataset-format + --dataset-path."
    )


def _resolve_dataset_from_args(
    dataset: Optional[str],
    dataset_format: Optional[str],
    dataset_path: Optional[str],
    dataset_kwargs_json: str,
) -> tuple[str, str]:
    """Combine the three dataset-selection flags into a (spec, kwargs_json) pair.

    Three valid combinations:
      1. ``--dataset <alias-or-spec>`` only — pre-shipped dataset; kwargs
         come solely from ``--dataset-kwargs``.
      2. ``--dataset-format <fmt> --dataset-path <p>`` — generic adapter
         pointed at a local file/dir/repo; extra kwargs in ``--dataset-kwargs``
         get merged on top.
      3. ``--dataset emboviz.module:Class --dataset-kwargs '{...}'`` —
         power-user explicit module path.

    Mutually exclusive: cannot pass both ``--dataset`` and
    ``--dataset-format``.
    """
    import json

    if dataset and dataset_format:
        raise click.UsageError(
            "Pass EITHER --dataset (alias / module:class) OR "
            "--dataset-format (generic format shortcut), not both."
        )
    if not dataset and not dataset_format:
        raise click.UsageError(
            "Specify a dataset via either:\n"
            "  --dataset <alias>            (e.g. bridge, libero-spatial, droid-sample)\n"
            "  --dataset-format <fmt>       (e.g. lerobot, hdf5, mcap, rlds, hf)\n"
            "       --dataset-path <path>   (required when --dataset-format is set)"
        )

    if dataset_format:
        if dataset_format not in _DATASET_FORMATS:
            raise click.UsageError(
                f"Unknown --dataset-format '{dataset_format}'. Available: "
                + ", ".join(sorted(_DATASET_FORMATS))
            )
        if not dataset_path:
            raise click.UsageError(
                f"--dataset-format {dataset_format} requires --dataset-path."
            )
        spec, path_kwarg = _DATASET_FORMATS[dataset_format]
        extra = {}
        if dataset_kwargs_json:
            try:
                extra = json.loads(dataset_kwargs_json)
                if not isinstance(extra, dict):
                    raise ValueError("must be a JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                raise click.UsageError(f"--dataset-kwargs is not valid JSON: {e}")
        if path_kwarg in extra and extra[path_kwarg] != dataset_path:
            raise click.UsageError(
                f"--dataset-path={dataset_path!r} conflicts with "
                f"--dataset-kwargs key {path_kwarg}={extra[path_kwarg]!r}. "
                f"Use one or the other."
            )
        extra[path_kwarg] = dataset_path
        return spec, json.dumps(extra)

    return _resolve_dataset_spec(dataset), dataset_kwargs_json


@click.command("analyze")
@click.option("--model", required=True,
              help="Model adapter alias (e.g. 'openvla', 'oft', 'pi0', 'gr00t').")
@click.option("--model-kwargs", "model_kwargs_json", default="",
              help="JSON dict of kwargs passed to the model adapter constructor.")
@click.option("--dataset", default=None,
              help="Pre-shipped dataset alias (e.g. 'bridge', 'libero-spatial', "
                   "'droid-sample') OR a full 'module.path:Class' spec. "
                   "Mutually exclusive with --dataset-format.")
@click.option("--dataset-format", "dataset_format", default=None,
              type=click.Choice(sorted(_DATASET_FORMATS), case_sensitive=False),
              help="Generic dataset / recording format for users with their "
                   "own local data. Choose from: "
                   + ", ".join(sorted(_DATASET_FORMATS))
                   + ". Use together with --dataset-path. Adapter-specific "
                     "options (camera_keys, builder_name, topic_map, ...) go "
                     "in --dataset-kwargs.")
@click.option("--dataset-path", "dataset_path", default=None,
              type=click.Path(),
              help="Path to the local dataset file / directory / HF repo id. "
                   "Required when --dataset-format is given; ignored otherwise.")
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
                   "the text-prompted detector (SAM 3 by default; --detector "
                   "gd-sam for the legacy GroundingDINO+SAM combo). If empty "
                   "and --target-annotations is also empty, memorization is "
                   "skipped.")
@click.option("--target-annotations", "target_annotations", default="",
              type=click.Path(),
              help="Per-frame target-annotation file (JSON or COCO). When "
                   "set, replaces text-prompted detection entirely — no "
                   "GroundingDINO, no SAM, no SAM 3. Use when your tracker "
                   "(motion capture, fiducials, hand labels) already knows "
                   "where the target is.")
@click.option("--detector", "detector", default="sam3",
              type=click.Choice(["sam3", "gd-sam"], case_sensitive=False),
              help="Text-to-mask detector backend. 'sam3' (default) — single "
                   "model, faster, native concept prompting. 'gd-sam' — "
                   "legacy GroundingDINO + SAM combo, kept as a maintained "
                   "fallback. Ignored when --target-annotations is set.")
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
@click.option("--dry-run", "dry_run", is_flag=True, default=False,
              help="Print the per-frame and per-episode forward-pass "
                   "estimate without running the diagnostic suite. Use "
                   "this BEFORE committing GPU hours on a long episode.")
def analyze_cmd(
    model: str, model_kwargs_json: str,
    dataset: Optional[str],
    dataset_format: Optional[str], dataset_path: Optional[str],
    dataset_kwargs_json: str,
    episodes_spec: str, frame_start: int, n_frames: int, frame_stride: int,
    target_text: str, target_annotations: str, detector: str,
    out_dir: str,
    sensitivity_grid_side: int,
    modality_pool_size: int, modality_k_samples: int,
    modality_pool_seed: int, modality_pool_cache_dir: Optional[str],
    show_imitation: bool, dry_run: bool,
) -> None:
    """Analyze a model on one or more episodes and write diagnostics.

    Examples:

    \b
        # Pre-shipped dataset (LeRobot / Bridge), all frames of episode 0
        emboviz analyze --model openvla --dataset bridge --episodes 0 \\
            --target "the spoon" --output ./report

    \b
        # Local HDF5 file (Robomimic / ALOHA / Isaac Lab Mimic). The
        # camera_keys / state_key / instruction tell the adapter where
        # in the HDF5 hierarchy to pull each modality from.
        emboviz analyze --model pi0 \\
            --dataset-format hdf5 --dataset-path /data/demos.hdf5 \\
            --dataset-kwargs '{"camera_keys": {"primary": "agentview_rgb"}, "instruction": "pick up the mug"}' \\
            --episodes 0 --target "the white mug" --output ./report

    \b
        # RLDS / TFDS (needs `uv pip install 'emboviz[rlds]'`)
        emboviz analyze --model gr00t \\
            --dataset-format rlds --dataset-path /tfds \\
            --dataset-kwargs '{"builder_name": "bridge_orig", "camera_keys": {"primary": "image_0"}}' \\
            --episodes 0,1 --target "the green block" --output ./report

    \b
        # MCAP deployment recording (ROS 2 / Isaac SIM)
        emboviz analyze --model gr00t \\
            --dataset-format mcap --dataset-path /logs/rollout.mcap \\
            --dataset-kwargs '{"topic_map": {"primary": "/camera/color/image_raw", "state": "/joint_states", "action": "/cmd_joint"}}' \\
            --episodes 0 --target "the green block" --output ./report

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
    dataset_spec, dataset_kwargs_json = _resolve_dataset_from_args(
        dataset=dataset,
        dataset_format=dataset_format.lower() if dataset_format else None,
        dataset_path=dataset_path,
        dataset_kwargs_json=dataset_kwargs_json,
    )
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
            target_annotations=target_annotations,
            detector=detector,
            show_imitation=show_imitation,
            dry_run=dry_run,
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
