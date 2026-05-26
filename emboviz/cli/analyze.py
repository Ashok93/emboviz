"""`emboviz analyze` — the single user-facing analyze command.

The user gives:
  • Their model (an adapter name we ship, or an HF repo id)
  • Their dataset OR deployment recording (path or HF id)
  • An episode (or range) to analyze
  • The target object (for memorization grounding)
  • Where to write the report

We produce:
  • Per-episode summary.json with Findings + raw numbers
  • Per-episode rollout.rrd with overlays scrubbable in Rerun
  • (Phase 9) Markdown / HTML report

This module is a thin click wrapper that translates user-friendly args
into the AnalysisConfig the orchestrator (``emboviz._internal.runner``)
already speaks. The orchestrator does the heavy lifting; this file
exists to give users a clean CLI surface that matches the README, NOT
to re-implement the runner.

Phase 4 scope: single-episode analyze. Phase 5 extends this to
multi-episode / full-episode aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import click


# Built-in model adapter shortcuts → ``module:attr:registry-key`` specs
# the orchestrator's ``_resolve`` can consume. Users typing
# ``--model openvla`` get this expansion; users with their own fine-tune
# pass a HuggingFace repo id and we route through the matching base
# adapter (set via ``--adapter`` when ambiguous).
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

# Built-in dataset adapter shortcuts.
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
    if model.count("/") >= 1 and ":" not in model:
        # Looks like a HuggingFace repo id (e.g. "org/my-finetuned-openvla").
        # Defer to a per-adapter resolver in the future; for now error.
        raise click.UsageError(
            f"HuggingFace repo id '{model}' resolution is not implemented "
            "yet in Phase 4. Use an adapter alias from this list: "
            + ", ".join(sorted(_MODEL_ALIASES))
        )
    if ":" in model:
        return model  # power-user passed a full module:attr[:key] spec
    raise click.UsageError(
        f"Unknown model '{model}'. Choose one of: "
        + ", ".join(sorted(_MODEL_ALIASES))
        + " — or pass a full module:attr:key spec (advanced)."
    )


def _resolve_dataset_spec(dataset: str) -> str:
    if dataset in _DATASET_ALIASES:
        return _DATASET_ALIASES[dataset]
    if ":" in dataset:
        return dataset
    raise click.UsageError(
        f"Unknown dataset '{dataset}'. Choose one of: "
        + ", ".join(sorted(_DATASET_ALIASES))
        + " — or pass a full module:attr spec (advanced)."
    )


@click.command("analyze")
@click.option("--model", required=True,
              help="Model adapter alias (e.g. 'openvla', 'oft', 'pi0', "
                   "'gr00t') OR a full 'module:attr[:registry-key]' spec.")
@click.option("--model-kwargs", "model_kwargs_json", default="",
              help="JSON dict of kwargs passed to the model adapter "
                   "constructor (e.g. '{\"camera_mapping\": {...}}' for GR00T).")
@click.option("--dataset", required=True,
              help="Dataset alias (e.g. 'bridge', 'libero-spatial', "
                   "'droid-sample') OR a full 'module:attr' spec.")
@click.option("--dataset-kwargs", "dataset_kwargs_json", default="",
              help="JSON dict of kwargs passed to the dataset adapter "
                   "constructor.")
@click.option("--episode", "episode_idx", type=int, required=True,
              help="Episode index in the dataset.")
@click.option("--frame-start", type=int, default=0,
              help="First frame index in the window (default 0).")
@click.option("--n-frames", type=int, default=8,
              help="How many frames in the window (default 8). Phase 5 "
                   "extends this to full-episode aggregation.")
@click.option("--target", "target_text", default="",
              help="Target object phrase (e.g. 'the red cup') passed to "
                   "GroundingDINO for the memorization diagnostic. "
                   "If empty, memorization is skipped.")
@click.option("--output", "out_dir", type=click.Path(), required=True,
              help="Output directory. Per-episode summary.json + "
                   "rollout.rrd are written here.")
@click.option("--sensitivity-grid-side", type=int, default=4,
              help="Side length of the occlusion grid for "
                   "scene-sensitivity (default 4 → 16 patches).")
@click.option("--modality-pool-size", type=int, default=20,
              help="Episodes sampled to build the SHAP-marginal "
                   "modality dropout pool (default 20).")
@click.option("--modality-k-samples", type=int, default=10,
              help="Substitution samples per modality per frame "
                   "(default 10).")
@click.option("--modality-pool-seed", type=int, default=0,
              help="RNG seed for the modality pool sampler — keep "
                   "consistent across runs to hit the disk cache.")
@click.option("--modality-pool-cache-dir", type=click.Path(), default=None,
              help="Optional directory where the modality pool is cached "
                   "on disk. Shared between probe + runner.")
@click.option("--show-imitation", is_flag=True, default=False,
              help="Compute and show imitation L2 vs recorded expert "
                   "action. Hidden by default — see emboviz analyze --help "
                   "for why.")
def analyze_cmd(
    model: str, model_kwargs_json: str,
    dataset: str, dataset_kwargs_json: str,
    episode_idx: int, frame_start: int, n_frames: int,
    target_text: str, out_dir: str,
    sensitivity_grid_side: int,
    modality_pool_size: int, modality_k_samples: int,
    modality_pool_seed: int, modality_pool_cache_dir: Optional[str],
    show_imitation: bool,
) -> None:
    """Analyze a model on an episode and write diagnostics.

    Examples:

    \b
        emboviz analyze --model openvla --dataset bridge \\
            --episode 0 --frame-start 10 --n-frames 8 \\
            --target "the spoon" --output ./report

    \b
        emboviz analyze --model gr00t \\
            --model-kwargs '{"camera_mapping": {"primary": "exterior_image_1_left", "wrist_left": "wrist_image_left"}}' \\
            --dataset droid-sample --episode 1 --frame-start 80 \\
            --target "the blue block" --output ./report
    """
    model_spec = _resolve_model_spec(model)
    dataset_spec = _resolve_dataset_spec(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Translate click options into the argparse Namespace the
    # orchestrator already speaks. (Phase 5 will replace this with a
    # proper AnalysisConfig dataclass alongside the aggregation refactor.)
    args = argparse.Namespace(
        story_id=f"{model}:{dataset}:ep{episode_idx}_f{frame_start}",
        model_builder=model_spec,
        model_kwargs_json=model_kwargs_json,
        dataset_builder=dataset_spec,
        dataset_kwargs_json=dataset_kwargs_json,
        episode_idx=episode_idx,
        frame_start=frame_start,
        n_frames=n_frames,
        sensitivity_grid_side=sensitivity_grid_side,
        out_dir=str(out),
        modality_pool_size=modality_pool_size,
        modality_k_samples=modality_k_samples,
        modality_pool_seed=modality_pool_seed,
        modality_pool_cache_dir=modality_pool_cache_dir,
        target_text=target_text,
        show_imitation=show_imitation,
    )

    # Defer heavy import so `emboviz --help` works without torch installed.
    from emboviz._internal.runner import run_story
    try:
        run_story(args)
    except Exception as e:
        click.echo(f"emboviz analyze: {type(e).__name__}: {e}", err=True)
        sys.exit(1)
