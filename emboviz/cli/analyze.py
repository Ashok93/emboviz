"""`emboviz analyze` — the single user-facing analyze command.

One run is fully described by ONE config file (see :mod:`emboviz.config`
and the shipped templates under ``configs/``). The config declares the
model (adapter + the user's checkpoint kwargs), the dataset mapping
(format + path + the camera / state-convention / gripper bindings the
format can't encode), and the analysis parameters (episodes, memorization
target, diagnostics, output). There is no CLI flag soup:

    emboviz analyze --config configs/openvla-bridge.yaml
    emboviz analyze --config openvla-bridge          # shipped template name
    emboviz analyze --config my-run.yaml --dry-run   # cost estimate only

We produce, per episode:
  • ``<out>/episode_<idx>/summary.json``  — per-axis Findings + raw numbers
  • ``<out>/episode_<idx>/rollout.rrd``   — Rerun playback w/ overlays
  • ``<out>/episode_<idx>/report.md``     — human-readable per-episode report

And across all episodes:
  • ``<out>/aggregate.json`` / ``aggregate.md`` / ``aggregate.html``

The diagnostic orchestration is unchanged — the config is resolved into
the same per-episode ``emboviz._internal.runner.run_story`` call the
runner has always used; only the inputs now come from one file.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import click


# The dataset section of every config is read through the one uniform
# manifest builder, regardless of format (lerobot / hdf5 / rlds). The
# runner resolves this ``module:attr`` spec and calls
# ``build_source(**dataset_section)``.
_MANIFEST_BUILDER = "emboviz.datasets.manifest:build_source"


# Legacy in-process adapter aliases, still accepted as a ``model.adapter``
# value. The common path is the entry-point adapter registry — ``adapter:
# openvla`` connects to (or spawns) the installed ``emboviz-openvla``
# worker. ``mock`` is the in-process test adapter.
_LEGACY_MODEL_ALIASES: dict[str, str] = {
    "openvla-7b":  "adapter:openvla",
    "openvla-oft": "adapter:oft",
    "pi05":        "adapter:pi0",
    "gr00t-n1":    "adapter:gr00t",
    "mock":        "emboviz.models.registry:get_model:mock",
}


def _resolve_model_spec(adapter: str) -> str:
    """Resolve ``model.adapter`` to a spec the runner understands.

    Lookup order:

      1. Verbatim ``adapter:<name>`` (or any ``module:Class`` form) →
         passed through; the runner handles each.
      2. Match against the installed adapter entry-point registry. If a
         package called ``emboviz-<adapter>`` is installed, route as
         ``adapter:<adapter>`` (the common path).
      3. Match against the legacy alias table (``mock`` + back-compat
         aliases).
      4. Otherwise: raise listing what IS installed and the aliases, so
         the user knows their next move.
    """
    if ":" in adapter:
        return adapter

    from emboviz.adapters import list_adapters
    installed = list_adapters()
    if adapter in installed:
        return f"adapter:{adapter}"

    if adapter in _LEGACY_MODEL_ALIASES:
        return _LEGACY_MODEL_ALIASES[adapter]

    raise click.UsageError(
        f"Unknown model.adapter '{adapter}'.\n"
        f"  Installed adapters (entry-point): {sorted(installed) or '(none)'}\n"
        f"  Legacy aliases:                   {sorted(_LEGACY_MODEL_ALIASES)}\n"
        f"  Power-user form:                  '<module>:<Class>'\n"
        f"  To add '{adapter}' as a ZMQ-worker adapter, run:\n"
        f"      uv pip install emboviz-{adapter}\n"
        f"      emboviz install-{adapter}\n"
        f"      emboviz-{adapter} serve &"
    )


# ────────── Diagnostic selection ─────────────────────────────────────
#
# The diagnostic suite exposes the following axes (canonical short names
# + full axis names). Both work in ``analysis.diagnostics``.

DIAGNOSTIC_SHORT_NAMES: dict[str, str] = {
    "memorization":    "vision.memorization",
    "modality":        "input.modality_dropout",
    "modality_dropout":"input.modality_dropout",
    "sensitivity":     "vision.scene_sensitivity",
    "scene_sensitivity":"vision.scene_sensitivity",
    "chunk":           "internal.chunk_consistency",
    "chunk_consistency":"internal.chunk_consistency",
    "attention":       "internal.attention_drift",
    "attention_drift": "internal.attention_drift",
}

ALL_DIAGNOSTICS: frozenset[str] = frozenset({
    "vision.memorization",
    "input.modality_dropout",
    "vision.scene_sensitivity",
    "internal.chunk_consistency",
    "internal.attention_drift",
})


def _canon_diagnostic(token: str) -> str:
    """Map a config diagnostic name to its canonical axis name.

    Accepts both the short name (``"memorization"``) and the full axis
    name (``"vision.memorization"``). Raises with the known list if the
    user typed something we don't recognise.
    """
    t = token.strip().lower()
    if not t:
        raise click.UsageError("empty diagnostic name in analysis.diagnostics")
    if t in ALL_DIAGNOSTICS:
        return t
    if t in DIAGNOSTIC_SHORT_NAMES:
        return DIAGNOSTIC_SHORT_NAMES[t]
    known = sorted(ALL_DIAGNOSTICS | set(DIAGNOSTIC_SHORT_NAMES))
    raise click.UsageError(
        f"Unknown diagnostic '{token}'. We only support these names: {known}"
    )


def _resolve_diagnostics(diagnostics_spec: str) -> frozenset[str]:
    """Resolve ``analysis.diagnostics`` (normalized to a comma string by
    :meth:`RunConfig.diagnostics_str`) into the enabled axis set.

    Accepts:
      • ``"all"``        — every diagnostic
      • ``"X,Y,Z"``      — an explicit include list
      • ``"all,-X"``     — all minus X
      • ``"X,Y,-Z"``     — include X and Y, then drop Z

    Unknown names raise.
    """
    enabled: set[str] = set()
    for raw in (diagnostics_spec or "all").split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "all":
            enabled |= set(ALL_DIAGNOSTICS)
        elif token.startswith("-"):
            enabled.discard(_canon_diagnostic(token[1:]))
        else:
            enabled.add(_canon_diagnostic(token))
    return frozenset(enabled)


@click.command("analyze")
@click.option("--config", "config_ref", required=True,
              help="Path to a run config YAML, or the name of a shipped "
                   "template under configs/ (e.g. 'openvla-bridge', "
                   "'pi0-libero'). The config declares the model, dataset "
                   "mapping, and analysis parameters — everything for the run.")
@click.option("--dry-run", "dry_run", is_flag=True, default=False,
              help="Print the per-frame and per-episode forward-pass estimate "
                   "without running the diagnostic suite. Use this BEFORE "
                   "committing GPU hours on a long episode.")
def analyze_cmd(config_ref: str, dry_run: bool) -> None:
    """Analyze a model on one or more episodes from a single config file.

    \b
        # Shipped template (copy + edit it for your own checkpoint/data):
        emboviz analyze --config configs/openvla-bridge.yaml

    \b
        # By template name:
        emboviz analyze --config pi0-libero

    \b
        # Size the GPU budget before a long run:
        emboviz analyze --config my-run.yaml --dry-run
    """
    from emboviz._internal.multi_episode import (
        EpisodeReport,
        aggregate_episodes,
        parse_episode_spec,
        write_aggregate_html,
        write_aggregate_markdown,
    )
    from emboviz._internal.report import write_episode_reports
    from emboviz.config import load_run_config

    cfg = load_run_config(config_ref)

    model_spec = _resolve_model_spec(cfg.model.adapter)
    model_kwargs_json = json.dumps(cfg.model.kwargs)
    dataset_spec = _MANIFEST_BUILDER
    dataset_kwargs_json = json.dumps(cfg.dataset_build_kwargs())

    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    enabled_diagnostics = _resolve_diagnostics(cfg.diagnostics_str())
    if not enabled_diagnostics:
        raise click.UsageError(
            "analysis.diagnostics resolved to an empty set; nothing to run."
        )
    click.echo(f"[analyze] config: {config_ref}")
    click.echo(f"[analyze] model:  {model_spec}  kwargs={cfg.model.kwargs}")
    click.echo(f"[analyze] dataset: format={cfg.dataset.format} path={cfg.dataset.path}")
    click.echo(f"[analyze] diagnostics enabled: {sorted(enabled_diagnostics)}")

    # Resolve --episodes. For "all" we need the dataset's episode count,
    # which means building the source (cheap — reads the schema) and asking
    # it to enumerate.
    n_available = None
    episodes_spec = cfg.analysis.episodes.strip()
    if episodes_spec == "all":
        from emboviz._internal.runner import _resolve as resolve_builder
        click.echo("[analyze] resolving dataset to count episodes for episodes='all' ...")
        ds = resolve_builder(dataset_spec, dataset_kwargs_json)
        try:
            n_available = len(ds.list_episodes())
        except Exception as e:
            raise click.UsageError(
                f"episodes='all' requires dataset.list_episodes() to work: {e}"
            )
    episode_indices = parse_episode_spec(episodes_spec, n_available)
    if not episode_indices:
        raise click.UsageError("analysis.episodes resolved to an empty list")
    click.echo(f"[analyze] analyzing {len(episode_indices)} episode(s): "
               f"{episode_indices[:10]}{' ...' if len(episode_indices) > 10 else ''}")

    # Per-episode loop. run_story re-loads model + dataset internally per
    # call (the dataset source caches its handle, the model worker stays
    # warm across calls in the same session).
    episode_reports: list[EpisodeReport] = []
    for ep_idx in episode_indices:
        ep_dir = out / f"episode_{ep_idx:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"\n[analyze] ===== episode {ep_idx} =====")

        args = argparse.Namespace(
            story_id=f"{cfg.model.adapter}:{cfg.dataset.format}:ep{ep_idx}",
            model_builder=model_spec,
            model_kwargs_json=model_kwargs_json,
            dataset_builder=dataset_spec,
            dataset_kwargs_json=dataset_kwargs_json,
            episode_idx=ep_idx,
            frame_start=cfg.analysis.frame_start,
            n_frames=cfg.analysis.n_frames,
            frame_stride=cfg.analysis.frame_stride,
            sensitivity_grid_side=cfg.analysis.sensitivity_grid_side,
            out_dir=str(ep_dir),
            modality_pool_size=cfg.analysis.modality_pool_size,
            modality_k_samples=cfg.analysis.modality_k_samples,
            modality_pool_seed=cfg.analysis.modality_pool_seed,
            modality_pool_cache_dir=cfg.analysis.modality_pool_cache_dir,
            target_text=cfg.analysis.mask_query,
            target_annotations=cfg.analysis.target_annotations or "",
            detector=cfg.analysis.detector,
            enabled_diagnostics=enabled_diagnostics,
            show_imitation=cfg.analysis.show_imitation,
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
            episode_reports.append(EpisodeReport(
                episode_idx=ep_idx,
                out_dir=ep_dir,
                summary_path=summary_path,
                rollout_rrd_path=rrd_path if rrd_path.exists() else None,
            ))

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

    # A --dry-run produces no summary.json (run_story returns early); that's
    # expected, not a failure.
    if dry_run:
        click.echo("\n[analyze] --dry-run: cost estimate(s) printed above; "
                   "no diagnostics run, nothing aggregated.")
        return

    # Aggregate cross-episode patterns.
    if not episode_reports:
        click.echo("\n[analyze] no episodes produced summary.json — nothing to aggregate.", err=True)
        sys.exit(1)
    click.echo(f"\n[analyze] aggregating across {len(episode_reports)} episode(s) ...")
    aggregate = aggregate_episodes(episode_reports)
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str))
    md = write_aggregate_markdown(aggregate, model_id=cfg.model.adapter, out_path=out / "aggregate.md")
    html = write_aggregate_html(
        aggregate, model_id=cfg.model.adapter, episodes=episode_reports,
        out_path=out / "aggregate.html",
    )
    click.echo(f"[analyze] wrote {out / 'aggregate.json'}")
    click.echo(f"[analyze] wrote {md}")
    if html is not None:
        click.echo(f"[analyze] wrote {html}")
    else:
        click.echo("[analyze] (skipped aggregate.html — install 'emboviz[viz]' for HTML reports)")
    click.echo(f"[analyze] per-episode reports in {out}/episode_*/")
