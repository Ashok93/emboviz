"""Run the full profile + coverage analysis + verdict card on one scene.

This is the 'killer demo' entry point — what a robotics engineer runs on a
failure rollout.

Usage:
    uv run python -m policylens.cli.run_battery \
        --model openvla-7b --scene bridge:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from policylens.cli._loaders import load_model, load_scene
from policylens.core.results import Severity
from policylens.coverage.gap_detector import detect_gaps
from policylens.coverage.text_analyzer import analyze_dataset_coverage
from policylens.reports.failure_matrix import render_failure_matrix
from policylens.reports.json_export import export_json
from policylens.reports.markdown import render_markdown_report
from policylens.reports.verdict_card import render_verdict_card
from policylens.suites.full_profile import build_full_profile
from policylens.taxonomy.object_categories import NOUN_CATEGORIES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", required=True, help="e.g. bridge:0")
    parser.add_argument("--outdir", default="outputs/battery")
    parser.add_argument("--dataset", default="bridge",
                        help="Dataset for coverage analysis (currently only 'bridge').")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[battery] model: {args.model}", flush=True)
    model = load_model(args.model)

    print(f"[battery] scene: {args.scene}", flush=True)
    scene = load_scene(args.scene)

    print("[battery] running full profile...", flush=True)
    suite = build_full_profile()
    result = suite.run(model, scene)

    # Print quick console summary
    print("\n[battery] === results summary ===")
    for name, r in result.results.items():
        marker = {
            Severity.CRITICAL: "🟥",
            Severity.MODERATE: "🟧",
            Severity.PASS: "🟩",
            Severity.INFO: "🟦",
            Severity.UNKNOWN: "⬜",
        }.get(r.severity, "?")
        print(f"  {marker} {r.axis:40s}  score={r.scalar_score:.3f}  ({r.severity.value})")

    # Coverage analysis (Bridge only for now)
    if args.dataset == "bridge":
        print("\n[battery] running coverage analysis on BridgeV2...", flush=True)
        from policylens.datasets.lerobot_bridge import BridgeEpisodeSource, DATASET_REPO
        src = BridgeEpisodeSource()
        instructions = src.all_instructions()
        snapshot = analyze_dataset_coverage(instructions, dataset_name=DATASET_REPO)
        # Map failing language axes to coverage categories.
        failing_axes = [
            {"axis": "noun_swap", "category": c} for c in NOUN_CATEGORIES
        ]
        cov_report = detect_gaps(snapshot, failing_axes)
        print(f"        {len(cov_report.gaps)} category-level gaps identified")

    print("\n[battery] rendering...", flush=True)
    render_failure_matrix(result, outdir / "failure_matrix.png")
    render_markdown_report(result, outdir / "REPORT.md")
    render_verdict_card(result, scene, outdir / "verdict_card.png")
    export_json(result, outdir / "results.json")

    print(f"\n[battery] done → {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
