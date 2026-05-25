"""Run a named Suite on (model, scene) and render outputs.

Usage:
    uv run python -m policylens.cli.run_suite \
        --model openvla-7b --suite language_grounding --scene bridge:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from policylens.cli._loaders import build_suite, load_model, load_scene
from policylens.reports.failure_matrix import render_failure_matrix
from policylens.reports.json_export import export_json
from policylens.reports.markdown import render_markdown_report
from policylens.reports.verdict_card import render_verdict_card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Registered model name (e.g. openvla-7b, mock)")
    parser.add_argument("--suite", required=True,
                        choices=["language_grounding", "visual_robustness",
                                 "full_profile", "quick_smoke"])
    parser.add_argument("--scene", required=True, help="Scene spec (e.g. bridge:0)")
    parser.add_argument("--outdir", default="outputs/suite_run")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cli] loading model {args.model!r}...", flush=True)
    model = load_model(args.model)
    print(f"      capabilities: {model.capabilities}")

    print(f"[cli] loading scene {args.scene!r}...", flush=True)
    scene = load_scene(args.scene)
    print(f"      instruction: \"{scene.instruction}\"")

    print(f"[cli] running suite {args.suite!r}...", flush=True)
    suite = build_suite(args.suite)
    suite_result = suite.run(model, scene)

    print(f"[cli] rendering outputs to {outdir}...", flush=True)
    render_failure_matrix(suite_result, outdir / "failure_matrix.png")
    render_markdown_report(suite_result, outdir / "REPORT.md")
    render_verdict_card(suite_result, scene, outdir / "verdict_card.png")
    export_json(suite_result, outdir / "results.json")

    print("[cli] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
