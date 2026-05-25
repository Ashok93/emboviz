"""Compare two models on the same Suite and Scene; render the diff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emboviz.cli._loaders import build_suite, load_model, load_scene
from emboviz.reports.comparison import render_comparison
from emboviz.reports.json_export import export_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--suite", required=True,
                        choices=["language_grounding", "visual_robustness",
                                 "full_profile", "quick_smoke"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--outdir", default="outputs/comparison")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    suite = build_suite(args.suite)
    scene = load_scene(args.scene)

    print(f"[cmp] running on {args.model_a}...", flush=True)
    a = suite.run(load_model(args.model_a), scene)
    export_json(a, outdir / f"{args.model_a}.json")

    print(f"[cmp] running on {args.model_b}...", flush=True)
    b = suite.run(load_model(args.model_b), scene)
    export_json(b, outdir / f"{args.model_b}.json")

    print("[cmp] rendering comparison...", flush=True)
    render_comparison(a, b, outdir / "comparison.png")
    print("[cmp] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
