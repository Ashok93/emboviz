"""Generic per-(model, dataset) sanity probe.

Confirms whether a model's predicted action and the dataset's recorded
expert action live in the same space. The point is to surface action-
space convention mismatches BEFORE we publish diagnostic numbers that
silently fold the mismatch into a meaningless L2.

Run with:
    PYTHONPATH=/root/emboviz $VENV/bin/python scripts/probe_convention.py \
        --model emboviz.models.X:Y[:arg] [--model-kwargs-json '{...}'] \
        --dataset emboviz.datasets.X:Y[:arg] [--dataset-kwargs-json '{...}'] \
        --episode 0 --frames 0,27,55,82,109

Prints per-dim predicted vs expert for sampled frames, plus a 5x identical-
input variance probe (the model's actual noise floor), plus cross-frame
variability ratio. Convention mismatches surface as a single dim with
huge per-dim Δ while others agree.
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np


def _resolve(spec: str, kwargs_json: str = ""):
    parts = spec.split(":")
    module = importlib.import_module(parts[0])
    obj = getattr(module, parts[1])
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if len(parts) == 2:
        return obj(**kwargs)
    intermediate = obj(parts[2])
    if isinstance(intermediate, type):
        return intermediate(**kwargs)
    return intermediate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-kwargs-json", default="")
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-kwargs-json", default="")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", default="0,27,55,82,109",
                   help="comma-separated frame indices to sample")
    p.add_argument("--n-noise-reps", type=int, default=5)
    args = p.parse_args()

    print(f"[probe] loading dataset {args.dataset}", flush=True)
    ds = _resolve(args.dataset, args.dataset_kwargs_json)
    traj = ds.load_trajectory(args.episode)
    print(f"  frames: {len(traj.frames)}", flush=True)
    print(f"  instruction: {traj.frames[0].instruction!r}", flush=True)
    print(f"  cameras: {sorted(traj.frames[0].observations.images)}", flush=True)

    print(f"\n[probe] loading model {args.model}", flush=True)
    m = _resolve(args.model, args.model_kwargs_json)
    print(f"  model_id: {m.model_id}", flush=True)
    print(f"  required_cameras: {sorted(m.required_inputs.cameras)}", flush=True)

    # Action dim names (from profile if available)
    profile = traj.frames[0].profile
    dim_names = (
        list(profile.action.dim_names)
        if profile is not None and profile.action is not None
           and profile.action.dim_names is not None
        else []
    )
    print(f"  dataset action dim_names: {dim_names}", flush=True)

    # Sample frames
    requested = [int(x) for x in args.frames.split(",") if x.strip()]
    sample_idxs = [i for i in requested if 0 <= i < len(traj.frames)]
    if not sample_idxs:
        sample_idxs = [0, len(traj.frames) // 2, len(traj.frames) - 1]
    print(f"\n[probe] sampling frames {sample_idxs}", flush=True)

    n_dims = None
    rows: list[dict] = []
    for fi in sample_idxs:
        scene = traj.frames[fi]
        pred = m.predict(scene).action
        exp = np.asarray(scene.metadata.get("expert_action", [0] * len(pred)),
                         dtype=np.float32)
        n = min(len(pred), len(exp))
        n_dims = n
        rows.append({
            "frame": fi,
            "pred":  pred[:n],
            "exp":   exp[:n],
            "dist":  float(np.linalg.norm(pred[:n] - exp[:n])),
        })

    print(f"\nframe   {'predicted':<55s}  {'expert':<55s}  L2-dist")
    print("-" * 130)
    for r in rows:
        ps = np.array2string(r["pred"], precision=3, suppress_small=True)
        es = np.array2string(r["exp"], precision=3, suppress_small=True)
        print(f"{r['frame']:>5}   {ps:<55s}  {es:<55s}  {r['dist']:.3f}")

    # Per-dim mean |Δ| across sampled frames
    diffs = np.stack([r["pred"] - r["exp"] for r in rows])
    abs_diffs = np.abs(diffs)
    per_dim_mean = abs_diffs.mean(axis=0)
    per_dim_max = abs_diffs.max(axis=0)
    dim_label = (dim_names[:n_dims] if dim_names and len(dim_names) >= n_dims
                 else [f"d{i}" for i in range(n_dims)])
    print(f"\nper-dim |Δ| across sampled frames:")
    print(f"  {'dim':<10s}  mean   max")
    for _nm, _mean_v, _max_v in zip(dim_label, per_dim_mean, per_dim_max):
        print(f"  {_nm:<10s}  {_mean_v:.3f}  {_max_v:.3f}")
    # Convention-mismatch detector — use PER-DIM MEDIAN |Δ| across frames,
    # not mean. A single bad-prediction frame can spike a dim's mean |Δ|
    # without indicating a convention issue. A median that's systematically
    # high means EVERY frame is off — that's the real convention smell.
    per_dim_median = np.median(np.abs(diffs), axis=0)
    overall_median = float(np.median(per_dim_median))
    suspects = [
        (nm, m, mn) for nm, m, mn in zip(dim_label, per_dim_mean, per_dim_median)
        if overall_median > 1e-6 and mn >= 3 * overall_median
    ]
    print(f"\nper-dim MEDIAN |Δ| (more robust to outlier frames):")
    for _nm, _med in zip(dim_label, per_dim_median):
        print(f"  {_nm:<10s}  {_med:.3f}")
    if suspects:
        print(f"\n⚠️ likely convention mismatch (median|Δ| >= 3× overall median {overall_median:.3f}):")
        for nm, mean_v, med_v in suspects:
            print(f"    {nm}: median={med_v:.3f}  mean={mean_v:.3f}")
    else:
        print(f"\n✓ no systematic per-dim mismatch (all dim medians within "
              f"3× overall median {overall_median:.3f}) — occasional high "
              "mean|Δ| reflects model errors on specific frames, not "
              "convention issues")

    # Noise floor probe
    print(f"\n[probe] {args.n_noise_reps}x identical-input variance on frame 0:")
    preds = [m.predict(traj.frames[0]).action for _ in range(args.n_noise_reps)]
    deltas = [float(np.linalg.norm(preds[i] - preds[j]))
              for i in range(len(preds)) for j in range(i + 1, len(preds))]
    print(f"  pairwise L2 = {[round(d, 3) for d in deltas]}")
    nf = float(np.mean(deltas)) if deltas else 0.0
    print(f"  mean noise floor (raw L2): {nf:.3f}")
    print(f"  max pairwise: {max(deltas):.3f}")

    # Cross-frame variability
    all_preds = np.stack([m.predict(s).action for s in traj.frames[: min(8, len(traj.frames))]])
    cross = [float(np.linalg.norm(all_preds[i] - all_preds[j]))
             for i in range(len(all_preds)) for j in range(i + 1, len(all_preds))]
    cf = float(np.mean(cross)) if cross else 0.0
    print(f"\n[probe] cross-frame mean L2 (first 8 frames): {cf:.3f}")
    ratio = cf / max(nf, 1e-9)
    print(f"  cross-frame / noise-floor ratio = {ratio:.2f}")
    if ratio < 1.5:
        print(f"  ⚠️ predictions barely depend on frame — investigate input format")
    else:
        print(f"  ✓ predictions vary with frame content")


if __name__ == "__main__":
    main()
