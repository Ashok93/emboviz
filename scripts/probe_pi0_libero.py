"""Honest sanity probe for π0 on LIBERO-spatial.

Dumps the actual predicted vs expert action vectors side-by-side per frame
so we can tell whether π0 is responding to input + whether the action
spaces are aligned. No diagnostics, no aggregates — just the raw numbers.

Run inside the openpi venv with emboviz installed:
    cd /root/openpi && .venv/bin/python /root/emboviz/scripts/probe_pi0_libero.py
"""
from __future__ import annotations

import numpy as np

from emboviz.datasets.lerobot_libero import LiberoSpatialSource
from emboviz.models.pi0 import Pi0Adapter

print("[probe] loading LIBERO-spatial ep0...", flush=True)
src = LiberoSpatialSource()
traj = src.load_trajectory(0)
print(f"  frames: {len(traj.frames)}", flush=True)
print(f"  instruction: {traj.frames[0].instruction!r}", flush=True)
print(f"  cameras: {sorted(traj.frames[0].observations.images)}", flush=True)

print("\n[probe] loading π0 (config=pi0_libero)...", flush=True)
m = Pi0Adapter(config_name="pi0_libero")
print(f"  model_id: {m.model_id}  required: {m.required_inputs.cameras}", flush=True)

# Pick 5 frames spread across the trajectory
sample_indices = [0, len(traj.frames) // 4, len(traj.frames) // 2,
                  3 * len(traj.frames) // 4, len(traj.frames) - 1]
print(f"\n[probe] sampling frames {sample_indices}\n", flush=True)

print(f"{'frame':>6}  {'predicted action':>50s}   {'expert action':>50s}   {'L2 dist'}")
print("-" * 130)
for fi in sample_indices:
    scene = traj.frames[fi]
    pred = m.predict(scene).action
    exp = np.asarray(scene.metadata.get("expert_action", [0]*7), dtype=np.float32)
    n = min(len(pred), len(exp))
    dist = float(np.linalg.norm(pred[:n] - exp[:n]))
    print(f"{fi:>6}  {np.array2string(pred[:n], precision=3, suppress_small=True):>50s}   "
          f"{np.array2string(exp[:n], precision=3, suppress_small=True):>50s}   {dist:.3f}")

# Identical-input variance probe (5 reps on frame 0)
print(f"\n[probe] 5x identical-input variance on frame 0 — actual π0 noise floor:", flush=True)
preds = [m.predict(traj.frames[0]).action for _ in range(5)]
for i, p in enumerate(preds):
    print(f"  rep {i}: {np.array2string(p, precision=3, suppress_small=True)}  "
          f"magnitude={np.linalg.norm(p):.3f}")
deltas = [float(np.linalg.norm(preds[i] - preds[j]))
          for i in range(5) for j in range(i+1, 5)]
print(f"  pairwise L2: {[round(d,3) for d in deltas]}")
print(f"  mean = {np.mean(deltas):.3f}, max = {max(deltas):.3f}")

# Per-dim range across the trajectory — see what units each dim is in
print(f"\n[probe] per-dim min/max/std across 8 baseline predictions vs expert:")
all_preds = np.stack([m.predict(s).action for s in traj.frames[:8]])
all_expert = np.stack([np.asarray(s.metadata.get("expert_action", [0]*7), dtype=np.float32)
                       for s in traj.frames[:8]])
print(f"  pred  min: {np.array2string(all_preds.min(0), precision=3, suppress_small=True)}")
print(f"  pred  max: {np.array2string(all_preds.max(0), precision=3, suppress_small=True)}")
print(f"  pred  std: {np.array2string(all_preds.std(0), precision=3, suppress_small=True)}")
print(f"  exp   min: {np.array2string(all_expert.min(0), precision=3, suppress_small=True)}")
print(f"  exp   max: {np.array2string(all_expert.max(0), precision=3, suppress_small=True)}")
print(f"  exp   std: {np.array2string(all_expert.std(0), precision=3, suppress_small=True)}")

# Critical test: different frames, see if predictions differ MORE than noise
print(f"\n[probe] cross-frame action variability vs noise floor:")
cross_frame_deltas = [float(np.linalg.norm(all_preds[i] - all_preds[j]))
                      for i in range(8) for j in range(i+1, 8)]
print(f"  cross-frame mean L2: {np.mean(cross_frame_deltas):.3f}, max: {max(cross_frame_deltas):.3f}")
print(f"  noise-floor mean L2:  {np.mean(deltas):.3f}")
print(f"  ratio (cross-frame / noise): {np.mean(cross_frame_deltas)/max(np.mean(deltas), 1e-9):.2f}")
print(f"  → if ratio ≈ 1, predictions don't depend on the frame at all (= broken)")
print(f"  → if ratio >> 1, predictions vary with frame content (= real model behavior)")
