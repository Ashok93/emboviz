"""Integration test #4 — GR00T-N1.7-3B + DROID-style as if I'm a real user.

User story:
  I'm at an industrial automation startup. We're evaluating NVIDIA's
  GR00T-N1.7-3B as a baseline for our warehouse pick-and-place tasks.
  Our setup is closest to OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT.
  I want to know: does it actually follow language? Does it use the
  proprioception we feed it? Where are its blind spots?
"""
from pathlib import Path
import sys, time, traceback
import numpy as np

OUT = Path("/root/itest/gr00t")
OUT.mkdir(parents=True, exist_ok=True)

print("[user] Real user: I'm evaluating GR00T-N1.7-3B as a baseline for our warehouse pick-and-place.")

# Step 1
print("\n[user] Step 1: load GR00T (cached, ~30s)...", flush=True)
t0 = time.time()
from emboviz.models.gr00t import Gr00tAdapter
m = Gr00tAdapter(
    model_path="nvidia/GR00T-N1.7-3B",
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    device="cuda:0",
)
print(f"[user] Loaded in {time.time()-t0:.1f}s")
print(f"[user] video_keys={m._video_keys}  state_keys={m._state_keys}  action_keys={m._action_keys}")

# Step 2: build DROID-style scene
print("\n[user] Step 2: build a Scene with my DROID-style data (Bridge image + identity eef_9d)...", flush=True)
from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
from emboviz.core.types import Scene, Observations
from emboviz.core.observations import RGBImage, Proprioception, GripperState
bridge = BridgeEpisodeSource().load_episode("0")[0]
img = bridge.primary_image_data
# Identity rotation in 6D continuous + position
identity_eef_9d = np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
obs = Observations(
    images={"primary": RGBImage(data=img, camera_id="primary")},
    state=Proprioception(values=identity_eef_9d, convention="ee_pose"),
    gripper=GripperState(value=0.5, kind="parallel_jaw", units="unit"),
)
scene = Scene(observations=obs, instruction=bridge.instruction)
print(f"[user] Scene: instruction=\"{scene.instruction}\"")

# Step 3: focused language + vision suite
print("\n[user] Step 3: focused suite — language axes + occlusion + lighting...", flush=True)
from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.diagnostics.sweep import SweepDiagnostic
from emboviz.perturb.instruction import (
    NounSwapPerturber, EmptyInstructionPerturber, OODTaskPerturber, NegationPerturber,
)
from emboviz.perturb.image import OcclusionPerturber, LightingShiftPerturber, GaussianNoisePerturber
from emboviz.suites.base import Suite

suite = Suite(
    name="gr00t_focused",
    description="GR00T-N1.7 — language + vision robustness",
    diagnostics=[
        CounterfactualDiagnostic(NounSwapPerturber()),
        CounterfactualDiagnostic(EmptyInstructionPerturber()),
        CounterfactualDiagnostic(OODTaskPerturber()),
        CounterfactualDiagnostic(NegationPerturber()),
        SweepDiagnostic(OcclusionPerturber(), level_param_key="coverage"),
        CounterfactualDiagnostic(LightingShiftPerturber()),
        CounterfactualDiagnostic(GaussianNoisePerturber()),
    ],
)
t0 = time.time()
result = suite.run(m, scene)
print(f"[user] Suite ran in {time.time()-t0:.1f}s (note: GR00T is slower per-prediction than smaller models)")

print(f"\n{'='*80}\nGR00T-N1.7 FINDINGS\n{'='*80}")
for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
    print(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}")
print(f"{'='*80}")

critical = [r for r in result.results.values() if r.severity.value == "critical"]
print(f"\n[user] {len(critical)} CRITICAL findings.")
for r in critical:
    print(f"\n  🟥 {r.axis}: {r.explanation}")

from emboviz.exporters import render_scorecard, render_detail_pages
render_scorecard(result, OUT / "scorecard.png",
                 title="Emboviz — GR00T-N1.7-3B on DROID-style",
                 subtitle=f'instruction: "{scene.instruction}"')
render_detail_pages(result, OUT / "details")

with open(OUT / "summary.txt", "w") as f:
    f.write(f"GR00T-N1.7-3B (OXE_DROID embodiment)\n")
    f.write(f"Instruction: {scene.instruction}\n\n")
    for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
        f.write(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}\n")
    f.write("\n=== Per-finding ===\n")
    for r in result.results.values():
        f.write(f"\n[{r.severity.value.upper()}] {r.axis}\n  {r.explanation}\n")

print(f"\n[user] Outputs at {OUT}. GR00T_INTEGRATION_OK")
