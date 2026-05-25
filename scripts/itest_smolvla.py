"""Integration test #3 — SmolVLA + real ALOHA dataset as if I'm a real user.

User story:
  I'm at a small lab. We can't afford H100s, so we use HuggingFace's
  SmolVLA (450M params) on consumer GPUs. We're evaluating it on the
  lerobot/aloha_sim_transfer_cube_human dataset. I want to know if
  this model actually uses the cameras or just memorizes — and whether
  it grounds language at all on our task.
"""
from pathlib import Path
import sys, time, traceback
import numpy as np

OUT = Path("/root/itest/smolvla")
OUT.mkdir(parents=True, exist_ok=True)

print("[user] Real user: I'm evaluating SmolVLA on my ALOHA cube-transfer task.")

# Step 1
print("\n[user] Step 1: load SmolVLA via LeRobotPolicyAdapter...", flush=True)
t0 = time.time()
from emboviz.models.lerobot_policy import LeRobotPolicyAdapter
m = LeRobotPolicyAdapter(repo_id="lerobot/smolvla_base", device="cuda")
print(f"[user] Loaded in {time.time()-t0:.1f}s  needs_lang={m._needs_language} needs_state={m._needs_state}")

# Step 2: load a real frame from my dataset
print("\n[user] Step 2: load a real frame from lerobot/aloha_sim_transfer_cube_human...", flush=True)
from emboviz.datasets.lerobot import LeRobotEpisodeSource
from emboviz.core.profile import RobotProfile, CameraSpec, StateSpec, GripperSpec, ActionSpec
profile = RobotProfile(
    name="aloha_sim",
    cameras=[CameraSpec("primary")],
    state=StateSpec(dim=14, convention="joint_angles"),
    gripper=GripperSpec(kind="parallel_jaw", units="unit"),
    action=ActionSpec(dim=14),
)
src = LeRobotEpisodeSource(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    profile=profile,
    image_keys={"primary": "observation.images.top"},
    state_key="observation.state",
    action_key="action",
    n_episodes=50,
)
scene = src.load_episode("0")[0]
if not scene.instruction:
    scene = scene.with_instruction("pick up the red cube and transfer it to the other arm")
print(f"[user] Scene: instruction=\"{scene.instruction}\"")

# Step 3: focused suite — language + vision robustness
print("\n[user] Step 3: focused suite (language + vision robustness)...", flush=True)
from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.diagnostics.sweep import SweepDiagnostic
from emboviz.diagnostics.memorization import MemorizationDiagnostic
from emboviz.perturb.instruction import (
    NounSwapPerturber, EmptyInstructionPerturber, OODTaskPerturber,
)
from emboviz.perturb.image import OcclusionPerturber, LightingShiftPerturber, GaussianNoisePerturber
from emboviz.suites.base import Suite

suite = Suite(
    name="smolvla_focused",
    description="SmolVLA on ALOHA — language + vision",
    diagnostics=[
        CounterfactualDiagnostic(NounSwapPerturber()),
        CounterfactualDiagnostic(EmptyInstructionPerturber()),
        CounterfactualDiagnostic(OODTaskPerturber()),
        SweepDiagnostic(OcclusionPerturber(), level_param_key="coverage"),
        CounterfactualDiagnostic(LightingShiftPerturber()),
        CounterfactualDiagnostic(GaussianNoisePerturber()),
        MemorizationDiagnostic(),
    ],
)
t0 = time.time()
result = suite.run(m, scene)
print(f"[user] Suite ran in {time.time()-t0:.1f}s")

print(f"\n{'='*80}\nSMOLVLA FINDINGS\n{'='*80}")
for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
    print(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}")
print(f"{'='*80}")

critical = [r for r in result.results.values() if r.severity.value == "critical"]
moderate = [r for r in result.results.values() if r.severity.value == "moderate"]
print(f"\n[user] {len(critical)} CRITICAL, {len(moderate)} MODERATE.")
for r in critical:
    print(f"\n  🟥 {r.axis}: {r.explanation}")

from emboviz.exporters import render_scorecard, render_detail_pages
render_scorecard(result, OUT / "scorecard.png",
                 title="Emboviz — SmolVLA on ALOHA",
                 subtitle=f'instruction: "{scene.instruction}"')
render_detail_pages(result, OUT / "details")

with open(OUT / "summary.txt", "w") as f:
    f.write(f"SmolVLA on lerobot/aloha_sim_transfer_cube_human\n")
    f.write(f"Instruction: {scene.instruction}\n\n")
    for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
        f.write(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}\n")
    f.write("\n=== Per-finding ===\n")
    for r in result.results.values():
        f.write(f"\n[{r.severity.value.upper()}] {r.axis}\n  {r.explanation}\n")

print(f"\n[user] Outputs at {OUT}. SMOLVLA_INTEGRATION_OK")
