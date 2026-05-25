"""Integration test #2 — π0 + ALOHA as if I'm a real user.

User story:
  I'm at a startup doing bimanual manipulation with Physical Intelligence's
  π0 model. We use the pi0_aloha_sim variant for desktop testing before
  deploying to our real ALOHA. Lately our model produces oddly similar
  action chunks regardless of the task description we feed it. I want
  to know if π0 is actually listening to language at all on our setup.

What I do:
  1. Install path: openpi's own venv (we already followed their README)
  2. Build a Scene matching ALOHA's observation contract
  3. Run a focused diagnostic set on language axes (since that's my hypothesis)
  4. Look at the verdict
"""
from pathlib import Path
import sys, time, traceback
import numpy as np
from PIL import Image

OUT = Path("/root/itest/pi0")
OUT.mkdir(parents=True, exist_ok=True)

print("[user] As a real user, my goal: confirm whether π0 is grounding language on my ALOHA setup.")
print(f"[user] Output dir: {OUT}")

# Step 1: load my model
print("\n[user] Step 1: load π0 (pi0_aloha_sim, cached from earlier session)...", flush=True)
t0 = time.time()
from emboviz.models.pi0 import Pi0Adapter
m = Pi0Adapter(config_name="pi0_aloha_sim")
print(f"[user] Loaded in {time.time()-t0:.1f}s", flush=True)

# Step 2: build a realistic ALOHA scene (14-DOF state, 4 cams)
print("\n[user] Step 2: build a Scene with my ALOHA-shaped data...", flush=True)
from emboviz.core.types import Scene, Observations
from emboviz.core.observations import RGBImage, Proprioception

img = Image.fromarray((np.random.rand(640, 480, 3) * 255).astype(np.uint8))
obs = Observations(
    images={
        "primary": RGBImage(data=img, camera_id="primary"),
        "cam_high": RGBImage(data=img, camera_id="cam_high"),
        "cam_low": RGBImage(data=img, camera_id="cam_low"),
        "cam_left_wrist": RGBImage(data=img, camera_id="cam_left_wrist"),
        "cam_right_wrist": RGBImage(data=img, camera_id="cam_right_wrist"),
    },
    state=Proprioception(values=np.zeros(14, dtype=np.float32), convention="joint_angles"),
)
scene = Scene(observations=obs, instruction="grab the red cube and place it in the bin")
print(f"[user] Scene: instruction=\"{scene.instruction}\", cameras={list(scene.observations.images.keys())}")

# Step 3: focused language-axis diagnostic suite
print("\n[user] Step 3: run language-axis diagnostics (noun_swap, color_swap, empty, refusal, OOD)...", flush=True)
from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.perturb.instruction import (
    NounSwapPerturber, ColorSwapPerturber, EmptyInstructionPerturber,
    RefusalPerturber, OODTaskPerturber, NegationPerturber,
)
from emboviz.suites.base import Suite

suite = Suite(
    name="pi0_language_focus",
    description="Does π0 ground on language?",
    diagnostics=[
        CounterfactualDiagnostic(NounSwapPerturber()),
        CounterfactualDiagnostic(ColorSwapPerturber()),
        CounterfactualDiagnostic(EmptyInstructionPerturber()),
        CounterfactualDiagnostic(RefusalPerturber()),
        CounterfactualDiagnostic(OODTaskPerturber()),
        CounterfactualDiagnostic(NegationPerturber()),
    ],
)
t0 = time.time()
result = suite.run(m, scene)
print(f"[user] Suite ran in {time.time()-t0:.1f}s")

# Step 4: read the results
print(f"\n{'='*80}\nπ0 LANGUAGE DIAGNOSTIC RESULTS\n{'='*80}")
for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
    print(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}")
print(f"{'='*80}")

critical = [r for r in result.results.values() if r.severity.value == "critical"]
if critical:
    print(f"\n[user] {len(critical)} CRITICAL findings:")
    for r in critical:
        print(f"\n  🟥 {r.axis}")
        print(f"     {r.explanation}")
else:
    print("\n[user] No critical findings — π0 might be grounding more than I thought.")

# Outputs
from emboviz.exporters import render_scorecard, render_detail_pages
render_scorecard(result, OUT / "scorecard.png",
                 title="Emboviz — π0 (pi0_aloha_sim) language profile",
                 subtitle=f'instruction: "{scene.instruction}"')
render_detail_pages(result, OUT / "details")

with open(OUT / "summary.txt", "w") as f:
    f.write(f"π0 (pi0_aloha_sim) — language-axis diagnostics\n")
    f.write(f"Instruction: {scene.instruction}\n\n")
    for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
        f.write(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}\n")
    f.write("\n=== Per-finding ===\n")
    for r in result.results.values():
        f.write(f"\n[{r.severity.value.upper()}] {r.axis}\n  {r.explanation}\n")

print(f"\n[user] Outputs: {OUT}/scorecard.png + {OUT}/details/ + {OUT}/summary.txt")
print("[user] PI0_INTEGRATION_OK")
