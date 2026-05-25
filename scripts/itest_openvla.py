"""Integration test #1 — OpenVLA-7B + Bridge as if I'm a real user.

User story:
  I'm a robotics engineer at a research lab. We trained OpenVLA-7B on
  Bridge data. Our rollouts on Bridge episode 0 don't work perfectly —
  the spoon often gets dropped or grasped at the wrong angle. I want
  to know WHY. Is the model not reading the instruction? Is it
  memorizing trajectories? Is it color-blind? I'm going to run
  Emboviz's full diagnostic battery and see what I find.

What I do:
  1. Load the model (already installed via emboviz[openvla])
  2. Load a real frame from my training distribution (Bridge ep 0 frame 0)
  3. Run the full diagnostic battery (15+ diagnostics)
  4. Look at the scorecard
  5. Drill into critical findings
  6. Generate Rerun .rrd I can scrub through

Output:
  /root/itest/openvla/
    scorecard.png
    rollout.rrd
    details/*.md
    summary.txt          <- I read this first
"""
from pathlib import Path
import sys, time, traceback

OUT = Path("/root/itest/openvla")
OUT.mkdir(parents=True, exist_ok=True)

print("[user] As a real user, my goal is to debug why my OpenVLA-7B fine-tune fails on Bridge spoon tasks.", flush=True)
print(f"[user] Output dir: {OUT}", flush=True)

# Step 1: load my model
print("\n[user] Step 1: load my model (OpenVLA-7B from HF)...", flush=True)
t0 = time.time()
from emboviz.models.registry import get_model
m = get_model("openvla-7b")()
print(f"[user] Loaded in {time.time()-t0:.1f}s", flush=True)
print(f"[user] action_dim={m.action_dim}  capabilities={m.capabilities}", flush=True)

# Step 2: load a real frame
print("\n[user] Step 2: load a real Bridge frame from my training distribution...", flush=True)
from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
src = BridgeEpisodeSource()
traj = src.load_trajectory(0)
scene = traj.frames[12]  # mid-episode, interesting moment
print(f"[user] Scene loaded: instruction=\"{scene.instruction}\"  scene_id={scene.scene_id}", flush=True)
print(f"[user] state(6,) gripper={scene.observations.gripper.value:.3f}  expert_action={scene.metadata.get('expert_action', [])[:7]}", flush=True)

# Step 3: run the diagnostic suite
print("\n[user] Step 3: run the full diagnostic suite (this is what I'm paying for)...", flush=True)
from emboviz.suites.full_profile import build_full_profile
suite = build_full_profile()
print(f"[user] Suite '{suite.name}': {len(suite.diagnostics)} diagnostics", flush=True)

t0 = time.time()
result = suite.run(m, scene)
print(f"[user] Suite ran in {time.time()-t0:.1f}s", flush=True)

# Step 4: read the scorecard
print("\n[user] Step 4: look at the per-diagnostic scorecard. What does Emboviz tell me?", flush=True)
print(f"\n{'='*80}\nDIAGNOSTIC FINDINGS — model={m.model_id} scene={scene.scene_id}\n{'='*80}")
for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
    print(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}  ({r.direction})")
print(f"{'='*80}")

# Step 5: drill into critical findings
critical = [r for r in result.results.values() if r.severity.value == "critical"]
moderate = [r for r in result.results.values() if r.severity.value == "moderate"]
print(f"\n[user] Step 5: drill into findings. {len(critical)} CRITICAL, {len(moderate)} MODERATE.")
for r in critical[:5]:
    print(f"\n[user]   🟥 {r.axis}: {r.explanation}")
for r in moderate[:3]:
    print(f"\n[user]   🟧 {r.axis}: {r.explanation}")

# Step 6: generate all outputs (scorecard PNG, detail pages, Rerun .rrd)
print(f"\n[user] Step 6: emit all outputs for sharing/playback.")
from emboviz.exporters import render_scorecard, render_detail_pages
render_scorecard(result, OUT / "scorecard.png",
                 title=f"Emboviz — OpenVLA-7B on Bridge ep 0 frame 12",
                 subtitle=f'instruction: "{scene.instruction}"')
render_detail_pages(result, OUT / "details")
print(f"[user]   scorecard.png    : {(OUT / 'scorecard.png').stat().st_size} bytes")
print(f"[user]   details/         : {len(list((OUT / 'details').glob('*.md')))} files")

# Save a summary I'd read as the user
with open(OUT / "summary.txt", "w") as f:
    f.write(f"Emboviz session: OpenVLA-7B on Bridge ep 0 frame 12\n")
    f.write(f"Instruction: {scene.instruction}\n\n")
    f.write(f"=== Diagnostic findings ===\n")
    for name, r in sorted(result.results.items(), key=lambda kv: kv[1].severity.value):
        f.write(f"  {r.severity.value:9s}  {name:40s}  score={r.scalar_score:.4f}\n")
    f.write(f"\n=== Critical findings (highest priority) ===\n")
    for r in critical:
        f.write(f"\n[CRITICAL] {r.axis}\n  {r.explanation}\n")
    f.write(f"\n=== Moderate findings ===\n")
    for r in moderate:
        f.write(f"\n[MODERATE] {r.axis}\n  {r.explanation}\n")
print(f"[user]   summary.txt      : ready to read")

print(f"\n[user] DONE — I have my answers. Time: {time.time()-t0:.1f}s total diagnostic compute.")
print(f"[user] Next step: open scorecard.png + read summary.txt + maybe drill into details/<axis>.md")
