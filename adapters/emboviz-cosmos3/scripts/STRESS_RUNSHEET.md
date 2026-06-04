# Critical-moment stress test — GPU runsheet

Roll a short world-model clip at each decisive instant of a recorded episode
(grasp / release / settle) and compare it to what really happened. Everything
below the world model is built and tested offline; the only thing that needs the
GPU is the running Cosmos server.

## 0. Prerequisites
- A Cosmos 3 vLLM-Omni server reachable over HTTP (see `POD_RUNSHEET.md` for the
  pod: CUDA-13 GPU, `vllm/vllm-omni:cosmos3`, port 8000, `HF_TOKEN`, guardrail
  accepted). Note its proxy URL, e.g. `https://<podid>-8000.proxy.runpod.net`.
- The DROID config: `configs/droid.yaml` (already maps the cartesian state +
  gripper the keyframe detector and encoder need).

## 1. First run — recorded actions (the faithfulness receipt, zero policy risk)
This needs **no policy**. It seeds a clip at every keyframe, feeds the episode's
**real** actions, and compares the rendered clip to the real footage. Where
Cosmos is faithful, predicted ≈ real (low divergence) — that's the receipt that
the engine reproduces reality before we trust anything else.

```bash
uv run emboviz stop   # clear any warm worker holding different kwargs
uv run python -m emboviz.world_models.stress_cli \
    --config configs/droid.yaml --episode 0 \
    --world-model cosmos3 --server-url https://<podid>-8000.proxy.runpod.net \
    --domain droid_lerobot --action-dim 10 \
    --source recorded --n-actions 16 --lead-s 0.5 \
    --out outputs/cosmos_stress
```

Output (written incrementally, one clip at a time):
`outputs/cosmos_stress/clip_<frame>_<kind>/frames/*.png` (predicted | real
side-by-side) + `divergence.json`, and a top-level `summary.json`.

## 2. Policy-driven — the user's policy decides at each critical moment
The user's policy drives; Cosmos renders the consequence. Needs the policy
adapter and an **explicitly declared** action convention (never inferred — see
`emboviz_cosmos3/bridge.py`):
- `absolute_xyz_euler` — chunk rows are absolute next EE poses `[x,y,z,r,p,y,grip]`.
- `delta_xyz_euler_base` — chunk rows are base-frame deltas `[dx,dy,dz,dr,dp,dy,grip]`.

```bash
uv run python -m emboviz.world_models.stress_cli \
    --config configs/droid.yaml --episode 0 \
    --world-model cosmos3 --server-url https://<podid>-8000.proxy.runpod.net \
    --domain droid_lerobot --action-dim 10 \
    --source policy --policy-adapter pi0 --action-convention delta_xyz_euler_base \
    --n-actions 16 --lead-s 0.5 \
    --out outputs/cosmos_stress_policy
```

**Confirm the convention before trusting the policy clips.** The bridge math is
verified (its `absolute_xyz_euler` path reproduces the gold recorded encoder
bit-for-bit — `tests/test_bridge.py`), but *which* convention a given checkpoint
emits is the one thing only the real policy can confirm. Quick check: run step 2
with `--source recorded` semantics first (step 1) — if those clips track reality
and the policy clips look wildly off everywhere, the convention is wrong, not the
world model.

## What "works" looks like
- The grasp/miss reads clearly in `clip_<grasp-frame>_gripper_change/frames/`.
- Recorded clips have low `divergence` for the first frames (Cosmos faithful on
  real actions), rising as the autoregressive horizon is exceeded — that bound is
  the honest faithful window; keep `--n-actions` near one chunk (16) for the
  trustworthy part.

## Knobs
- `--n-actions` rollout length per clip (default 16 = one chunk ≈ the faithful
  window; larger drifts).
- `--lead-s` seconds before each keyframe to seed (default 0.5 — the pre-grasp
  approach, where a perturbation actually changes what the policy should do).
- `--metric pixel_l2 | ssim` divergence metric.

---

# 3. The closed-loop simulator — run the policy *inside* Cosmos (the product)

This is the real thing: at each keyframe, **perturb** the seed scene with an
editing instruction, then **fly the user's policy inside Cosmos** step by step
(policy acts → Cosmos renders → policy reacts → …) and ask the reasoner what
happened. The simulator is Cosmos; the policy is under test. Output is the dream
video + a verdict — **no divergence** (a perturbed scene never happened, so there
is nothing real to compare against).

Everything is config-driven via `analysis.cosmos_stress` (see `configs/droid.yaml`):
set `server_url`, `policy_adapter`, `action_convention`, `camera_map`, the
`perturbations` list, and (optionally) `reasoner_url`. Then:

```bash
uv run emboviz stop
uv run python -m emboviz.world_models.dream_cli \
    --config configs/droid.yaml --episode 0 \
    --out outputs/cosmos_dream
```

Output per clip — `outputs/cosmos_dream/clip_<frame>_<kind>__<perturbation>/`:
- `seed.png`      — the perturbed conditioning frame (cup→duck etc.),
- `step_NN.mp4`   — each closed-loop turn, saved incrementally,
- `dream.mp4`     — the full dream (the shareable clip),
- `verdict.json`  — Cosmos Reason's answer ("grasped / missed and how"),
plus a top-level `summary.json`.

### Two confirm-on-server items (only the live pod settles these)
1. **The image-edit endpoint.** The edit call is isolated in
   `emboviz_cosmos3/perturb.py` behind `endpoint_path` (default the documented
   vLLM-Omni image-edit chat endpoint). If Cosmos 3 registers `image2image`
   elsewhere, it's a one-line change — confirm with `curl {server}/v1/models`.
2. **The policy's `action_convention`.** The bridge math is verified
   (`absolute_xyz_euler` reproduces the gold encoder bit-for-bit), but *which*
   convention a checkpoint emits (`absolute_xyz_euler` vs `delta_xyz_euler_base`)
   is confirmed against the real policy. If the dream looks like the arm flies
   off immediately, the convention is wrong — flip it in the config.

### Honest limit
The dream drifts after the first turn or two (each turn builds on the last dream),
so keep `n_loop_steps` at 2–3. It's strongest in the first ~second around the
grasp — which is exactly where the failure shows.
