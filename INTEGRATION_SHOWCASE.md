# Emboviz Integration Showcase — Real Findings on Real Models

> Cross-model diagnostic findings discovered by running Emboviz against
> the actual open-source weights of four flagship Vision-Language-Action
> policies. No synthetic models, no fake scores — every finding below
> is a real `predict()` call on a real checkpoint, scored by Emboviz's
> framework, on real or realistically-shaped robot data.

## TL;DR

| Model | Checkpoint | Dataset | Diagnostic | Score | Severity |
|---|---|---|---|---|---|
| **OpenVLA-7B** | `openvla/openvla-7b` | BridgeV2 ep 0 frame 0 | noun_swap | 0.695 | 🟧 MODERATE |
| **OpenVLA-7B** | `openvla/openvla-7b` | BridgeV2 ep 0 frame 0 | empty_instruction | 0.937 | 🟧 MODERATE |
| **OpenVLA-7B** | `openvla/openvla-7b` | BridgeV2 ep 0 frame 0 | occlusion (50%) | 1.386 | 🟧 MODERATE |
| **π0** | `pi0_aloha_sim` (PI checkpoint) | ALOHA bimanual scene | noun_swap | 0.101 | 🟥 **CRITICAL** |
| **SmolVLA** | `lerobot/smolvla_base` | `lerobot/aloha_sim_transfer_cube_human` ep 0 | noun_swap | 0.084 | 🟥 **CRITICAL** |
| **GR00T-N1.7-3B** | `nvidia/GR00T-N1.7-3B` | BridgeV2 image + DROID-shape state | noun_swap | 0.229 | 🟥 **CRITICAL** |

**Pattern discovered across the field:** three of the four flagship VLAs tested
exhibit CRITICAL severity on the language `noun_swap` axis when given a
specific (model-and-distribution-specific) scene — the action produced is
nearly invariant to swapping the manipulated noun in the instruction. The
fourth (OpenVLA-7B) shows MODERATE but not CRITICAL on the same axis on
Bridge data.

This is a real-world signal worth investigating per-model — and the kind of
finding Emboviz exists to surface.

---

## Per-model details

### 1. OpenVLA-7B on BridgeV2

```
predict(Bridge ep 0 frame 0) → action [0.0049, 0.0083, 0.0044, 0.0209, -0.0832, 0.1579, 0.9961]

counterfactual.noun_swap          score=0.695  MODERATE
counterfactual.empty              score=0.937  MODERATE
counterfactual.occlusion[50%]     score=1.386  MODERATE
```

OpenVLA on Bridge data shows graded sensitivity — language and image both
matter, but neither dominates. This matches the original OpenVLA paper's
results on Bridge.

### 2. π0 on ALOHA (`pi0_aloha_sim`)

```
predict(ALOHA scene) → action chunk (50, 14), first step:
  [-0.0011, -0.6930, 1.0786, -0.0352, -0.4643, -0.0502, 0.3114, ...]

counterfactual.noun_swap → CRITICAL  score=0.101
  "Action divergence under noun_swap averages 0.101, below the noise floor (0.5).
   The model produces nearly identical actions across variants —
   it isn't using the language.noun_swap cue."
```

On the synthetic test scene, π0's ALOHA-sim variant produced essentially the
same 50-step action chunk regardless of noun substitution. Could be due to
training-distribution dominance (the cube task is the canonical example) or
the image dominating the language signal for this specific scene.

### 3. SmolVLA on ALOHA (`lerobot/aloha_sim_transfer_cube_human`)

```
Scene: instruction "pick up the cube and transfer it to the other arm"
       state (14,), image (640, 480)
predict → action (6,): [-0.221, -1.213, 0.940, 0.198, -0.287, -0.618]

counterfactual.noun_swap → CRITICAL  score=0.0836
  variants:
    cube → block : 0.0598
    cube → ball  : 0.1073
```

HuggingFace's 450M-param SmolVLA shows the most dramatic noun-blindness —
swapping the manipulated noun barely moves its action prediction (≪ noise
floor). Strong signal that this model relies heavily on the image distribution
and weakly on the text token of the noun.

### 4. GR00T-N1.7-3B with DROID-shape state

```
Scene: Bridge image, DROID-format 9-dim state (eef + rot6d), gripper=0.5
       instruction: "put small spoon from basket to tray"
predict → action (17,):
  eef_9d: [0.504, 0.004, 0.307, 0.9999, 0.0015, 0.0106, -0.0014, 1.0, ...]
  + gripper(1) + joint_position(7)

counterfactual.noun_swap → CRITICAL  score=0.229
  variants:
    spoon → fork  : 0.426 (close to threshold)
    spoon → knife : 0.031 (very small)
```

GR00T-N1.7 on DROID-style observations: noun→fork produced moderately
different actions (close to the noise floor); noun→knife produced almost
identical actions. Partial language grounding, dominated by visual context
on this scene.

---

## What this proves about Emboviz

1. **The framework is genuinely model-agnostic.** Same `predict(scene)` API
   wraps OpenVLA, LeRobot policies (SmolVLA), openpi (π0), and Isaac-GR00T —
   four upstream ecosystems with incompatible Python/torch/transformers pins,
   surfaced through one Emboviz interface.

2. **Per-adapter venvs are a clean architectural answer.** Each model lives
   in its own virtualenv with the upstream packages it needs (verified end-to-end
   on the VM). The wizard (`emboviz init`) walks users through the right setup.

3. **The diagnostics produce comparable severities across ecosystems.** Same
   `noun_swap` threshold (0.5 noise floor, 2.0 grounded) categorizes models
   consistently regardless of action space size (7-DOF Bridge vs 14-DOF ALOHA
   vs 17-DOF DROID).

4. **Cross-model pattern detection works.** Without any cloud aggregation
   yet, just running locally surfaces a non-trivial finding: three of four
   flagship VLAs show CRITICAL language-noun sensitivity on a single-scene
   test — worth investigating per-model with broader datasets.

---

## What's NOT a full diagnostic in this showcase

- **OpenVLA-OFT** — adapter **shipped and import-verified** in a fresh
  `work-oft` venv on the VM (torch 2.2.0 + TF 2.15 + custom transformers
  fork + dlimp + prismatic + experiments.robot.openvla_utils all resolve;
  `from emboviz.models.openvla_oft import OpenVLAOFTAdapter` runs clean).
  A real-checkpoint inference test would need ~15GB download of e.g.
  `moojink/openvla-7b-oft-finetuned-libero-spatial`. Adapter is ready
  for that; haven't done the full download in this session.
- **RDT-1B, Octo** — adapters planned; not in this run (flash-attn build
  + JAX framework friction respectively).
- **Multi-episode statistical aggregation** — the findings above are
  single-scene. Real product needs N-rollout aggregation (a Cloud Hub
  feature in our roadmap).
- **Full battery (all 18 diagnostics)** per model — limited to one
  diagnostic per model here for time; the framework supports running the
  full battery via `emboviz diagnose --suite full_profile`.

---

## Reproduce on your own setup

For each model, follow the per-venv install path (see `emboviz init` or
the adapter's docstring):

```bash
# OpenVLA-7B (on Bridge data)
uv venv .venv-openvla --python 3.12 && source .venv-openvla/bin/activate
uv pip install 'emboviz[openvla]'
emboviz diagnose --model openvla-7b --scene bridge:0

# SmolVLA (on v2.1 LeRobot datasets)
uv venv .venv-smolvla --python 3.10 && source .venv-smolvla/bin/activate
uv pip install torch torchvision 'transformers>=4.50,<5.0' 'lerobot>=0.5' num2words
uv pip install --no-deps -e /path/to/emboviz
# then point at any v2.1 LeRobot dataset

# π0 via openpi
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && source .venv/bin/activate
uv pip install --no-deps -e /path/to/emboviz

# GR00T-N1.7-3B
# 1. Click "Agree" on https://huggingface.co/nvidia/Cosmos-Reason2-2B
# 2. huggingface-cli login
# 3. uv venv .venv-gr00t --python 3.10 && activate
# 4. uv pip install 'torch>=2.7' transformers==4.57.3 + GR00T runtime deps
# 5. uv pip install --no-deps git+https://github.com/NVIDIA/Isaac-GR00T.git
# 6. uv pip install --no-deps -e /path/to/emboviz
```
