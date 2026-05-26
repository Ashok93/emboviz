# Emboviz

**The X-ray for your deployed robot policy.**

> Your VLA picks the wrong cup. Freezes mid-grasp. Reaches the wrong side of the table. Validation loss looked great. Why is the real robot doing this?
>
> Emboviz takes your trained model and your recorded deployment episodes — especially the failing ones — and tells you, in plain English, what the policy was actually consuming, what it was ignoring, and where in the trajectory things went off the rails. With overlays you scrub frame-by-frame in Rerun.

![status](https://img.shields.io/badge/status-alpha-orange) ![license](https://img.shields.io/badge/license-Apache%202.0-blue) ![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

---

## The problem we solve

You trained a VLA. You deployed it on the real robot. Some episodes work. Many don't. You're staring at footage of the robot doing dumb things and you have **one** artifact to interrogate (a black-box policy) and **one** source of truth (your recorded episodes).

Validation metrics can't help you here — the model already passed them. Sim eval can't help either — the failure is in the real world. The questions you actually need answered are:

- **Is my policy looking at the target object, or has it gone visually blind?**
- **Is it listening to the instruction, or just replaying memorized motion from training?**
- **Where in the trajectory did attention drift away from the task?**
- **Was the model leaning on the wrist camera that just got occluded — and that's why it froze?**
- **Are the action chunks internally consistent, or did the policy lose coherence at step 87?**

Today the only answer is *"run more rollouts and guess."* Emboviz answers those questions directly, on your own recordings, in one command.

## What Emboviz is NOT

- Not a sim eval framework. Real-world failures don't reproduce in sim.
- Not a training-loss tool. The model already converged; that's not the question.
- Not closed-loop evaluation. We analyze what your model DID, not what it might do.
- Not "imitation accuracy vs expert." VLAs are trained to generalize, not to copy. Pixel-matching the teleop demonstrator on training data tells you nothing about why real deployments fail.

We **surface signals**. You form conclusions. Debugger, not oracle.

---

## Quickstart

One system dep on Linux: `sudo apt install ffmpeg` (Mac: `brew install ffmpeg`).
That's the ONE thing pip can't handle for you — `lerobot` reads videos
via `torchcodec` which needs FFmpeg system libraries.

Then per model, three commands (each in its own venv — the model
adapter extras are mutually exclusive because the upstream packages
pin incompatible transformers / torch versions):

### OpenVLA-7B (or any fine-tune of it) on Bridge / LIBERO / any LeRobot data

```bash
uv venv && source .venv/bin/activate
uv pip install "emboviz[openvla]"
uv run emboviz analyze \
    --model openvla --dataset bridge \
    --episodes 0,1,2 --target "the spoon" \
    --output ./report
```

### OpenVLA-OFT on LIBERO

```bash
uv venv && source .venv/bin/activate
uv pip install "emboviz[oft]"     # pulls moojink fork + transformers patch automatically
uv run emboviz analyze \
    --model oft --dataset libero-spatial \
    --episodes 0,1,2 --target "the black bowl" \
    --output ./report
```

### π0 / pi0_libero (Physical Intelligence)

```bash
uv venv && source .venv/bin/activate
uv pip install "emboviz[pi0]"     # pulls openpi automatically
# 3rd command ONLY if you want the attention-extraction diagnostic
# (the other 4 diagnostics work on the JAX checkpoint as-is):
uv run emboviz convert-pi0 pi0_libero
uv run emboviz analyze \
    --model pi0 --dataset pi-libero \
    --episodes 0,1,2 --target "the white mug" \
    --output ./report
```

### GR00T-N1.7 (NVIDIA) on DROID or your LeRobot data

```bash
uv venv && source .venv/bin/activate
uv pip install "emboviz[gr00t]"
# Isaac-GR00T's flash-attn build dep fails under pip's standard build
# isolation. ``install-gr00t`` does the `--no-deps` install (gr00t's
# adapter falls back to SDPA / eager attention anyway):
uv run emboviz install-gr00t
uv run emboviz analyze \
    --model gr00t \
    --model-kwargs '{"camera_mapping": {"primary": "exterior_image_1_left", "wrist_left": "wrist_image_left"}}' \
    --dataset droid-sample --episodes 1 --target "the blue block" \
    --output ./report
```

### Bring your own data

The per-model quickstarts above use `--dataset <alias>` for the canonical
training datasets we pre-shipped (`bridge`, `libero-spatial`, `pi-libero`,
`droid-sample`, ...). For your own dataset or your own deployment recording,
point emboviz at it directly with `--dataset-format <fmt>` + `--dataset-path <path>`:

```bash
# Local HDF5 file (Robomimic / ALOHA / NVIDIA Isaac Lab Mimic)
uv run emboviz analyze --model pi0 \
    --dataset-format hdf5 --dataset-path /data/my_demos.hdf5 \
    --dataset-kwargs '{"camera_keys": {"primary": "agentview_rgb", "wrist": "robot0_eye_in_hand_rgb"}, "instruction": "pick up the mug"}' \
    --episodes 0 --target "the white mug" --output ./report

# RLDS / Open-X-Embodiment (needs `uv pip install 'emboviz[rlds]'`)
uv run emboviz analyze --model gr00t \
    --dataset-format rlds --dataset-path /tfds \
    --dataset-kwargs '{"builder_name": "bridge_orig", "camera_keys": {"primary": "image_0"}}' \
    --episodes 0,1 --target "the green block" --output ./report

# MCAP deployment recording from ROS 2 / NVIDIA Isaac SIM
uv run emboviz analyze --model oft \
    --dataset-format mcap --dataset-path /logs/rollout_2026_05_26.mcap \
    --dataset-kwargs '{"topic_map": {"primary": "/camera/color/image_raw", "state": "/joint_states", "action": "/cmd_joint"}}' \
    --episodes 0 --target "the black bowl" --output ./report

# Rerun .rrd deployment recording
uv run emboviz analyze --model openvla \
    --dataset-format rerun-rrd --dataset-path /logs/rollout.rrd \
    --episodes 0 --target "the spoon" --output ./report
```

Each format's adapter requires a small set of mapping kwargs (which
HDF5 group / RLDS field / ROS topic holds each camera, the state, the
action) — pass them via `--dataset-kwargs '<JSON>'`. See
`emboviz.datasets.hdf5:HDF5EpisodeSource`,
`emboviz.datasets.rlds:RLDSEpisodeSource`, etc. for the full signature.

**For your own LeRobot v2/v3 dataset:** use one of the pre-shipped
aliases (`bridge`, `libero-spatial`, `droid-sample`, ...) if your robot
matches a pre-shipped `RobotProfile`. For a different robot, subclass
`emboviz.datasets.lerobot:LeRobotEpisodeSource` (~10 lines — see
`emboviz/datasets/lerobot_bridge.py` for the template), then pass
`--dataset emboviz.your_module:YourSource`. LeRobot adapters need
a `RobotProfile` object (action axes, gripper convention) that can't
be expressed in JSON, so they don't fit the generic
`--dataset-format` flow.

Run `uv run emboviz list-datasets` to see every supported format on the
current install (✓ = installed, · = needs an extra) with the exact
`uv pip install` line for any missing one.

### What you get back, per episode

In every `report/episode_<idx>/`:

- **`summary.json`** — every metric the diagnostic suite produced, with the per-frame numbers.
- **`report.md` / `report.html`** — plain-English findings ("masked the bowl, action barely moved — memorized-trajectory signature, try an unseen episode") sorted worst-first. No internal severity words.
- **`rollout.rrd`** — open in Rerun: scrub frame-by-frame with attention heatmaps, GroundingDINO bbox + per-fill masked images for memorization, per-modality response timeline, per-camera occlusion grids, action plots with abrupt-shift markers.

Across all the episodes you analyzed, at the top of `report/`:

- **`aggregate.json` / `aggregate.md` / `aggregate.html`** — cross-episode patterns ("on 7/10 episodes the wrist camera was IGNORED in the final 20 frames before failure") with links to per-episode pages.

No prose synthesis, no "we think your model is broken because…" — just evidence, in the tools you already use, scrubbable frame by frame.

---

## What's supported

### Model adapters (one file per model)

Each adapter declares which interpretability surfaces it exposes — inference, attention, hidden states, FFN activations, residual patching, neuron ablation. The diagnostic suite checks the capabilities and runs every applicable test.

| Family | Inference | **Attention extraction** | Hidden states / patching | Install |
|---|---|---|---|---|
| OpenVLA-7B | ✅ | ✅ shipped (HF `output_attentions`) | ✅ full mechanistic-interp suite | `uv pip install 'emboviz[openvla]'` |
| **OpenVLA-OFT** | ✅ | ✅ shipped (moojink LLaMA backbone, BOS-aware token ranges) | — | `uv pip install 'emboviz[oft]'`; separate venv |
| **π0 / π0.5** | ✅ | ✅ shipped (PaliGemma VLM inside openpi; needs `emboviz convert-pi0`) | — | `uv pip install 'emboviz[pi0]'`; separate venv |
| **GR00T-N1 / N1.7** | ✅ | ✅ shipped (Eagle-2 VLM inside Gr00tPolicy) | — | `uv pip install 'emboviz[gr00t]'` + `uv run emboviz install-gr00t` |
| LeRobot policies (ACT, Diffusion Policy, TDMPC2, VQ-BeT) | ✅ via `LeRobotPolicyAdapter` | 🚧 case-by-case (depends on backbone) | — | base install |
| Mock (no GPU) | ✅ — for diagnostic-side dev | N/A | N/A | base install |
| RDT-1B | 📅 planned (flash-attn build complexity) | | | |
| Octo | 📅 planned (JAX backend) | | | |

**Attention is core, not a nice-to-have.** Modern policies are transformers; their visual attention IS the interpretability surface most teams want. We extract it for every VLA we support — even when the upstream inference helper wraps it away. Per-adapter extraction work is non-trivial, but it's the work the product exists to do.

> **Why separate venvs?** Several upstream VLA/robotics packages pin
> different (and incompatible) versions of `transformers` and `torch`. We
> ship adapter code that wraps each cleanly, but mixing all of them in one
> venv is not possible today. Per-adapter optional-dep groups in
> `pyproject.toml` make this explicit.

### Robot profiles (preshipped configs)

| Robot | Status |
|---|---|
| BridgeV2 (`bridge_orig`) | ✅ shipped |
| Franka Panda + Robotiq 2F-85 | ✅ shipped |
| UR5 / UR10 + Robotiq | ✅ shipped |
| Trossen ViperX-300 (single-arm ALOHA) | ✅ shipped |
| ALOHA bimanual | 📅 roadmap |
| Unitree H1 / G1 | 📅 roadmap |

Custom robots: write a ~30-line `RobotProfile` and drop it in `emboviz/profiles/`.

### Data formats

| Format | Ingest | Export | Selector |
|---|---|---|---|
| LeRobot v2/v3 (BridgeV2, LIBERO, DROID, ALOHA, custom HF uploads) | ✅ | — | `--dataset-format lerobot` |
| HuggingFace `datasets` (generic) | ✅ | — | `--dataset-format hf` |
| HDF5 (Robomimic, ALOHA, NVIDIA Isaac Lab Mimic) | ✅ | — | `--dataset-format hdf5` |
| RLDS / TFDS (Open-X-Embodiment, RT-X, Octo) | ✅ (extra: `rlds`) | — | `--dataset-format rlds` |
| MCAP (ROS 2, NVIDIA Isaac SIM) | ✅ | — | `--dataset-format mcap` |
| Rerun `.rrd` | ✅ | ✅ **(killer feature)** | `--dataset-format rerun-rrd` |
| ROS bag (native) | 📅 roadmap | — | — |

---

## What each metric actually is

Plain-English: what each diagnostic answers, what we do, what you get back.

### 1. Memorization — *"Is my policy actually looking at the object, or playing back a memorized motor pattern?"*

You tell us what object to mask (`"the mug"`, `"the lid"`, `"the welding torch"`). For every frame, we find that object in the image with GroundingDINO + SAM, mask it (channel-mean fill **and** Gaussian-blur fill — we require both to agree before calling memorization), and measure how much the predicted action changes.

**Output per frame:** action delta (normalized to % of typical action) under each fill, plus the masked image so you can eyeball that the mask actually covered the right thing.
**Skips with reason when:** GroundingDINO can't confidently find the object, or the mask is too low-contrast to count as a real intervention. **No fabricated verdicts.**

### 2. Modality dropout — *"Which inputs is my policy actually using?"*

Your model declares it consumes [primary camera, wrist camera, state, gripper, action history, instruction]. For each declared input we swap it with a real value sampled from a *different episode in the same dataset* (a real state from another rollout, an instruction from another task, etc.) and measure how much the predicted action changes. If swapping the instruction barely moves the action → your model isn't using language.

**Output per modality per frame:** intervention magnitude (how different the swapped value was from the original) and response magnitude (how much the action changed). The ratio is the headline.
**Skips with reason when:** the substitute pool is too uniform to give a meaningfully different sample (e.g. instruction dropout on a 3-task dataset).

### 3. Scene sensitivity — *"Where in the image does my policy look?"*

A sliding occluder sweeps each camera, region by region. We measure how much the action changes per region and aggregate into a per-pixel heatmap. The shape of that heatmap tells you whether the model is focused (good — uses specific regions) or diffuse (bad — relies on background cues).

**Output per camera per frame:** heatmap PNG + Hoyer-sparsity scalar (calibration-aware — z-scored against a null distribution of shuffled cells, so the threshold doesn't lie about pure noise).

### 4. Attention map / attention drift — *"What is the model paying attention to inside its own forward pass?"*

We hook into the model's attention layers, extract the per-frame visual-attention distribution, and report:
- **Attention heatmap** overlaid on each camera (where the model is looking)
- **Pointing accuracy** — fraction of attention mass inside your target's bounding box (is it anchored on the right thing?)
- **Drift** between consecutive frames (Wasserstein-2 + top-K IoU) — does focus stay coherent or wander?

This is the load-bearing interpretability surface for modern transformer-based policies. **We extract this for every model we support** (OpenVLA, OpenVLA-OFT, π0, GR00T) — not a fundamental limitation of the model, just per-adapter extraction work.

### 5. Chunk consistency — *"Can I trust the model's multi-step plan, or do I need to replan every step?"*

For policies that predict an action *chunk* (OpenVLA-OFT, π0, GR00T, ACT, Diffusion Policy): at frame *t* the model predicts `[a_t, a_{t+1}, ...]`. At frame *t+1* it predicts `[a'_{t+1}, ...]`. We compare what it said for *t+1* at time *t* vs what it says for *t+1* at time *t+1*. If they agree, your lookahead is stable and you can commit multiple steps open-loop. If they disagree past step 0, you need to replan every step.

**Output:** "safely-committable horizon" — how many steps from the chunk you can actually trust — plus the per-step delta curve.

---

## Research foundations

Every diagnostic above is derived from published methodology. We do not invent algorithms; we implement the literature standard, faithfully, and refuse verdicts when our setup violates the methodology's assumptions.

| Metric | Direct prior art (2024-2026) | Foundational (theoretical bedrock) |
|---|---|---|
| **Memorization** | BYOVLA (Hancock et al. 2024, [arXiv:2410.01971](https://arxiv.org/abs/2410.01971)) — direct precedent. LIBERO-PRO (Geng et al. 2025, [arXiv:2510.03827](https://arxiv.org/abs/2510.03827)) — memorization signature framing. GroundingDINO (Liu et al. ECCV 2024, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)) — phrase grounding. SAM 2 (Ravi et al. 2024). | Causal mediation (Vig et al. NeurIPS 2020, [arXiv:2004.12265](https://arxiv.org/abs/2004.12265)). Baseline blindness (Sturmfels, Lundberg & Lee, Distill 2020). Sanity checks for saliency (Adebayo et al. NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)). |
| **Modality dropout** | "Do You Need Proprioceptive States?" (Lin et al. 2025, [arXiv:2509.18644](https://arxiv.org/abs/2509.18644)) — direct precedent for state ablation in VLAs. CAST counterfactual labels (2025, [arXiv:2508.13446](https://arxiv.org/abs/2508.13446)). "When Vision Overrides Language" ([arXiv:2602.17659](https://arxiv.org/abs/2602.17659)). | SHAP (Lundberg & Lee NeurIPS 2017, [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)) — marginal-distribution attribution. Janzing, Minorics & Blöbaum AISTATS 2020 ([arXiv:1910.13413](https://arxiv.org/abs/1910.13413)) — marginal vs conditional. Hooker & Mentch 2019 ([arXiv:1905.03151](https://arxiv.org/abs/1905.03151)) — permutation pitfalls. Zhou et al. CVPR 2019 ([arXiv:1812.07035](https://arxiv.org/abs/1812.07035)) — why zeros break structured representations. |
| **Scene sensitivity** | BYOVLA (Hancock et al. 2024). "Shortcut Learning in Generalist Robot Policies" (CoRL 2025, [arXiv:2508.06426](https://arxiv.org/abs/2508.06426)). Policy Contrastive Decoding (2025, [arXiv:2505.13255](https://arxiv.org/abs/2505.13255)). | Occlusion sensitivity (Zeiler & Fergus ECCV 2014, [arXiv:1311.2901](https://arxiv.org/abs/1311.2901)) — still the Captum default. Hoyer sparsity axioms (Hurley & Rickard 2009, IEEE TIT). RISE smooth-mask refinement (Petsiuk et al. BMVC 2018, [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)). Adebayo et al. 2018 sanity-checks methodology. |
| **Attention** | AVA-VLA (2025, [arXiv:2511.18960](https://arxiv.org/abs/2511.18960)) — visual attention failure modes in VLAs. Head Pursuit (2025, [arXiv:2510.21518](https://arxiv.org/abs/2510.21518)) — head specialization. "Functional Roles of Attention Heads in VLMs" (2025, [arXiv:2512.10300](https://arxiv.org/abs/2512.10300)). "How Multimodal LLMs Solve Image Tasks" (2025, [arXiv:2508.20279](https://arxiv.org/abs/2508.20279)) — layer-wise visual-grounding stages. "Understanding Sink Tokens in MLLMs" (OpenReview 2024). | Attention-is/isn't-Explanation debate (Jain & Wallace NAACL 2019; Wiegreffe & Pinter 2019, [arXiv:1908.04626](https://arxiv.org/abs/1908.04626)). Attention rollout (Abnar & Zuidema ACL 2020, [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)). Wasserstein for saliency (Liu et al. PLOS ONE 2017). |
| **Chunk consistency** | Bidirectional Decoding (Liu et al. ICLR 2025, [arXiv:2408.17355](https://arxiv.org/abs/2408.17355)) — direct precedent, defines our metric. Mixture of Horizons (2025, [arXiv:2511.19433](https://arxiv.org/abs/2511.19433)) — safely-committable horizon. Adaptive Action Chunking ([arXiv:2604.04161](https://arxiv.org/abs/2604.04161)). | ACT (Zhao et al. RSS 2023, [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)). Diffusion Policy (Chi et al. 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)). π0 (Black et al. 2024). OpenVLA-OFT (Kim et al. 2025, [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)). GR00T N1 (NVIDIA 2025, [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)). |

**Full literature reference, with citations and per-model methodology notes:** see [`LITERATURE.md`](./LITERATURE.md). Every algorithm in this repo is justified there or it doesn't ship.

Each diagnostic is one file in `emboviz/diagnostics/`. Adding a new technique from next month's paper is a single-file change.

---

## Architecture

```
                       ┌─────────────────────┐
                       │  exporters/ + viz/  │   Rerun + Foxglove + scorecards
                       ├─────────────────────┤
                       │  suites/            │   composable batteries
                       ├─────────────────────┤
                       │  diagnostics/       │   orchestrate perturb + metric + model
                       ├──────────┬──────────┤
              perturb/ │ metrics/ │ probes/  │   composable primitives
              ├────────┴──────────┴──────────┤
              │  models/  (VLAModel ABC)     │   one adapter per model family
              ├──────────────────────────────┤
              │  core/  types + observations │   pure foundation
              └──────────────────────────────┘

       datasets/  ╳  profiles/  ╳  taxonomy/  ╳  coverage/
```

The **core engine** (everything outside `models/`) is model-agnostic. Adding a new VLA = one adapter file. Adding a new diagnostic = one file. Adding a new data format = one file. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the contract.

---

## How it stays out of your way

- **Your model weights never leave your infrastructure.** The engine runs wherever the model lives — your workstation, training cluster, CI runner.
- **Your raw rollout video stays local by default.**
- **No telemetry from the engine unless you explicitly opt in.**
- **Apache 2.0 forever.** The engine and all adapters/profiles/formats are open source under a permissive license.

A hosted Emboviz Hub is planned for team workflows (run history, regression alerts, CI integration, AI-powered drill-down with cross-team context). The engine and outputs you get locally are not a feature-flagged demo — they're the full thing.

---

## Repository layout

```
emboviz/
  core/            Layer 0 — pure types, Observations, RobotProfile
  models/          Layer 1 — VLAModel protocol + adapters
  perturb/         Layer 2 — instruction / image / state perturbers
  metrics/         Layer 2 — divergences, JS, pointing-game, ablation drops
  probes/          Layer 2 — trainable linear probes (e.g., failure predictor)
  diagnostics/     Layer 3 — orchestrate perturb + metric + model
  suites/          Layer 4 — preset diagnostic batteries
  exporters/       Layer 5 — Rerun, Foxglove, scorecard, JSON
  viz/             Layer 5 — plotting primitives
  datasets/        Adjacent — episode source adapters per format
  profiles/        Adjacent — preshipped robot profiles
  coverage/        Adjacent — dataset gap analysis
  taxonomy/        Adjacent — canonical lists (failure modes, prepositions)
  cli/             Layer 6 — entry points (diagnose, init, validate, compare)
```

---

## Roadmap

**Now (foundation):**
- Multi-modal Scene refactor (typed Observations, RobotProfile, RequiredInputs, capability gating)
- State-side perturbers (gripper_flip and friends)
- Generic rollout loader + format adapters (LeRobot v3, RLDS, ROS bag, HuggingFace, Rerun, Foxglove)
- Rerun + Foxglove EXPORT (the killer integration)
- Model adapter coverage (OpenVLA-OFT, π0, GR00T, ACT, Diffusion Policy, RDT, Octo)
- Robot profile coverage (Franka, UR, Trossen)

**Next (productization):**
- Onboarding wizard (`emboviz init`)
- Pluggable failure labelers + composable suites
- Sim integration (Isaac Lab, RoboSuite, Mujoco MJX)
- Documentation site + example gallery

**Later (Hub):**
- Persistent run history, training-history timelines
- CI integration + regression alerts
- Hosted compute for public models
- AI-powered conversational drill-down with cross-team context

---

## Contributing

Adapters, robot profiles, and format loaders are the highest-leverage contributions — each one helps every team using that model or robot or data format. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the protocols you'd implement.

---

## License

Apache 2.0. See [`LICENSE`](./LICENSE).
