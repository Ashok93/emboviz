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

### Architecture in one diagram

```
                                                ┌── ~/.emboviz/venvs/openvla
                                                │   torch + transformers 4.49 + lerobot
                                                │   + openvla checkpoint code
                                                │   ↑ emboviz-openvla serve
                                                ↑   (ZMQ ROUTER on /tmp/.../openvla.sock)
~/.venv-emboviz   ────────  ZMQ DEALER  ──────  ↑
(your main venv:                                ↑   each adapter runs in its OWN venv,
 emboviz core +                                 ↑   on its OWN Python version, with its
 emboviz-openvla,                               ↑   OWN torch/transformers/etc. pinned —
 emboviz-oft, ...                               ↑   they never share dependencies.
 — no model deps)                               │
                                                ├── ~/.emboviz/venvs/pi0     (Python 3.11)
                                                ├── ~/.emboviz/venvs/gr00t   (Python 3.11)
                                                ├── ~/.emboviz/venvs/oft     (Python 3.11)
                                                └── ~/.emboviz/venvs/sam3    (Python 3.12)
```

The wire is msgpack-framed ZeroMQ over a Unix socket. Bytes, not pickle —
so each adapter can be on any Python version and the main venv can move
forward whenever you want.

### System deps

One system dep on Linux: `sudo apt install ffmpeg` (Mac: `brew install ffmpeg`).
That's the ONE thing pip can't handle for you — `lerobot` reads videos
via `torchcodec` which needs FFmpeg system libraries.

### Install (per adapter you actually want)

```bash
# Main venv — emboviz core + lightweight shims (~10 KB each; no model
# deps, no torch, no lerobot).
uv venv --python 3.11
uv pip install emboviz emboviz-lerobot emboviz-openvla emboviz-pi0 emboviz-gr00t emboviz-oft emboviz-sam3
```

`emboviz-lerobot` is the **shim** for the LeRobot dataset reader — its
heavy `lerobot` install (plus `torch` and lerobot's `rerun-sdk<0.27` pin)
lives in an **isolated reader venv**, built by `emboviz install-lerobot`,
*never* in the host. That isolation is deliberate: lerobot's rerun cap
would otherwise collide with core's own `rerun>=0.32` `.rrd` exporter.
The reader runs as a ZeroMQ worker and hands the host universal `Scene`s
over the wire — exactly like a model worker. If your episodes are HDF5 or
RLDS instead, you don't need `emboviz-lerobot` at all (`hdf5` reads
in-process; `rlds` is its own extra: `uv pip install 'emboviz[rlds]'`).

That's it. The first time you run `emboviz analyze --config <file>`,
emboviz transparently:

1. **Materialises** the adapter's runtime venv at `~/.emboviz/venvs/<name>`
   if it doesn't exist yet (the heavy install — torch, transformers,
   lerobot, openpi, etc. — runs once, prints visible progress, and is
   reused forever after).
2. **Spawns** the ZeroMQ worker in the background if no warm one is
   running yet, and waits for it to come up.
3. **Runs** the analysis through the worker.

You don't need to know which Python version each adapter pins, which
git ref openpi needs, or which `--no-deps` quirk gr00t has — the
adapter's `AdapterSpec` declares all of that and the lifecycle layer
follows it.

If you want to do the install step ahead of time (e.g. in CI, before a
long benchmark run), the explicit subcommand is still available:

```bash
emboviz install-openvla    # a model's runtime venv (explicit; same as the lazy path)
emboviz install-lerobot    # the isolated LeRobot dataset-reader venv
```

### Optional: PyTorch backend for π0's attention diagnostic

```bash
emboviz convert-pi0 pi0_libero
```

### Run an analysis

One run is described by **one config file** — model, dataset mapping, and
analysis parameters all in one place. No CLI flag soup. Shipped templates
live under `configs/` (one per model on its canonical dataset):

```bash
emboviz analyze --config configs/openvla-bridge.yaml
emboviz analyze --config pi0-libero            # by shipped-template name
emboviz analyze --config my-run.yaml --dry-run # cost estimate, no GPU spend
```

`--dry-run` is the only flag besides `--config`; everything else lives in
the config.

### Bring your own model + dataset

Copy the template closest to your setup and edit it:

```bash
cp configs/openvla-bridge.yaml configs/my-run.yaml
$EDITOR configs/my-run.yaml
emboviz analyze --config configs/my-run.yaml
```

A config looks like this (abridged — see `configs/README.md` for the full
field reference):

```yaml
model:
  adapter: openvla                 # installed adapter: openvla | oft | pi0 | gr00t
  kwargs:
    hf_repo: your-org/your-finetune # YOUR checkpoint (HF id or local dir)
    unnorm_key: bridge_orig

dataset:
  format: lerobot                  # lerobot | gr00t | hdf5 | rlds  (self-describing formats)
  path: IPEC-COMMUNITY/bridge_orig_lerobot   # HF repo id, local dir, h5 file, or TFDS builder name
  cameras:
    primary: observation.images.image_0      # model camera role -> dataset key
  state:
    key: observation.state
    convention: ee_pose            # joint_angles | ee_pose | ... (the format never encodes this)
  action: {key: action}
  gripper: {source: 6, kind: parallel_jaw, units: unit, range: [0.0, 1.0]}
  instruction: {from: tasks}

analysis:
  episodes: "537"                  # "7" / "0,3,7" / "0-5" / "all"
  mask_query: "the cloth"          # object SAM 3 localizes for the memorization diagnostic
  detector: sam3                   # sam3 | gd-sam
  diagnostics: all                 # or a list: [memorization, attention]

output: ./report/my-run
```

The schema is **identical for every input format** (`lerobot` / `gr00t` /
`hdf5` / `rlds`) — only the reader behind each `key` changes. The dataset's own
schema (dims, per-dim names) is read from the format itself (`meta/info.json`
/ HDF5 array shapes / the TFDS feature spec); you only declare what the
format *can't* encode (which camera is `primary`, the state convention, the
gripper spec). For `rlds`, `dataset.path` is the TFDS builder name with
optional `extra: {data_dir, split}`.

> Rerun `.rrd` and MCAP/rosbag2 are recording / debugging-viz formats, not
> dataset inputs — they are intentionally not config-accepted.

**Diagnostics** (`analysis.diagnostics`): `all`, an explicit list
(`[memorization, attention]`), or `all,-chunk` to subtract. Names —
`memorization` (`vision.memorization`), `modality` (`input.modality_dropout`),
`sensitivity` (`vision.scene_sensitivity`), `chunk` (`internal.chunk_consistency`),
`attention` (`internal.attention_drift`); short or full both accepted.

### What you get back, per episode

In every `report/episode_<idx>/`:

- **`summary.json`** — every metric the diagnostics produced, with the per-frame numbers.
- **`report.md` / `report.html`** — plain-English findings ("masked the bowl, action barely moved — memorized-trajectory signature, try an unseen episode") sorted worst-first. No internal severity words.
- **`rollout.rrd`** — open in Rerun: scrub frame-by-frame with attention heatmaps, GroundingDINO bbox + per-fill masked images for memorization, per-modality response timeline, per-camera occlusion grids, action plots with abrupt-shift markers.

Across all the episodes you analyzed, at the top of `report/`:

- **`aggregate.json` / `aggregate.md` / `aggregate.html`** — cross-episode patterns ("on 7/10 episodes the wrist camera was IGNORED in the final 20 frames before failure") with links to per-episode pages.

No prose synthesis, no "we think your model is broken because…" — just evidence, in the tools you already use, scrubbable frame by frame.

---

## What's supported

### Model adapters (one file per model)

Each adapter declares which interpretability surfaces it exposes — inference, attention, hidden states, FFN activations, residual patching, neuron ablation. emboviz checks those capabilities and runs every applicable diagnostic, skipping the rest with a clear "not supported" note.

| Family | Inference | **Attention extraction** | Hidden states / patching | Install |
|---|---|---|---|---|
| OpenVLA-7B | ✅ | ✅ shipped (HF `output_attentions`) | ✅ full mechanistic-interp suite | `uv pip install emboviz-openvla` + `emboviz install-openvla` |
| **OpenVLA-OFT** | ✅ | ✅ shipped (moojink LLaMA backbone, BOS-aware token ranges) | — | `uv pip install emboviz-oft` + `emboviz install-oft` |
| **π0 / π0.5** | ✅ | ✅ shipped (PaliGemma VLM inside openpi; needs `emboviz convert-pi0`) | — | `uv pip install emboviz-pi0` + `emboviz install-pi0` |
| **GR00T-N1 / N1.7** | ✅ | ✅ shipped (Qwen3-VL backbone inside Gr00tPolicy) | — | `uv pip install emboviz-gr00t` + `emboviz install-gr00t` |
| Mock (no GPU) | ✅ — for diagnostic-side dev | N/A | N/A | base install |

> Planned: LeRobot policies (ACT, Diffusion Policy, TDMPC2, VQ-BeT) as a future isolated adapter worker — not yet integrated.
| RDT-1B | 📅 planned (flash-attn build complexity) | | | |
| Octo | 📅 planned (JAX backend) | | | |

**Attention is core, not a nice-to-have.** Modern policies are transformers; their visual attention IS the interpretability surface most teams want. We extract it for every VLA we support — even when the upstream inference helper wraps it away. Per-adapter extraction work is non-trivial, but it's the work the product exists to do.

> **Why separate venvs?** Several upstream VLA/robotics packages pin
> different (and incompatible) versions of `transformers` and `torch`. We
> ship adapter code that wraps each cleanly, but mixing all of them in one
> venv is not possible today. Per-adapter optional-dep groups in
> `pyproject.toml` make this explicit.

### Robot profile (declared in your config)

There are no preshipped robot presets to match against — you describe your
robot inline in the run config's `dataset` block: the **state convention**
(joint-angles vs ee-pose — the one thing no dataset format encodes), the
**gripper** location/kind, and which dataset image key maps to which model
**camera role**. The reader builds a typed `RobotProfile` from that, reading
dims and per-dim names from the dataset's own schema (never hand-typed). See
`configs/` for ready-made templates and `configs/README.md` for the full
field reference.

### Data formats

| Format | Ingest | Export | Config `dataset.format` |
|---|---|---|---|
| LeRobot v3.0 (BridgeV2, LIBERO, DROID, ALOHA, custom HF uploads) | ✅ | — | `lerobot` |
| GR00T format — LeRobot v2.1 + `modality.json` (NVIDIA Isaac-GR00T) | ✅ (pkg: `emboviz-reader-gr00t`) | — | `gr00t` |
| HDF5 (Robomimic, ALOHA, NVIDIA Isaac Lab Mimic) | ✅ | — | `hdf5` |
| RLDS / TFDS (Open-X-Embodiment, RT-X, Octo) | ✅ (extra: `rlds`) | — | `rlds` |
| Rerun `.rrd` | — | ✅ **(killer feature)** | — (viz output, not a dataset input) |

These ingest formats are the self-describing "saved episode" formats:
emboviz reads dims/per-dim names from each format's own schema. Rerun `.rrd`
and MCAP/rosbag2 are recording / debugging-viz formats, not dataset inputs.

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

Everything heavy or version-conflicting runs in its **own isolated venv** and
talks to the lean host over a bytes wire (msgpack/ZeroMQ). The host has no
torch and no lerobot.

```
  host venv (lean)                              isolated worker venvs
  ┌────────────────────────────────┐           ┌────────────────────────────┐
  │ diagnostics · perturb · metrics │  msgpack  │ emboviz-openvla / oft /    │
  │ calibration · runner            │ ── over ▶ │ pi0 / gr00t   (VLA models) │
  │ exporters (Rerun .rrd +         │  ZMQ/UDS  │ emboviz-sam3   (detector)  │
  │   report.md/html)               │ ◀ (bytes) │ emboviz-lerobot (dataset   │
  │ + emboviz-wire contracts        │           │   reader)                  │
  └────────────────────────────────┘           └────────────────────────────┘
```

The **host engine** is model- and dataset-agnostic: it only speaks the
`emboviz-wire` contracts (`Scene` in, `ActionResult` out for models; `Scene`/
`Trajectory` out for readers). Adding a new VLA, detector, or dataset format =
one isolated worker package. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the contract.

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
emboviz/                 the lean host engine (no torch, no lerobot)
  core/            pure types (re-exported from emboviz-wire) + divergences
  models/          VLAModel protocol + mock (in-process) + model registry
  perturb/         instruction / image perturbers + target detection
  metrics/         action divergence, attention JS, pointing-game, ...
  probes/          trainable linear failure probes
  diagnostics/     the shipped diagnostics — perturb × metric × model
  exporters/       Rerun .rrd writer + failure-moment correlation
  datasets/        manifest builder (hdf5/rlds in-process; lerobot/gr00t → reader workers)
  taxonomy/        canonical failure-mode / preposition lists
  adapters/        worker registry + lifecycle (connect / connect_reader)
  _internal/       runner (run_story) + multi-episode aggregation + report.md/html
  cli/             analyze · list-models · list-datasets · install-<name> · convert-pi0

adapters/                isolated worker packages — one venv each:
  emboviz-wire     the shared ZMQ wire + contracts (Scene, VLAModel, EpisodeSource, codec)
  emboviz-openvla · emboviz-oft · emboviz-pi0 · emboviz-gr00t    VLA model workers
  emboviz-sam3     text→mask detector worker
  emboviz-lerobot  LeRobot dataset-reader worker
```

---

## Roadmap

**Shipped:**
- Multi-modal Scene refactor (typed Observations, RobotProfile, RequiredInputs, capability gating)
- Isolated-worker architecture over the bytes wire (models, detector, dataset reader each in its own venv)
- Dataset readers: LeRobot (isolated worker), HDF5 + RLDS (in-process)
- Rerun `.rrd` export (per-frame overlays, verdict ribbons, metric time-series)
- VLA adapters: OpenVLA, OpenVLA-OFT, π0/π0.5, GR00T-N1.7 (+ SAM 3 detector)
- The five diagnostics: memorization, modality-dropout, scene-sensitivity, chunk-consistency, attention-drift

**Next:**
- LeRobot-policy adapter (ACT, Diffusion Policy, TDMPC2, VQ-BeT) as an isolated worker
- Reasoning-output diagnostics (faithfulness / stability) for VLAs that emit chain-of-thought
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
