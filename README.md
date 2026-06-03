# Emboviz

**An interpretability toolkit for deployed Vision-Language-Action (VLA) policies.**

![status](https://img.shields.io/badge/status-alpha-orange) ![license](https://img.shields.io/badge/license-Apache%202.0-blue) ![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

Emboviz takes a trained VLA policy and your recorded episodes and tells you what
the policy was actually consuming, what it was ignoring, and how stable its
behaviour was — as per-frame metrics and scrubbable overlays you open in
[Rerun](https://rerun.io). It surfaces evidence from the model's own forward
pass; it does not score the policy or guess at root causes.

Every diagnostic is derived from published methodology and cited in
[`LITERATURE.md`](./LITERATURE.md).

<p align="center">
  <img src="./assets/attention.gif" alt="Emboviz attention overlay scrubbed frame-by-frame in Rerun" width="100%">
</p>

---

## The diagnostics

| Diagnostic | The question it answers |
|---|---|
| **Memorization** | Is the policy actually looking at the target object, or replaying a memorized motion? We mask the object and measure how much the action changes. |
| **Modality dropout** | Which inputs does the policy actually use? We swap each input (camera, state, gripper, history, instruction) with a real value from another episode and measure the response. |
| **Scene sensitivity** | Where in each camera image does the policy look? A sliding occluder sweeps the frame and we build a per-pixel saliency heatmap. |
| **Attention drift** | Where does the model attend inside its own forward pass, and does that focus stay anchored across the trajectory or wander? |
| **Chunk consistency** | For policies that predict action chunks: can you trust the multi-step lookahead, or must you replan every step? We measure how far ahead the plan stays self-consistent. |

Short and full names are both accepted: `memorization`, `modality`,
`sensitivity`, `attention`, `chunk`.

---

## System requirements

emboviz loads your policy in its native runtime, so you need roughly the
**same machine you'd run that model's inference on** — plus a bit of headroom
for the SAM 3 detector and LaMa fill when you use the memorization diagnostic.

| | |
|---|---|
| **GPU** | Enough VRAM to run your policy (~24 GB for a 7B VLA like OpenVLA / OFT; less for smaller policies) |
| **CPU** | Any modern multi-core (video decoding is CPU-bound) |
| **RAM** | 16 GB for a 7B VLA (enough to stage its weights); less for smaller policies |

---

## Installation

Emboviz is **not yet published to PyPI** — install it from a clone of this
repository with [`uv`](https://docs.astral.sh/uv/). One `uv sync` command sets
up everything: the core engine, both dataset readers, the SAM 3 detector and the
LaMa fill, and the model adapter you ask for — all from the clone.

### 1. System dependencies

```bash
sudo apt install ffmpeg python3-dev build-essential
```

### 2. Clone + install your model

Pick the extra for the policy you want to analyze — the command is the same,
only the name changes:

```bash
git clone https://github.com/Ashok93/emboviz.git && cd emboviz

uv sync --extra openvla     # OpenVLA-7B
uv sync --extra oft         # OpenVLA-OFT
uv sync --extra pi0         # π0 / π0.5
uv sync --extra gr00t       # GR00T-N1 / N1.7
uv sync --extra act         # ACT
uv sync --extra smolvla     # SmolVLA
# uv sync --extra all       # every adapter at once
```

That single command installs everything host-side: the core engine, the **SAM 3
detector** and **LaMa fill** (used by the memorization diagnostic), **both**
dataset readers (LeRobot v3.0 and GR00T-format — `dataset.format: lerobot |
gr00t`), and the model adapter. Each adapter is a thin shim; its heavy runtime
(torch, transformers, openpi, …) is built into an isolated worker environment
**automatically on first analysis**.

> **Memorization needs SAM 3 (gated).** The shipped configs run four
> diagnostics by default — `[modality, sensitivity, attention, chunk]` — which
> need no token. The **memorization** diagnostic is left out by default because
> it uses Meta's [SAM 3](https://huggingface.co/facebook/sam3) to locate the
> target object, and SAM 3 is a **gated** model on the Hugging Face Hub. To use
> it, accept its license once, authenticate, then add `memorization` to your
> config's `diagnostics:` list:
>
> ```bash
> uv run hf auth login          # or: export HF_TOKEN=hf_xxx
> ```
>
> ```yaml
> analysis:
>   diagnostics: [memorization, modality, sensitivity, attention, chunk]
> ```

> **π0 attention** needs a one-time PyTorch conversion of the checkpoint:
> `uv run emboviz convert-pi0 pi0_libero`. Plain inference needs nothing extra.

---

## Running an analysis

One run is one config file — model, dataset mapping, and analysis parameters in
one place. Templates live in `configs/` (one per model on its canonical dataset).

```bash
uv run emboviz analyze --config configs/openvla.yaml
```

To analyze your own checkpoint and data, copy the closest template and edit it:

```yaml
model:
  adapter: openvla                  # openvla | oft | pi0 | gr00t | act | smolvla
  kwargs:                           # constructor overrides → your checkpoint
    hf_repo: your-org/your-finetune # HF repo id or local dir
    unnorm_key: bridge_orig         # adapter-specific; see the shipped template for your model

dataset:
  format: lerobot                   # lerobot | gr00t
  path: your-org/your-dataset       # HF repo id or local dir
  cameras:                          # model camera role → this dataset's image key
    primary: observation.images.image_0
  state:    {key: observation.state, convention: ee_pose}   # convention: joint_angles | ee_pose | ... (required, never guessed)
  action:   {key: action}
  gripper:  {source: 6, kind: parallel_jaw, units: unit, range: [0.0, 1.0]}   # optional; omit to leave the gripper inside the state vector
  instruction: {from: tasks}        # natural-language instruction from the dataset's task table

analysis:
  episodes: "537"                   # the episode to analyze
  frame_start: 0                    # first frame analyzed
  n_frames: -1                      # -1 = the whole episode; set a number to cap it
  frame_stride: 5                   # analyze every 5th frame

  # memorization is omitted by default — it needs SAM 3 (see the gated-model
  # note above). Add `memorization` once you've authenticated.
  diagnostics: [modality, sensitivity, attention, chunk]

  # Used only by the memorization diagnostic (when added above):
  mask_query: "the cloth"           # the manipulated object to mask
  detector: sam3                    # sam3 | gd-sam
  # detector_score_threshold: 0.5   # optional; SAM 3's default. Lower to catch faint/small targets
  # detector_mask_threshold: 0.5    # optional; per-pixel mask cutoff (SAM 3's default). Lower = fuller object removal
  # memorization_require_cameras: primary  # primary (default) | all | [roles]; views that must show the target
  fills: [channel_mean, gaussian_blur]   # add lama_inpaint for the on-manifold fill (needs emboviz-lama)

output: ./report/my-run
```

The schema is identical for every input format — only the reader behind each
`key` changes. Dimensions and per-dim names are read from the dataset's own
schema; you declare only what the format can't encode (camera roles, state
convention, gripper spec). See `configs/README.md` for the full field reference.

### What you get back

Per episode, in `report/episode_<idx>/`:

- **`summary.json`** — every metric, with the per-frame numbers.
- **`report.md`** — plain-English findings, worst-first.
- **`rollout.rrd`** — open in Rerun: scrub frame-by-frame with attention
  heatmaps, memorization mask + per-fill overlays, per-modality response
  timelines, occlusion grids, and action plots.

Across all analyzed episodes, at the top of `report/`:

- **`aggregate.{json,md}`** — cross-episode patterns, linked to per-episode
  pages.

---

## Supported models

| Model | Inference | Attention | Notes |
|---|---|---|---|
| **OpenVLA-7B** | ✅ | ✅ | Full mechanistic-interp surface (hidden states, FFN, patching, ablation). |
| **OpenVLA-OFT** | ✅ | ✅ | Multi-stream (primary + wrist). |
| **π0 / π0.5** | ✅ | ✅ | Attention needs `emboviz convert-pi0`. |
| **GR00T-N1 / N1.7** | ✅ | ⚠️ | Attention is the DiT motor pathway — dispersed, not a tight object localizer (see below). |
| **ACT** | ✅ | ✅ | lerobot ACTPolicy. Vision + state, no language. Attention is the DETR decoder cross-attention (action pathway). |
| **SmolVLA** | ✅ | ✅ | lerobot SmolVLAPolicy. Vision + language + state; stochastic (flow-matching). Attention from the SmolVLM2 prefix (instruction → image). |

> **GR00T attention — read with care.** OpenVLA, OFT and π0 are single-stack:
> the action is produced *through* the VLM's attention, so "where the last token
> looks" is where it acts, and the map locks onto the manipulated object. GR00T
> is dual-system — a *frozen* Qwen3-VL reasoning model feeds a *separate*
> diffusion-transformer (DiT) action head. We extract GR00T's map from the DiT's
> action→image cross-attention (the only action-grounded signal), but that is the
> **motor pathway** and is spatially **dispersed** across the workspace rather
> than anchored on the target. This is a documented property of VLAs, not an
> emboviz bug — see [`LITERATURE.md` §4](./LITERATURE.md) for the citations
> (ReconVLA, the VLA survey, the GR00T-N1.5 mechanistic study). Treat it as
> "where the action pathway attends," not as a reliable object localizer.

---

## Research foundations

Every diagnostic is grounded in published methodology, with per-model
methodology notes and the antipatterns we deliberately avoid. The full
reference — citations, algorithms, and shipped-vs-design-target status for each
metric — is in [`LITERATURE.md`](./LITERATURE.md).

---

## Contributing

The highest-leverage contributions are **new model adapters** and **new dataset
readers** — each one unlocks emboviz for everyone using that policy or data
format. Both plug in behind a small, stable contract without touching the core
engine, and adding a new diagnostic is a single file under
`emboviz/diagnostics/`.

[`ARCHITECTURE.md`](./ARCHITECTURE.md) is the contributor's guide: it explains
how the pieces fit together and walks through adding an adapter, a reader, and a
diagnostic step by step.

---

## License

Emboviz is released under the **Apache License 2.0** (see [`LICENSE`](./LICENSE)).
This covers the source code in this repository — the core engine, the adapters,
and the diagnostics.

### Third-party models and datasets

Emboviz does not redistribute model weights or datasets. The adapters download
checkpoints from their original sources, and you supply your own datasets. These
components are governed by their own licenses, which are independent of
Emboviz's. You are responsible for reviewing and complying with the license of
each model and dataset you use, including any restrictions on commercial use,
fields of use, and redistribution.

The models accessible through the shipped adapters are:

| Adapter | Upstream | License |
|---|---|---|
| `openvla`, `oft` | [OpenVLA](https://github.com/openvla/openvla) | MIT (code and checkpoints); weights inherit the Llama 2 Community License from the base model |
| `pi0` | [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) | Apache 2.0 |
| `gr00t` | [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | Code Apache 2.0; model weights under the NVIDIA License |
| `act`, `smolvla` | [LeRobot](https://github.com/huggingface/lerobot) | Code Apache 2.0; checkpoint weights carry the license of the specific model you load |
| `sam3` | [Meta Segment Anything 3](https://huggingface.co/facebook/sam3) | SAM License — source-available, permits commercial use with restrictions; not OSI open-source. The `--detector gd-sam` alternative uses GroundingDINO and SAM 2, both Apache 2.0 |
| `lama` | [LaMa / big-lama](https://github.com/advimman/lama) | Apache 2.0 (code and checkpoints). The default TorchScript export is fetched from [`okaris/big-lama`](https://huggingface.co/okaris/big-lama), pinned to a commit |

Datasets read through the LeRobot and GR00T-format readers (e.g. Open
X-Embodiment, LIBERO, DROID, BridgeData) each carry their own license and terms;
consult the dataset's own documentation.

The license identifications above are provided for convenience and may change
upstream. The authoritative terms are those distributed with each model and
dataset.
