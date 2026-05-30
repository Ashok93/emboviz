# Emboviz

**An interpretability toolkit for deployed Vision-Language-Action (VLA) policies.**

![status](https://img.shields.io/badge/status-alpha-orange) ![license](https://img.shields.io/badge/license-Apache%202.0-blue) ![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

Emboviz takes a trained VLA policy and your recorded episodes and tells you what
the policy was actually consuming, what it was ignoring, and how stable its
behaviour was — as per-frame metrics and scrubbable overlays you open in
[Rerun](https://rerun.io). It surfaces evidence from the model's own forward
pass; it does not score the policy or guess at root causes.

Every diagnostic is derived from published methodology and cited in
[`LITERATURE.md`](./LITERATURE.md). If an algorithm can't be justified from the
literature, it isn't shipped.

---

## The diagnostics

Run one or more per analysis. Each one refuses to emit a verdict (and says why)
when its preconditions aren't met, rather than fabricating a number.

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

## Installation

### System dependency

Emboviz reads episode video through `torchcodec`, which needs FFmpeg system
libraries:

```bash
sudo apt install ffmpeg     # Linux
brew install ffmpeg         # macOS
```

### Core + dataset reader

The core package has no model or dataset dependencies. Install it together with
the reader for your dataset format:

```bash
uv venv --python 3.11
uv pip install emboviz                 # core engine
uv pip install emboviz-lerobot         # LeRobot v3.0 datasets
```

Supported dataset formats:

| Format | Install |
|---|---|
| LeRobot v3.0 (BridgeV2, LIBERO, DROID, ALOHA, custom HF uploads) | `emboviz-lerobot` |
| GR00T format (LeRobot v2.1 + `modality.json`) | `emboviz-reader-gr00t` |

### The model you want

Install the adapter for each policy you intend to analyze. An adapter is a small
package; its model runtime (torch, transformers, openpi, and others) is
installed into an isolated worker environment on first use.

```bash
uv pip install emboviz-openvla         # OpenVLA-7B
uv pip install emboviz-oft             # OpenVLA-OFT
uv pip install emboviz-pi0             # π0 / π0.5
uv pip install emboviz-gr00t           # GR00T-N1 / N1.7
uv pip install emboviz-sam3            # SAM 3 detector (for the memorization mask)
```

Adapters do not share dependencies. Each runs in its own virtual environment and
Python version and communicates with emboviz core over a msgpack/ZeroMQ socket.
These environments are created automatically on first use.

> π0's attention diagnostic needs the PyTorch-converted checkpoint:
> `emboviz convert-pi0 pi0_libero` (one-time).

---

## Supported models

| Model | Inference | Attention | Notes |
|---|---|---|---|
| **OpenVLA-7B** | ✅ | ✅ | Full mechanistic-interp surface (hidden states, FFN, patching, ablation). |
| **OpenVLA-OFT** | ✅ | ✅ | Multi-stream (primary + wrist). |
| **π0 / π0.5** | ✅ | ✅ | Attention needs `emboviz convert-pi0`. |
| **GR00T-N1 / N1.7** | ✅ | ⚠️ | Attention is the DiT motor pathway — dispersed, not a tight object localizer (see below). |

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

## Running an analysis

One run is one config file — model, dataset mapping, and analysis parameters in
one place. Templates live in `configs/` (one per model on its canonical dataset).

```bash
emboviz analyze --config configs/openvla-bridge.yaml
emboviz analyze --config configs/my-run.yaml --dry-run   # cost estimate, no GPU
```

To analyze your own checkpoint and data, copy the closest template and edit it:

```yaml
model:
  adapter: openvla                  # openvla | oft | pi0 | gr00t
  kwargs:
    hf_repo: your-org/your-finetune
    unnorm_key: bridge_orig

dataset:
  format: lerobot                   # lerobot | gr00t
  path: your-org/your-dataset       # HF repo id or local dir
  cameras:
    primary: observation.images.image_0
  state:    {key: observation.state, convention: ee_pose}
  action:   {key: action}
  gripper:  {source: 6, kind: parallel_jaw, units: unit, range: [0.0, 1.0]}
  instruction: {from: tasks}

analysis:
  episodes: "537"                   # "7" / "0,3,7" / "0-5" / "all"
  mask_query: "the cloth"           # object the memorization diagnostic masks
  detector: sam3                    # sam3 | gd-sam
  diagnostics: all                  # or [memorization, attention]

output: ./report/my-run
```

The schema is identical for every input format — only the reader behind each
`key` changes. Dimensions and per-dim names are read from the dataset's own
schema; you declare only what the format can't encode (camera roles, state
convention, gripper spec). See `configs/README.md` for the full field reference.

### What you get back

Per episode, in `report/episode_<idx>/`:

- **`summary.json`** — every metric, with the per-frame numbers.
- **`report.md` / `report.html`** — plain-English findings, worst-first.
- **`rollout.rrd`** — open in Rerun: scrub frame-by-frame with attention
  heatmaps, memorization mask + per-fill overlays, per-modality response
  timelines, occlusion grids, and action plots.

Across all analyzed episodes, at the top of `report/`:

- **`aggregate.{json,md,html}`** — cross-episode patterns, linked to per-episode
  pages.

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
| `sam3` | [Meta Segment Anything 3](https://huggingface.co/facebook/sam3) | SAM License — source-available, permits commercial use with restrictions; not OSI open-source. The `--detector gd-sam` alternative uses GroundingDINO and SAM 2, both Apache 2.0 |

Datasets read through the LeRobot and GR00T-format readers (e.g. Open
X-Embodiment, LIBERO, DROID, BridgeData) each carry their own license and terms;
consult the dataset's own documentation.

The license identifications above are provided for convenience and may change
upstream. The authoritative terms are those distributed with each model and
dataset.
