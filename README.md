# Emboviz

**Diagnostic and interpretability framework for embodied-AI policies.**

> Drop in a rollout. Get back rich per-frame diagnostic data, playable in Rerun or Foxglove, that shows you exactly when and why your policy fails — across language, vision, state, and mechanism axes.

![status](https://img.shields.io/badge/status-alpha-orange) ![license](https://img.shields.io/badge/license-Apache%202.0-blue) ![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

Emboviz turns the black box of your robot policy into a debugger. Whether you're running OpenVLA, π0, GR00T, ACT, Diffusion Policy, or your own fine-tune — on a Franka, UR, ALOHA, or humanoid — Emboviz surfaces the evidence your model leaves behind and lets you see it inside the playback tools you already use.

**Debugger, not oracle.** We surface signals; you form conclusions.

---

## Why this exists

Robot policies are getting bigger, more capable, and more opaque. Teams ship a VLA, get a 60% success rate, and have no principled way to find out what's wrong. Is the model ignoring language? Memorizing trajectories? Color-blind on the target object? Not using gripper state at all? Today the answer is "run more rollouts and guess." Emboviz answers it in one command.

---

## Quickstart

```bash
# Install
uv add emboviz

# Diagnose a rollout (auto-detects format)
uv run emboviz diagnose ./episode.bag --model openvla-7b --profile franka_robotiq

# Open the results in Rerun (or Foxglove)
rerun ./out/diagnostics.rrd
```

You get back:

- **Rerun `.rrd`** with every diagnostic emitted as a toggleable timeline track — per-frame attention heatmaps overlaid on your camera, severity tape per axis, P(failure) curve, predicted-vs-expert action vectors.
- **Foxglove `.mcap`** with the same data as topics.
- **A scorecard PNG** — axis-by-axis severity grid for at-a-glance triage.
- **Per-diagnostic detail pages** — drill into one axis without wading through the rest.

No prose synthesis, no "we think your model is broken because…" — just evidence, in the tools you already use, scrubbable frame by frame.

---

## What's supported

### Model adapters (one file per model)

| Family | Status |
|---|---|
| OpenVLA-7B | ✅ shipped |
| OpenVLA-OFT | 🚧 in progress |
| π0 / π0.5 | 🚧 in progress (gated on public weights) |
| GR00T-N1 | 🚧 in progress |
| ACT | 🚧 in progress |
| Diffusion Policy | 🚧 in progress |
| RDT-1B | 🚧 in progress |
| Octo | 🚧 in progress |
| Mock (no GPU) | ✅ shipped — for diagnostic-side dev |

### Robot profiles (preshipped configs)

| Robot | Status |
|---|---|
| Franka Panda + Robotiq 2F-85 | 🚧 in progress |
| UR5 / UR10 + Robotiq | 🚧 in progress |
| Trossen (ALOHA single arm) | 🚧 in progress |
| ALOHA bimanual | 📅 roadmap |
| Unitree H1 / G1 | 📅 roadmap |

Custom robots: write a ~30-line YAML profile + (optional) custom adapter.

### Data formats

| Format | Ingest | Export |
|---|---|---|
| LeRobot v3 (BridgeV2, etc.) | ✅ | — |
| Rerun `.rrd` | 🚧 | 🚧 **(killer feature)** |
| Foxglove `.mcap` | 🚧 | 🚧 |
| ROS bag (mcap) | 🚧 | — |
| HuggingFace dataset (generic) | 🚧 | — |
| RLDS | 🚧 | — |

---

## The diagnostic catalog (18 axes)

| Axis | What it tests |
|---|---|
| **Language** | noun swap, preposition swap, color swap, count swap, negation, refusal on absent, empty instruction, OOD task |
| **Vision** | occlusion sweep, viewpoint jitter, lighting shift, distractor injection, sensor noise, target-removal (memorization probe), per-region sensitivity map, **object recolor via GroundingDINO + SAM** (the color-binding test) |
| **State / proprio** | gripper flip (the "dropped item" test), state jitter, action-history ablate, action-history scramble |
| **Mechanism** | cross-modal attention divergence, FFN concept decomposition (Berkeley logit-lens), **activation patching** (causal mediation), neuron ablation |
| **Probing** | linear probes for "model sees but doesn't act," SAFE-style per-frame P(failure) probe |
| **Trajectory** | wraps any single-frame diagnostic into per-frame curves + auto-detected failure moments |
| **Coverage** | text-based dataset gap analysis → concrete data-collection recommendations |

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
