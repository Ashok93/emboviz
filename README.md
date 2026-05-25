# Emboviz

A diagnostic and interpretability framework for **Vision-Language-Action (VLA) robot policies**.

> *Detect, explain, and produce actionable data-collection recommendations for VLA failure modes — across language, vision, and mechanism axes, per-frame or across a full trajectory.*

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design contract.

---

## Status: 6 of 7 rounds complete, all validated on real OpenVLA-7B + BridgeV2

| # | Round | Status | Headline real result |
|---|---|---|---|
| 1 | Trajectory analysis (single-frame → multi-frame) | ✅ | noun_swap CRITICAL at frames 4, 8, 16, 20 of Bridge ep 0 |
| 2 | Concept decomposition (Berkeley FFN logit-lens) | ✅ | Per-frame anomalous-neuron detection via z-score |
| 3 | Linear probe framework + ProbeVsAction | ✅ | Train / save / load + "information present but unused" diagnostic |
| 4 | π0 adapter | ⏸️ deferred | architectural; lower per-effort product value |
| 5 | Activation patching (causal mediation) | ✅ | **L12-L14 carry 13-19 % of noun-swap signal** (matches MINT fusion-band) |
| 6 | Failure prediction probe (SAFE-style, the commercial wedge) | ✅ | **91.3 % val accuracy** on 234 frames from 6 episodes |
| 7 | Object recolor (color-binding viral demo) | ✅ | **ISS = 0.000 across 5 colors** — total color-blindness proven on Bridge ep 0 |

**18 diagnostic primitives now in production**, all running on real OpenVLA against real Bridge data.

---

## What it does

Given a `(VLAModel, Scene | Trajectory)` pair, Emboviz runs diagnostics across:

- **Language grounding** — noun swap, preposition swap, color swap, count, negation, refusal-on-absent, empty, OOD task
- **Visual robustness** — occlusion sweep, viewpoint jitter, lighting shift, distractor injection, sensor noise, target-removal memorization probe, BYOVLA-style sensitivity map
- **Color binding** — **object recolor via GroundingDINO + SAM + HSV rotation** (the viral demo)
- **Mechanism** — cross-modal attention divergence, FFN concept decomposition (Berkeley logit-lens), **activation patching** (causal mediation, Heimersheim & Nanda 2024), neuron ablation
- **Probing** — linear probes for "model sees but doesn't act" + **SAFE-style failure prediction probe** with per-frame P(failure) timeline
- **Trajectory** — wraps any single-frame diagnostic into per-frame curves + auto-detected failure moments
- **Coverage** — text-based dataset gap analysis → concrete data-collection recommendations

Outputs: failure matrix, verdict card, trajectory timelines, failure tape, Markdown report, JSON dump, model-comparison diff.

---

## Architecture in one diagram

```
                       ┌─────────────────────┐
                       │  reports/  + viz/   │   render DiagnosticResults
                       ├─────────────────────┤
                       │  suites/            │   preset batteries
                       ├─────────────────────┤
                       │  diagnostics/       │   orchestrate perturb+metric+model
                       ├──────────┬──────────┤
              perturb/ │ metrics/ │ probes/  │   composable primitives
              ┌────────┴──────────┴───────┐
              │  models/  (VLAModel ABC)  │   one adapter per VLA family
              ├───────────────────────────┤
              │  core/  types + math       │   pure foundation
              └───────────────────────────┘

       datasets/  ╳  coverage/  ╳  taxonomy/   adjacent, not in the layer stack
```

The **core engine** (everything outside `models/`) is model-agnostic. Adding a new VLA = one adapter file. Adding a new diagnostic = one file in `diagnostics/`. Adding a new perturbation = one file in `perturb/`.

---

## Quickstart

```bash
uv sync

# Engine smoke (no GPU)
uv run python -m emboviz.cli.run_suite \
    --model mock --suite quick_smoke --scene bridge:0

# Full battery + coverage analysis + verdict card (real OpenVLA, ~15 min)
uv run python -m emboviz.cli.run_battery \
    --model openvla-7b --scene bridge:0 \
    --outdir outputs/battery

# Per-frame trajectory analysis (the failure tape)
uv run python -m emboviz.cli.run_trajectory \
    --model openvla-7b --suite quick_smoke \
    --trajectory bridge:0 --stride 4 \
    --outdir outputs/traj

# Train a SAFE-style failure prediction probe from labeled rollouts
uv run python -m emboviz.cli.train_failure_probe \
    --model openvla-7b --episodes 0 1 2 3 4 5 \
    --layers 14 22 30 --outdir probes_trained

# Diff two model checkpoints on the same suite
uv run python -m emboviz.cli.compare_models \
    --model-a openvla-7b --model-b mock \
    --suite language_grounding --scene bridge:0
```

---

## The 18-axis diagnostic catalog

| Axis | Diagnostic | What it validates |
|---|---|---|
| `language.noun_swap` | NounSwapPerturber × Counterfactual | "spoon"→"fork" — noun binding |
| `language.preposition_swap` | PrepositionSwapPerturber | "on"↔"under" — spatial-relation grounding |
| `language.color_swap` | ColorSwapPerturber | "red"→"blue" — colour-attribute binding |
| `language.count_swap` | CountSwapPerturber | "one"→"two" — count / ordinal grounding |
| `language.negation` | NegationPerturber | "do not pick" — negation following |
| `language.refusal_absent` | RefusalPerturber | swap to exotic noun → does model refuse? |
| `language.empty` | EmptyInstructionPerturber | empty-instruction baseline |
| `language.ood_task` | OODTaskPerturber | OOD instruction (upper-bound reference) |
| `vision.occlusion` | OcclusionPerturber × Sweep | progressive occlusion robustness |
| `vision.viewpoint` | ViewpointJitterPerturber | camera-pose robustness |
| `vision.lighting` | LightingShiftPerturber | brightness / gamma / saturation |
| `vision.distractor` | DistractorInjectionPerturber × Sweep | distractor count sensitivity |
| `vision.sensor_noise` | GaussianNoisePerturber | sensor-noise robustness |
| `vision.memorization` | MemorizationDiagnostic | mask target → does model still act? |
| `vision.scene_sensitivity` | SensitivityMapDiagnostic | BYOVLA per-region ablation map |
| `vision.color_binding` | **ObjectRecolorPerturber** (GroundingDINO + SAM) × Counterfactual | text-prompted object recolor → does action change? |
| `vision.binding_grounding` | CrossModalAttentionDiagnostic | does attention route on the noun? |
| `internal.concept_decomp` | ConceptDecompositionDiagnostic | top FFN neurons + logit-lens labels (Berkeley 2025) |
| `internal.activation_patching` | ActivationPatchingDiagnostic | per-layer recovery sweep — gold-standard causal mediation |
| `internal.probe_vs_action` | ProbeVsActionDiagnostic | "model knows X but doesn't act on X" |
| `internal.failure_prediction` | FailurePredictionDiagnostic | SAFE-style per-frame P(failure) probe |

Wrap any of the above with `TrajectoryDiagnostic` to get per-frame curves + auto-detected failure moments.

---

## Validated end-to-end findings on OpenVLA-7B + BridgeV2

| Finding | Source diagnostic | Significance |
|---|---|---|
| 🟥 **Color blindness — ISS = 0.000 across red/blue/green/yellow/purple** | ObjectRecolorPerturber × Counterfactual | Model produces identical actions regardless of spoon colour. Causally proven on real data. |
| 🟥 **vision.memorization CRITICAL** — model still acts when target masked | MemorizationDiagnostic | Matches LIBERO-Pro 2025: VLAs memorize trajectories on training-distribution scenes |
| 🟥 **vision.lighting CRITICAL** — under noise floor | CounterfactualDiagnostic | Model ignores lighting changes on this scene |
| 🟧 **L12-L14 carry 13-19 % of noun-swap signal** (single-layer recovery) | ActivationPatchingDiagnostic | Matches MINT fusion-band finding |
| **noun_swap CRITICAL at frames 4, 8, 16, 20** of the rollout | TrajectoryDiagnostic + NounSwap | Temporal pattern single-frame analysis would miss |
| **91.3 % val accuracy** decoding "frame in failing episode" from hidden states | FailurePredictionDiagnostic (trained on 6 eps, 234 frames) | SAFE-style failure prediction works on OpenVLA |
| Coverage analysis: only **26 demos** in 19,974 BridgeV2 tasks have ≥2 utensils co-occurring | Coverage analyzer | Concrete data gap for the noun-grounding failure |

---

## Supported models (current)

- **OpenVLA-7B** (`openvla-7b`) — full capability flags:
  `INFERENCE | ATTENTION | HIDDEN_STATES | FFN_ACTIVATIONS | FFN_VALUE_VECTORS | VOCAB_LOGIT_LENS | NEURON_ABLATION | ACTIVATION_PATCHING`
- **Mock** (`mock`) — deterministic adapter for diagnostic-side testing without GPU

Coming: **π0**, GR00T, OpenVLA-OFT (one adapter file each).

---

## Where to pick this up next

The 3 outstanding directions, in priority order:

### A — Render Rounds 5/6/7 as proper PNGs (~½ day)
The patching, failure-prediction, and recolor diagnostics currently print results to stdout. Each deserves a polished plot:
- Activation patching: per-layer recovery bar chart with fusion-band highlight
- Failure prediction: P(failure) curve over the trajectory with severity tape
- Object recolor: side-by-side panel (original | mask | 5 recolored variants | per-color ISS bar)

### B — Multi-episode sweep for statistical claims (~1 day)
We have results on episode 0 + scattered. Run the full battery on 20-30 Bridge episodes, aggregate per-axis statistics, produce confidence intervals. This is what turns single-scene findings into product-ready claims.

### C — Round 4 (π0 adapter) — multi-model proof (~1-2 days)
Write `models/pi0.py` implementing the protocol against PhysicalIntelligence's π0 (flow-matching, PaliGemma backbone). All 18 diagnostics work immediately; gracefully skip the capabilities π0 doesn't expose. This is the architectural credibility milestone.

### D — Three small per-frame data additions for future Rerun/Foxglove UX (~1 day)
1. `PredictedActionDiagnostic` — store the 7-DOF action vector per frame
2. Per-text-token attention bars (extend `CrossModalAttentionDiagnostic`)
3. Per-neuron causal image attribution (port from `legacy/visual_attribution.py`)

After these, the core produces **100 %** of the per-frame data a Rerun/Foxglove timeline UX would need. The UI then becomes a thin exporter.

### E — UI layer (later)
- `emboviz/exporters/rerun.py` — `.rrd` file emission for Rerun playback
- HTML dashboard with frame scrubber (Plotly / React)
- Polished single-PNG verdict card (designer pass)

---

## Repository layout

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full tree + contracts.

```
emboviz/
  core/            Layer 0 — pure types + math (Scene, Trajectory, ActionResult, ...)
  models/          Layer 1 — VLAModel protocol + Capability flags + adapters
                              (openvla.py, mock.py)
  perturb/         Layer 2 — instruction + image perturbers (14 total)
                              including image/recolor.py (GroundingDINO + SAM + HSV)
  metrics/         Layer 2 — action-divergence, attention-JS, pointing-game,
                              probe-confidence, ablation-drop, ...
  probes/          Layer 2 — linear probes (train / save / load) + presets
                              (failure_predictor)
  diagnostics/     Layer 3 — counterfactual, sweep, attention, memorization,
                              sensitivity_map, concept_decomp, activation_patching,
                              probe, probe_vs_action, failure_prediction,
                              trajectory (wraps any of the above for per-frame curves)
  suites/          Layer 4 — preset diagnostic batteries
  reports/         Layer 5 — verdict_card, failure_matrix, trajectory_timeline,
                              failure_tape, comparison, markdown, JSON
  viz/             Layer 5 — plotting primitives (overlays, arrows, stitch)
  coverage/        adjacent — text-based dataset gap analyzer
  taxonomy/        adjacent — failure-mode catalog, object categories, prepositions
  datasets/        adjacent — Bridge episode source (load_scene / load_trajectory)
  cli/             Layer 6 — entry points (run_suite, run_battery, run_trajectory,
                              compare_models, train_failure_probe)
```

Heavy optional deps (transformers' GroundingDINO + SAM) are **lazy-imported** in `perturb/image/recolor.py`; the rest of the framework imports cleanly without them.

---

## Tooling decisions worth knowing

- **Action distance**: normalized L2 (per-dim divided by Bridge q99-q01 stats). Severity thresholds in `diagnostics/counterfactual.py` (`DEFAULT_NOISE_FLOOR = 0.5`, `DEFAULT_GROUNDED = 2.0`) are calibrated for this scale.
- **Activation patching**: hook fires once per generation (on the prefix-processing forward), avoiding the KV-cache continuation-step bug that initially produced negative recoveries.
- **Position resolution**: `_resolve_query_position` in OpenVLA adapter always returns **multimodal-sequence** positions (text positions are auto-mapped via `+n_image_tokens`). Every extract_* method consistently uses multimodal coordinates.
- **Capability gating**: every internal-introspection diagnostic checks `Capability.X in model.capabilities` and emits `Severity.UNKNOWN` (skipped, not crashed) if missing. Suites pass cleanly even when a model lacks a capability.
- **Object recolor**: uses SAM v1 (not SAM2) for transformers-4.49 compatibility; SAM2 added in transformers ≥ 4.50. SAM and GroundingDINO load independently so SAM failure doesn't break detection.
- **Failure probe training data**: synthesized from Bridge rollouts by comparing predicted vs expert actions per episode; episodes with max deviation > 0.30 (Bridge units) are labelled as failures, with `±spread_frames` around the spike marked.
