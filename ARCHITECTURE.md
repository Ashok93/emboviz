# Emboviz — Architecture

## Goal

A **diagnostic and interpretability framework for robot policies** (VLAs and successor architectures) where:

- The **core engine** — algorithms, perturbations, metrics, diagnostics, reporters — is model-agnostic.
- The **per-model adapters** are the only place model internals leak in.
- Adding a new VLA = one adapter file. Adding a new diagnostic technique = one file in `core/`. Adding a new visualization = one file in `viz/`.
- The interpretability research field moves fast; new techniques emerge monthly. The architecture must absorb new methods without refactors.

## Design principles

1. **Separation of concerns** — models, perturbations, metrics, diagnostics, coverage, viz, datasets are independent.
2. **Composition over inheritance** — a "diagnostic" is a recipe `(Perturber, Metric, Runner)`, not a class hierarchy.
3. **Capability-based interfaces** — adapters declare what they support; diagnostics check before running.
4. **Pure functions in core** — side-effects (I/O, GPU work) live in adapters, datasets, and reporters only.
5. **Optional heavy deps** — captum, SAM2, InstructPix2Pix import lazily and only when needed.
6. **Type-safe + dataclass-heavy** — Protocols, ABCs, dataclasses; minimum cleverness.
7. **Uniform result shape** — every diagnostic returns a `DiagnosticResult` so reporters and dashboards consume one schema.
8. **Reproducibility** — every diagnostic accepts a seed; results are hashed.

## Layers

```
                         ┌─────────────────────┐
                         │  Reporters / Viz    │  Layer 5: render results
                         ├─────────────────────┤
                         │  Suites             │  Layer 4: preset diagnostic batteries
                         ├─────────────────────┤
                         │  Diagnostics        │  Layer 3: orchestrate perturb + metric + runner
                         ├─────────────────────┤
            ┌────────────┼─────────┬───────────┤
            │ Perturbers │ Metrics │ Probes    │  Layer 2: primitives, composable
            ├────────────┴─────────┴───────────┤
            │  VLAModel Protocol               │  Layer 1: interface
            ├──────────────────────────────────┤
            │  Adapters: openvla, pi0, ...     │  Layer 1: model-specific impls
            ├──────────────────────────────────┤
            │  core/types, distances           │  Layer 0: pure data + math
            └──────────────────────────────────┘

                    Datasets, Coverage, Taxonomy live alongside,
                    not as a layer
```

## Directory layout

```
emboviz/
├── core/                     # Layer 0 — pure foundation
│   ├── types.py              # Scene, Action, ActionResult, AttentionMaps, HiddenStates, ...
│   ├── distances.py          # action distance metrics
│   ├── divergences.py        # JS, KL, Spearman
│   ├── seeding.py
│   └── results.py            # DiagnosticResult schema
│
├── models/                   # Layer 1 — model abstraction + adapters
│   ├── protocol.py           # VLAModel ABC + Capability flags
│   ├── registry.py           # name → adapter factory
│   ├── openvla.py            # OpenVLAAdapter
│   ├── pi0.py                # future
│   └── mock.py               # MockVLA for testing diagnostics without GPU
│
├── perturb/                  # Layer 2 — perturbations
│   ├── base.py               # Perturber Protocol + PerturbedScene type
│   ├── instruction/
│   │   ├── noun_swap.py
│   │   ├── preposition_swap.py
│   │   ├── color_swap.py
│   │   ├── count_swap.py
│   │   ├── negation.py
│   │   ├── refusal.py
│   │   ├── empty.py
│   │   └── ood_task.py
│   └── image/
│       ├── occlusion.py            # patch occlusion (no models)
│       ├── viewpoint.py            # homography proxy
│       ├── lighting.py             # gamma / HSV / CLAHE
│       ├── distractor.py           # paste colored rect / inpaint
│       ├── target_remove.py        # mask-out target region
│       ├── recolor.py              # OPTIONAL: SAM+IP2P
│       └── noise.py                # gaussian noise / sensor sim
│
├── metrics/                  # Layer 2 — metrics
│   ├── base.py               # Metric Protocol
│   ├── action_divergence.py
│   ├── instruction_sensitivity.py
│   ├── attention_js.py
│   ├── pointing_game.py
│   ├── ablation_drop.py
│   └── probe_confidence.py
│
├── probes/                   # Layer 2 — linear probes on hidden states
│   ├── base.py
│   ├── trainer.py
│   ├── store.py              # save / load
│   └── presets/
│       ├── object_presence.py
│       ├── object_color.py
│       └── object_position.py
│
├── diagnostics/              # Layer 3 — orchestration
│   ├── base.py               # Diagnostic ABC + DiagnosticResult
│   ├── counterfactual.py     # perturb + measure
│   ├── sweep.py              # parametric sweep (occlusion %, distractor count, ...)
│   ├── attention.py          # cross-modal attention diagnostics
│   ├── sensitivity_map.py    # BYOVLA-style per-region ablation
│   ├── memorization.py       # target-removed-still-acts test
│   ├── concept_decomp.py     # FFN logit-lens analysis
│   ├── ablation.py           # neuron-ablation diagnostics
│   └── probe.py              # probe-based confidence
│
├── suites/                   # Layer 4 — preset batteries
│   ├── base.py               # Suite class
│   ├── language_grounding.py # ALL language-axis diagnostics
│   ├── visual_robustness.py  # ALL vision-axis diagnostics
│   ├── full_profile.py       # everything
│   └── quick_smoke.py        # minimal smoke test
│
├── coverage/                 # Adjacent — dataset coverage
│   ├── text_analyzer.py
│   └── gap_detector.py
│
├── taxonomy/                 # Adjacent — canonical lists
│   ├── failure_modes.py
│   ├── object_categories.py
│   └── spatial_prepositions.py
│
├── datasets/                 # Adjacent — dataset adapters
│   ├── base.py               # EpisodeSource Protocol
│   ├── lerobot_bridge.py
│   └── custom.py
│
├── viz/                      # Layer 5 — plotting primitives
│   ├── overlays.py
│   ├── arrows.py
│   ├── bars.py
│   ├── grids.py
│   └── stitch.py             # PIL section stitching
│
├── reports/                  # Layer 5 — high-level reporters
│   ├── base.py               # Reporter Protocol
│   ├── verdict_card.py       # the killer single-PNG output
│   ├── failure_matrix.py     # per-axis grid
│   ├── markdown.py
│   ├── json_export.py
│   └── comparison.py         # model-vs-model diff
│
└── cli/                      # Layer 6 — user entry points
    ├── run_diagnostic.py
    ├── run_suite.py
    ├── run_battery.py
    ├── compare_models.py
    └── train_probes.py
```

## Key contracts

### `VLAModel` (in `models/protocol.py`)

```python
class Capability(Flag):
    INFERENCE = auto()
    ATTENTION = auto()
    HIDDEN_STATES = auto()
    FFN_ACTIVATIONS = auto()
    NEURON_ABLATION = auto()
    PROBABILITY_OUTPUT = auto()
    GRADIENT = auto()

class VLAModel(ABC):
    model_id: str
    capabilities: Capability

    def predict(image, instruction) -> ActionResult
    def predict_with_image(perturbed_image, instruction) -> ActionResult
    def extract_attention(image, instruction, query: TokenSelector) -> AttentionMaps
    def extract_hidden_states(image, instruction, layers) -> HiddenStates
    def extract_ffn_activations(image, instruction, layers) -> dict[int, Tensor]
    def predict_with_neuron_ablation(image, instruction, ablations) -> ActionResult
    def find_token_positions(prompt, word) -> list[int]
    def compare_actions(a, b) -> float    # adapter-defined distance
```

Adding π0 = implement these methods on a `Pi0Adapter` class. All diagnostics work immediately.

### `Perturber` (in `perturb/base.py`)

```python
class Perturber(Protocol):
    name: str
    axis: str           # category for grouping
    domain: Literal["instruction", "image", "joint"]

    def variants(scene: Scene) -> Iterable[PerturbedScene]
```

### `Metric` (in `metrics/base.py`)

```python
class Metric(Protocol):
    name: str

    def compute(baseline: ActionResult, perturbed: ActionResult) -> float
    # or
    def compute_pair(model, scene_a, scene_b) -> float
```

### `Diagnostic` + `DiagnosticResult`

```python
class Diagnostic(ABC):
    name: str
    axis: str
    required_capabilities: Capability

    def run(model: VLAModel, scene: Scene) -> DiagnosticResult

@dataclass
class DiagnosticResult:
    diagnostic_name: str
    axis: str
    scalar_score: float
    severity: Literal["pass", "moderate", "critical"]
    per_variant: dict[str, float]
    explanation: str
    recommendation: Optional[str]
    raw: dict                # diagnostic-specific data for viz
    metadata: dict
```

Reporters and dashboards consume `DiagnosticResult` only.

### `Suite`

```python
@dataclass
class Suite:
    name: str
    diagnostics: list[Diagnostic]

    def run(model, scene) -> dict[str, DiagnosticResult]
```

## How a new technique gets added

If a new paper drops next month — say "VLA-Watch" (made up) — the integration path is:

- **New perturbation?** → add a file in `perturb/instruction/` or `perturb/image/`.
- **New metric?** → add a file in `metrics/`.
- **New diagnostic algorithm?** → add a file in `diagnostics/` that composes existing perturbers and metrics, or builds its own primitives.
- **New attention-analysis trick?** → add a file in `diagnostics/attention.py`, calls `model.extract_attention()` only.
- **New SAE / probe?** → add a file in `probes/presets/`.
- **New visualization?** → add a file in `viz/` or `reports/`.

Nothing in the core hierarchy needs to change. New techniques drop in as additional files.

## How a new model gets added

- Write one file in `models/`: a class implementing `VLAModel` with the appropriate `Capability` flags.
- All ~15 existing diagnostics work immediately, gracefully skipping any that need capabilities the new model lacks.

## What this enables for the user

- One command: `emboviz run --model openvla --suite full_profile --episode bridge:0` → 14-axis failure profile + verdict card.
- Compare two checkpoints: `emboviz compare --models ckpt1.bin ckpt2.bin --suite language_grounding`.
- CI integration: any diagnostic emits JSON for regression dashboards.
- Custom workflows: `from emboviz import Diagnostic, Perturber, ...` and compose.

---

This is the contract. Implementation lives below.
