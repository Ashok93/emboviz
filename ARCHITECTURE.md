# Emboviz — Architecture

## Goal

A model-agnostic diagnostic / interpretability framework for VLA robot
policies. The engine — datasets, perturbers, metrics, diagnostics,
exporters — knows nothing about any specific model; each VLA family is a
self-contained adapter package. Adding a model is one adapter package;
adding a diagnostic or a dataset format is one file.

## Two processes, one wire

The hard constraint is dependency conflict: OpenVLA pins transformers
4.40–4.49, OFT a vendored transformers fork, π0 transformers 4.53, GR00T
4.57, SAM 3 ≥4.56 — none coexist in one venv. So emboviz runs as **two
kinds of process**:

```
 host venv (emboviz)                 isolated runtime venvs (~/.emboviz/venvs/<name>)
 ┌───────────────────────┐           ┌──────────────────────────────────────┐
 │ emboviz core          │  msgpack  │ emboviz-openvla  → torch + transformers│
 │  + adapter shims      │   over    │ emboviz-oft      → moojink fork        │
 │  + reader shim        │ ZMQ/UDS   │ emboviz-pi0      → openpi              │
 │  + emboviz-wire       │ ◀───────▶ │ emboviz-gr00t    → Isaac-GR00T         │
 │  (NO torch/lerobot)   │  (bytes)  │ emboviz-sam3     → SAM 3 (Python 3.12) │
 │                       │           │ emboviz-lerobot  → lerobot (dataset    │
 │                       │           │                    reader, v2.x)       │
 └───────────────────────┘           └──────────────────────────────────────┘
        DEALER                                      ROUTER (one per worker)
```

* The **host** holds the engine (diagnostics, exporters, orchestration) and
  a thin shim per installed adapter AND per installed dataset reader. It
  carries **no torch and no lerobot** — lerobot's transitive `rerun-sdk<0.27`
  pin would collide with the host's own `rerun>=0.32` `.rrd` exporter, so
  the LeRobot reader is isolated like a model (see below).
* Each **adapter worker** loads one model in its own venv (its own Python
  version, its own torch/transformers pins) and serves it over a ZMQ
  ROUTER on a Unix socket.
* The **LeRobot dataset reader** is isolated the SAME way: `emboviz-lerobot`
  is a thin host shim; its `lerobot` install lives in its own reader venv
  and serves universal `Scene`/`Trajectory` objects over the wire (the
  `EpisodeSource` contract) — the dataset-side mirror of a model worker.
  HDF5/RLDS readers (no conflicting pins) stay in-process in the host.
* The wire is **msgpack over ZeroMQ** — bytes, not pickle — so the two
  sides may run different Python / numpy / transformers versions. This is
  the `emboviz-wire` package, the only emboviz package a worker installs.

`emboviz analyze` resolves an adapter alias to its `AdapterSpec`,
attaches to a running worker (or spawns one in its runtime venv and waits
for `ping`), and drives it through a `VLAModel`-shaped client. Diagnostics
never import a worker's model code; they call the `VLAModel` protocol.

## Packages

```
emboviz-wire/            the shared contract (host AND every worker install it)
  types                  Scene, Observations, ActionResult, AttentionMaps, Trajectory, ...
  observations/          RGBImage, Proprioception, GripperState, ... (typed, unit-aware)
  profile                RobotProfile (cameras, state convention, gripper, action)
  model_protocol         VLAModel ABC + Capability flags + RequiredInputs (model side)
  reader_protocol        EpisodeSource ABC (dataset side — mirror of VLAModel)
  dataset_build          build_profile / make_gripper_extractor (shared by all readers)
  wire                   msgpack encode/decode for every wired type (Scene carries profile)
  client / server        ZMQ DEALER clients (ZMQAdapterClient, ZMQReaderClient) +
                         ROUTER serve() loop + VLAModelHandler + DatasetReaderHandler
  handler                AdapterSpec (what a worker venv needs + how to launch it)

adapters/emboviz-<name>/ one per backend: a shim (AdapterSpec + server entry
                         point) the host installs, plus the heavy code that
                         runs only inside the runtime venv.
                         openvla · oft · pi0 · gr00t (VLAs, group emboviz.adapters)
                         sam3 (detector) · lerobot (dataset reader, group emboviz.readers)

emboviz/                 the host engine (no model deps)
  core/                  re-exports emboviz_wire types + DiagnosticResult/Finding,
                         divergences, seeding
  adapters/              registry + reader_registry (entry-point discovery for
                         emboviz.adapters / emboviz.readers) + lifecycle
                         (venv install, connect / connect_reader worker spawn) + shims
  config.py              RunConfig — one YAML file describes a whole run
  calibration.py         per-trajectory noise-floor + typical-action anchors
  diagnostics/           the algorithms (one file each)
  perturb/               instruction / image / state perturbers
  metrics/               action divergence, attention JS, pointing game, ...
  datasets/              manifest builder (lerobot → reader worker; hdf5/rlds in-process)
  exporters/             Rerun .rrd writer (export_rerun) + failure-moment correlation
  models/                mock (no GPU) + the in-process model registry
  probes/ taxonomy/      linear failure-probe training/store + failure-mode taxonomy
  _internal/             runner (run_story) + multi-episode aggregation + report.md/html
  cli/                   analyze · list-models · list-datasets · version
                         · install-<adapter> · convert-pi0
```

## Key contracts

### `VLAModel` (`emboviz_wire.model_protocol`)

Every adapter implements this; diagnostics call only this.

```python
class Capability(Flag):
    INFERENCE; PROBABILITY_OUTPUT; ATTENTION; HIDDEN_STATES
    FFN_ACTIVATIONS; FFN_VALUE_VECTORS; VOCAB_LOGIT_LENS
    NEURON_ABLATION; GRADIENT; BATCH_INFERENCE; CHUNK_PREDICTION
    ACTIVATION_PATCHING

class VLAModel(ABC):
    model_id: str
    capabilities: Capability         # what the model can EXPOSE
    required_inputs: RequiredInputs  # what it must CONSUME from a Scene
    action_dim: int

    def predict(scene) -> ActionResult
    def extract_attention(scene, query) -> AttentionMaps          # ATTENTION
    def extract_hidden_states(scene, layers, query) -> HiddenStates
    def extract_ffn_activations(scene, layers, query) -> FFNActivations
    def predict_with_residual_patch(scene, patches, pos) -> ActionResult
    def predict_with_neuron_ablation(scene, ablations) -> ActionResult
    def find_token_positions(instruction, word) -> list[int]
    def compare_actions(a, b) -> float
```

`RequiredInputs` declares which cameras + modalities the model reads; the
runner validates a `Scene` against it before `predict`, and perturbers
auto-skip modalities the model doesn't consume. Capability-gated methods
raise `NotSupported` by default, so a diagnostic that needs `ATTENTION`
skips cleanly on a model that lacks it.

### `AdapterSpec` (`emboviz_wire.handler`)

One per adapter package, registered via the `emboviz.adapters`
entry-point group. Declares the runtime venv's Python version, its
`runtime_pip` (the heavy deps), env vars, and the `server_module` that
launches the worker. The host reads it without importing any model code.

### `DiagnosticResult` + `Finding` (`emboviz.core.results`)

```python
@dataclass
class DiagnosticResult:
    diagnostic_name; axis; model_id; scene_id
    scalar_score: float
    severity: Severity          # INTERNAL sort key — never rendered to users
    direction: "lower_is_worse" | "higher_is_worse"
    finding: Finding            # plain-English verdict (observed/meaning/next_step)
    per_variant: dict; raw: dict; metadata: dict
```

`Finding` is the user-facing verdict (three sentences + raw numbers); the
`Severity` enum is only used to sort, filter, and colour. Reports never
print severity words.

### `EpisodeSource` (`emboviz_wire.reader_protocol`)

`list_episodes()`, `load_episode(s)`, `load_trajectory(idx) -> Trajectory`,
`all_instructions()`. The dataset-side contract — mirror of the
`VLAModel` model-side contract, in the same wire package so an isolated
reader worker (which has the wire package, not core) implements it.
One implementation per self-describing format; dims and per-dim names are
read from the format's own schema, never hand-typed:

* **`lerobot`** → the isolated `emboviz-lerobot` worker; the host gets a
  `ZMQReaderClient` (an `EpisodeSource` over the wire) via
  `connect_reader("lerobot", ...)`. Scenes carry their `RobotProfile`
  across the wire so the host has action dim-names + conventions.
* **`hdf5`** / **`rlds`** → in-process in the host (no conflicting pins).

Rerun `.rrd` and MCAP are recording / viz formats, not dataset inputs.

### `Diagnostic`, `Perturber`, `Metric`, `Suite`

A `Diagnostic.run(model, scene) -> DiagnosticResult` composes a
`Perturber` (mutates one input modality) and a `Metric` (scores two
`ActionResult`s). A `Suite` is an ordered list of diagnostics for the
in-process composable path (mock / LeRobot policies).

## The `emboviz analyze` path

```
RunConfig (one YAML)
  └─ model.adapter ──▶ adapters.connect()  ──▶ ZMQ worker (VLAModel client)
  └─ dataset ───────▶ datasets.manifest.build_source() ──▶ EpisodeSource
  └─ analysis ──────▶ _internal.runner.run_story() per episode:
        calibrate ▸ per-frame baseline + attention ▸ the five diagnostics ▸
        summary.json + rollout.rrd + report.md/html
  └─ aggregate across episodes ──▶ aggregate.json / .md / .html
```

The five shipped diagnostics:

| axis | question |
|---|---|
| `vision.memorization` | is the policy looking at the target, or replaying memorized motion? |
| `input.modality_dropout` | which declared inputs does it actually use? (SHAP-marginal) |
| `vision.scene_sensitivity` | which image regions drive the action? (occlusion sweep) |
| `internal.chunk_consistency` | is the multi-step action chunk internally coherent? |
| `internal.attention_drift` | does internal visual attention stay anchored across frames? |

Every threshold is anchored to a per-trajectory calibration (noise floor +
typical action magnitude), and every diagnostic refuses a verdict — with a
stated reason — when its inputs don't meet the method's assumptions.

## How to extend

* **New model** → new `adapters/emboviz-<name>/` package: a `VLAModel`
  subclass + an `AdapterSpec` + a one-line `server.py`. Every applicable
  diagnostic works immediately; the rest skip on capability.
* **New diagnostic** → one file in `emboviz/diagnostics/`, returning a
  `DiagnosticResult`; wire it into `run_story` (the runner).
* **New dataset format** → either a branch in
  `datasets/manifest.build_source` (in-process, for conflict-free readers
  like HDF5/RLDS) or a new isolated reader package + `emboviz.readers`
  entry point (for a heavy/conflicting reader, like `emboviz-lerobot`).
* **New perturber / metric / exporter** → one file in the matching package.

Nothing in the core hierarchy changes; new techniques drop in as files.
