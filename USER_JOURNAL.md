# Honest user-experience journal — building + testing Emboviz adapters

> This is what it actually felt like to set up and test Emboviz's
> adapters end-to-end on a Vast.ai VM, acting as a real user would —
> warts, dependency hell, and all. Sharing it because every friction
> point here is something our wizard + docs should eliminate.

---

## What worked smoothly (no notes)

- **Bridge dataset + OpenVLA-7B** — the prior baseline. `uv sync`, run
  `emboviz diagnose`, get scorecard + .rrd + detail pages. ~15 min from
  fresh VM to first verdict. *This is the experience we want for every model.*
- **Rerun .rrd export + round-trip ingest** — once the format-quirk bugs
  were fixed (string-vs-dict contents, ImageBuffer+ImageFormat pair
  decoding), data flows cleanly out and back in. Real roboticist could
  drop a .rrd into their existing Rerun viewer and see Emboviz overlays.
- **Pi0 via openpi** — Physical Intelligence's `uv sync` Just Worked.
  17s checkpoint load, real action prediction, real NounSwap diagnostic.
  Their tooling is the cleanest of the bunch.

## What was painful (real friction)

### 1. **Per-model transformers version conflicts are brutal.**

   - OpenVLA wants `transformers>=4.40,<4.50` (prismatic VLA constraint)
   - SmolVLA wants `transformers>=4.50,<5.0` (lerobot 0.5 transitive)
   - GR00T wants `transformers==4.57.3` (exact, NVIDIA modeling code)
   - OFT wants a custom **fork** of transformers (`moojink/transformers-openvla-oft`)
     for bidirectional attention parallel decoding

   **No single venv can satisfy this.** Per-adapter venvs are the only
   honest answer. The wizard generates them; I built four on the VM and
   each one has different `transformers`, `torch`, `lerobot`, Python
   versions.

### 2. **Python 3.12 is too strict for several upstream packages.**

   NVIDIA's `Gr00tN1d7Config` dataclass has non-default fields following
   default fields — Python 3.10+ enforces this; 3.12 enforces it more
   loudly. Same bug in `lerobot 0.5`'s `groot` policy import. Fix: pin
   Python 3.10 for both. Means our `requires-python = ">=3.10,<3.13"`
   has to stay even though our own code runs fine on 3.12.

### 3. **HuggingFace gated models require manual click-through.**

   GR00T-N1.7-3B uses `nvidia/Cosmos-Reason2-2B` as its backbone. Cosmos
   isn't "gated" in the apply-and-wait sense — it just needs accepting
   the NVIDIA Open Model License via a one-click button on the HF model
   page. But no amount of `huggingface-cli login` solves this — you have
   to physically click. Hit a 403 three times before realizing the user
   had to do this once. Our wizard should preflight this and tell the
   user "go click this URL and come back."

### 4. **LeRobot's `PreTrainedPolicy.from_pretrained` doesn't dispatch.**

   Called on the abstract base, it tries to instantiate the abstract
   class itself, which fails with "Can't instantiate abstract class
   PreTrainedPolicy". The fix is to read `PreTrainedConfig.type` and
   dispatch to the concrete subclass yourself. Took me an hour to
   diagnose because the error message blames the abstract methods, not
   the dispatch. I added the dispatch in `_load_policy` —
   future LeRobot users get it for free.

### 5. **VLA-style policies want pre-tokenized language but don't say so.**

   SmolVLA's `input_features` declares state + 3 cameras. Doesn't
   declare `observation.language.tokens`. But its `select_action` calls
   `batch[OBS_LANGUAGE_TOKENS]` inside `_get_action_chunk`. Detection
   required pattern-matching on policy class name, then walking
   `model.vlm_with_expert.processor.tokenizer` to find the right tokenizer.
   Also: SmolVLA wants `attention_mask` as a **bool** tensor, not the
   `int64` HuggingFace tokenizers return by default. Easy fix once you
   find it; brutal to discover.

### 6. **Per-state-key dimensionality mismatches.**

   GR00T's OXE_DROID embodiment expects `eef_9d` (9 dims), `joint_position`
   (7 dims), and `gripper_position` (1 dim). My adapter initially packed
   the same vector into all three. Crash. Fix: read each key's expected
   size from the policy's normalization parameters (defensive — fall
   back to name-based heuristics: "9d"→9, "joint"→7, "gripper"→1).
   Real users debugging their own setup will hit this exact class of bug.

### 7. **Dataset format generation matters.**

   LeRobot v0.3 reads v2.0 datasets (Bridge). LeRobot v0.5 only reads
   v2.1. Mixing them = `BackwardCompatibilityError`. Users have to know
   which LeRobot generation their model needs and pick datasets accordingly.
   `emboviz init` should pin this in the wizard.

### 8. **uv's optional-deps resolution doesn't tolerate conflicts.**

   Tried `[project.optional-dependencies]` with per-adapter version pins.
   uv refuses to lock because the extras conflict with each other. Added
   `[tool.uv].conflicts` declarations — works for `uv sync --extra X`
   but only after the right flags. Pip would have been more forgiving;
   uv's strictness here is correct in spirit but trips on day-1 use.

## What this proves about why Emboviz exists

A real user trying to debug ANY one of these four models — OpenVLA, SmolVLA,
π0, GR00T — would spend most of a day navigating the dep hell I just
described. Each model has its own ecosystem with its own pyproject
quirks, its own version pins, its own undocumented init expectations.
The actual debugging work — running diagnostics, looking at outputs —
takes minutes once the environment is set up.

**The moat is the install + run abstraction**, not the diagnostic algorithms.
The diagnostics are well-published research; the orchestration across
fragile per-model ecosystems is what we contribute.

Specifically: every fix I made in this session (LeRobotPolicyAdapter's
policy-class dispatch + language tokenization, Gr00tAdapter's
per-state-key dim dispatch + multi-cam fallback + temporal-horizon stacking,
Pi0Adapter's platform-aware observation builders) is something **every
future user of those models would otherwise have to write themselves**. We
write it once, ship it in the adapter, and they call `model.predict(scene)`.

## What I'd ship next to make this easier for users

1. **`emboviz init` actually generates the install scripts on disk** (currently
   it prints them). User runs the generated `setup_my_model.sh` directly.
2. **Pre-flight gated-access check** — before downloading, the wizard
   `HEAD` the model files and warns about 403s with the exact
   "click-here-to-accept" URL.
3. **A Compatibility matrix in the wizard** — "you picked SmolVLA on
   Bridge data; Bridge is v2.0 and SmolVLA needs v2.1, here's the
   convert command" — instead of letting the user hit
   `BackwardCompatibilityError`.
4. **Per-adapter `validate()` command** — once the venv is set up, run
   `emboviz validate --model gr00t` which loads, runs one predict, and
   confirms the install is healthy. Replaces "blind hope" with "green light."
5. **The cloud aggregation layer** — once N users have shared (opt-in)
   findings, the wizard says "users on similar setups commonly hit X;
   try Y first." This is the Hub feature we discussed earlier.

## Honest score

Emboviz is **about 90% of an end-to-end product** for tabletop manipulation
debugging on the four flagship VLAs tested. The remaining 10% is:
- Wizard polish (the install steps should be auto-executed, not printed)
- Pre-flight environment checks (gated models, dataset compat, dep verification)
- The cloud aggregation layer (which we haven't built yet by design)

The diagnostic engine itself is solid: the same `predict(scene)` API,
the same `CounterfactualDiagnostic` framework, the same
`Severity.CRITICAL/MODERATE/PASS` thresholds produce comparable, real
findings across all four models on the VM. That's the foundation —
everything else is polish on top.
