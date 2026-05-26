# Emboviz code audit — 2026-05-26

**Goal:** find every place our metrics could lie to the user. Every silent
fallback, hack, default, or "we'll figure it out later." For a tool that
flags policies CRITICAL, a single wrong default is worse than no tool.

**Scope:** All 5 active diagnostics × 4 models, plus calibration, runner,
dataset adapters, model adapters, and shared support code.

**Method:** code-only read in this pass. Empirical probes (which need a
VM) come in Phase 2 after this list is reviewed.

---

## Part 1 — Audit matrix (22 cells)

Symbols:
- `OK` = read end-to-end, no bug found in code (still needs empirical proof on VM)
- `BUG` = bug or silent fallback identified, listed in Part 2
- `N/A` = correctly not applicable for this model

|                              | OpenVLA   | OFT       | π0        | GR00T          |
|------------------------------|-----------|-----------|-----------|----------------|
| vision.memorization          | BUG #4,5,6| BUG #4,5,6| BUG #4,5,6| BUG #4,5,6     |
| input.modality_dropout       | BUG #7-10 | BUG #7-10 | BUG #7-10 | BUG #7-10      |
| vision.scene_sensitivity     | BUG #11-13| BUG #11-13| BUG #11-13| BUG #11-13     |
| internal.chunk_consistency   | N/A (no chunk) | OK (chunk_size=8) | OK (chunk from openpi) | OK (T from policy) |
| internal.attention_drift     | BUG #19   | N/A       | N/A       | N/A            |

Shared infra:
- calibration.py        — BUG #15, #16, #17
- imitation_accuracy    — BUG #22 (dim truncation), gripper-Δ=0 needs probe
- itest_runner.py       — BUG #20 (paraphrase), #21 (substitution_state)
- dataset adapters      — BUG #1 (gr00t silent task), #2 (silent primary alias), #3 (dtype heuristic)
- model adapters        — BUG #27 (gr00t lang setdefault), #28 (gr00t temporal), #30 (gr00t state-dim heuristic), #34-35 (openvla string anchors), #36 (pi0 GCS hardcoded)

**Net: 5 of 5 frame-level diagnostics have at least one identified bug.
Two trajectory-level diagnostics (chunk_consistency, attention_drift) are
mostly clean except for the attention_drift fallback (only matters for
OpenVLA since the others lack attention).**

The dominant theme is **silent substitution / defaulting / averaging**
masking the model's true behaviour. Every fix follows the same principle:
either compute the value honestly, or refuse to emit a verdict.

---

## Part 2 — Findings, by severity

### TIER 1 — Lies to the user (must fix before users see numbers)

#### BUG #1 — Silent instruction fallback in GR00T loader
File: `emboviz/datasets/lerobot_droid.py:304-305`
```python
if not instruction:
    instruction = "pick up the object"
```
The dataset's recorded instruction is sometimes empty / unrecognised for
NVIDIA's droid_sample episodes. We silently substitute "pick up the
object". Every downstream metric (memorization noun parsing, paraphrase
baseline, model action) then runs on a fabricated task string — looks
fine, but is actually evaluating the wrong instruction.

**Fix:** raise `ValueError` listing the row columns we tried, asking the
caller to map the language column properly. No fabrication.

#### BUG #2 — Silent "promote first camera to primary" in lerobot loader
File: `emboviz/datasets/lerobot.py:190-194`
```python
if "primary" not in images and images:
    first_cam = next(iter(self.image_keys))
    images["primary"] = images.get(first_cam, next(iter(images.values())))
```
If a dataset doesn't have a key explicitly named "primary", we silently
alias the first declared camera as "primary". Diagnostics that operate
on the "primary" camera then run on whichever camera happens to be first
in declaration order — which may not be the semantically-primary view.

**Fix:** require every dataset adapter to declare which camera maps to
"primary" explicitly. Raise if missing.

#### BUG #3 — Silent uint8/float dtype heuristic in image conversion
File: `emboviz/datasets/lerobot.py:241-242`
```python
if a.max() <= 1.5:
    a = a * 255.0
```
We auto-multiply by 255 if the array's max is ≤ 1.5, on the assumption
that low max means float-[0,1] image. A genuinely very-dark uint8 image
(max value < 2) gets multiplied by 255 and either overflows or becomes
all 255. Rare in practice but exactly the kind of edge case that
produces unreproducible "wrong image was tested" reports.

**Fix:** check `a.dtype` explicitly — uint8 stays uint8, float multiplies.

#### BUG #4 — Memorization noun parser silently misses anything outside taxonomy
File: `emboviz/perturb/_target_detection.py:157-171`
```python
def _pick_noun(self, instruction: str) -> Optional[str]:
    from emboviz.taxonomy.object_categories import OBJECT_CATEGORIES
    priority = ["utensil", "food", "toy", "cloth", "tool", "container"]
    ...
```
The noun parser only matches a fixed taxonomy. Instructions like
"pick up **the object**" (GR00T ep0), "move the bottom right tip of
**the duvet**" (GR00T ep2), "open the **lid**", "press the **button**"
all return None → memorization silently becomes `not_applicable` for
those trajectories. Result: no memorization signal at all for GR00T.

**Proper fix (literature-backed):** GroundingDINO supports
*phrase grounding* — pass the entire referring expression
("the bottom right tip of the duvet to the left") to GroundingDINO
directly without noun extraction. Section 3.2 of the GroundingDINO
paper covers this. Removes the noun-extraction step entirely.

#### BUG #5 — SAM-fail fallback to bbox-only mask (silent degradation)
File: `emboviz/perturb/_target_detection.py:146-155`
```python
except (ImportError, OSError, RuntimeError) as e:
    _warnings.warn(...)
    self._sam = None
    self.use_sam = False
```
If SAM fails to load, we silently keep going with bbox-only masking.
A bbox covers the target *plus background*, so the "mask" intervention
is much weaker than intended (and noisier). The warning fires once on
load; the user reading 8 frames of memorization scores has no record
that masks were coarse.

**Fix:** propagate the SAM-disabled state into every `TargetDetection`
metadata so the diagnostic raw output shows it. Optionally hard-fail
when calibration depends on the assumed mask precision.

#### BUG #6 — Mask fill = full-image channel mean (may be invisible to model)
Files: `emboviz/diagnostics/memorization.py:130`,
`emboviz/diagnostics/sensitivity_map.py:71`
```python
chan_mean = arr.reshape(-1, 3).mean(axis=0)
masked_arr[mask] = chan_mean
```
The mask fill is the per-image channel mean. If the target object is
itself near-mean colored (a beige mug on a beige table), the fill is
visually identical to the original target. The "intervention" produces
no perceptual change, the model rightly emits the same action, and we
flag CRITICAL memorization. This is precisely the failure mode we
suspect on π0 LIBERO.

**Fix:** report `mask_contrast = ‖fill_color − mean(target_pixels)‖`
in the raw output. Refuse to emit a CRITICAL verdict when contrast is
below a sensible perceptual threshold. Optionally use a high-contrast
fill (color complementary to target mean) rather than image mean.

#### BUG #7 — modality_dropout state substitution may be near-baseline
File: `emboviz/diagnostics/modality_dropout.py:147-160` (+ runner #21)
The diagnostic substitutes `state` with a `substitution_state` from the
caller. The runner currently passes `last_frame.state`. For
mostly-stationary trajectories (GR00T's droid_sample, where the robot
barely moves), `last_frame.state ≈ current_frame.state` → intervention
magnitude is essentially zero → response magnitude is also near-zero →
diagnostic flags "state ignored" when in fact we never tested it.

**Proper fix (literature: Štrumbelj & Kononenko 2014, SHAP-style):**
sample substitution from the *empirical marginal distribution* of state
across a broader pool (other trajectories from the same dataset). And:
**always report `intervention_magnitude` alongside `response_magnitude`;
refuse to emit "ignored" when intervention is below a sensible threshold,
surface as "intervention too weak to test" instead.**

#### BUG #8 — modality_dropout action_history fallback to zeros
File: `emboviz/diagnostics/modality_dropout.py:182-183`
```python
else:
    sub_actions = np.zeros_like(hist.actions)
```
When `substitution_action_history` is None, falls back to all zeros.
Zero action history is the model's neutral / no-prior-motion prior —
likely the same point the model uses as its initialization. Δaction
will be near-zero, false "ignored" verdict.

**Fix:** same as #7 — require caller to provide a marginal-sampled
substitution. No zero fallback.

#### BUG #9 — modality_dropout gripper substitution = range midpoint
File: `emboviz/diagnostics/modality_dropout.py:166-169`
```python
mid = (range[0] + range[1]) / 2 if profile.gripper else 0.5
new_gripper = replace(gripper, value=float(mid))
```
Gripper substitution uses the range midpoint (e.g., 0.5 on [0,1]).
For models that interpret gripper as "currently open vs closed", midpoint
may be exactly the model's "no information" point → near-baseline output
→ false "ignored" verdict.

**Fix:** marginal-sample from observed gripper values across the dataset.

#### BUG #10 — modality_dropout instruction = single space
File: `emboviz/diagnostics/modality_dropout.py:195-198`
```python
ar = averaged_predict(model, scene.with_instruction(" "), n_samples)
```
A single space may be tokenizer-stripped (some preprocessors strip
whitespace before tokenizing → empty effective instruction → some models
raise, others fall through to a default behaviour). The "intervention"
is implementation-dependent rather than a controlled drop.

**Fix:** use a deliberately uninformative but well-formed instruction
("do nothing." or a sampled instruction from a different task entirely).

#### BUG #11 — sensitivity_map.direction is INVERTED
File: `emboviz/diagnostics/sensitivity_map.py:151`
```python
direction="higher_is_worse",   # diffuse sensitivity = worse
```
The comment is right about "diffuse = worse" — but the `scalar` is
*concentration* (HIGHER means MORE focused = BETTER). So with
`higher_is_worse`, the framework's `worst_frame_idx` selects the
MOST-focused frame as worst. This is exactly backwards.

**Fix:** flip to `direction="lower_is_worse"`. Verify no consumer of
the direction has compensating logic.

#### BUG #12 — sensitivity_map severity thresholds collide with grid noise
File: `emboviz/diagnostics/sensitivity_map.py:117-137`
For a 4×4 grid (16 cells), the top-4-cells-out-of-16 captures 25% of
total sensitivity by uniform expectation. The MODERATE threshold
(`scalar > 0.25`) is *at* the noise floor — pure noise grids will read
borderline-MODERATE.

**Fix:** subtract calibration noise floor from each cell's Δaction
before computing concentration. Or use a relative threshold:
"concentration > uniform_expectation + N·σ".

#### BUG #13 — sensitivity_map "consumed" threshold not calibration-aware
File: `emboviz/diagnostics/sensitivity_map.py:91`
```python
if total < 1e-9:
    per_camera_consumed[cam] = False
```
The "consumed" / "ignored" decision uses `total < 1e-9` — an absolute
threshold that ignores the model's noise floor. For a noisy model with
genuinely-small actions, this threshold is always satisfied (total far
exceeds 1e-9) so the camera always reads as "consumed" — even when the
per-cell Δ is pure decoding noise.

**Fix:** `total < noise_floor × n_cells × safety_factor` from calibration.

#### BUG #19 — attention_drift silently fills (0.5, 0.5) when attention is zero
File: `emboviz/diagnostics/attention_drift.py:94-96`
```python
if total <= 0:
    centroids.append((0.5, 0.5))
    continue
```
If a frame's attention sums to ≤ 0 (numerical edge case, or genuinely
zero attention from a misconfigured layer), we silently insert the
image-center centroid for that frame. The drift measurement gets
contaminated by that fake centroid being far from neighbouring real
ones.

**Fix:** raise — zero attention is a model/extraction bug, not a metric
to absorb.

#### BUG #20 — paraphrase loop uses raw `model.predict` instead of averaged
File: `scripts/itest_runner.py:209,212`
```python
baseline_action = model.predict(trajectory.frames[0]).action
for variant in pp.variants(trajectory.frames[0]):
    pred = model.predict(variant.scene).action
```
The paraphrase delta is `‖model.predict(variant) − model.predict(baseline)‖`
where both calls are single-sample. For stochastic models (π0
flow-matching), every "delta" includes single-sample decoding noise.
A model that's actually invariant to paraphrase will look like it
isn't, because the noise dominates.

**Fix:** use `averaged_predict(model, scene, calibration.n_samples)`
for both baseline and each variant.

#### BUG #21 — runner picks substitution_state = last_frame.state
File: `scripts/itest_runner.py:177-184`
```python
sub_state = np.asarray(
    last_scene.observations.state.values, ...)
```
See BUG #7 — last-frame state is the wrong choice. Combined with #7,
fixing the diagnostic requires both the diagnostic API change AND a
real marginal-pool sampling implementation in the runner.

#### BUG #22 — expert_delta silently truncates to shorter dim
File: `scripts/itest_runner.py:312-315`
```python
n = min(len(pred), len(expert_arr))
per_dim = (pred[:n] - expert_arr[:n]).tolist()
```
If the model's `pred` has 8 dims and the dataset's `expert` has 7 dims
(or vice versa), we silently truncate to 7 and compare. This hides a
real shape mismatch that would otherwise tell the user "your model and
dataset disagree on action layout."

**Fix:** raise on dim mismatch. The user wants to know if they've paired
a 7-DOF model with an 8-DOF dataset.

---

### TIER 2 — Correctness concerns under specific conditions

#### BUG #14 — single_sample_noise_floor < 1e-9 forces n_samples=1
File: `emboviz/calibration.py:204-208`
For a model with tiny actions (`single_sample_noise ≈ 0`), we set
`n_samples=1`. Sound for deterministic models. Concerning only if a
model has both tiny noise AND tiny signal — then we never average,
and Δaction interventions look like noise. Probably benign.

#### BUG #15 — `normalize()` silently returns 0 for zero-typical
File: `emboviz/calibration.py:88-90`
```python
if self.typical_action_magnitude < 1e-9:
    return 0.0
```
For a near-zero-action model, all normalized scores collapse to 0.0,
producing universal PASS verdicts without warning the user that
normalization is degenerate.

**Fix:** raise (or return NaN with a `score_undefined: true` flag in
the diagnostic raw output).

#### BUG #16 — averaged_predict silently truncates chunks to shortest
File: `emboviz/calibration.py:131-139`
```python
min_t = min(c.shape[0] for c in chunks)
mean_chunk = np.mean(np.stack([c[:min_t] for c in chunks], axis=0), ...)
```
For stochastic models, if chunks have different lengths (which they
shouldn't, but the truncate fallback exists "defensively"), we silently
discard data. If shapes do differ, that's a model bug — should raise.

**Fix:** assert shape consistency, raise on mismatch.

#### BUG #17 — mag_probe vs typical_action_magnitude use different sampling
File: `emboviz/calibration.py:197-200` vs `:212-215`
- `mag_probe` (used in n_samples math) = mean magnitude of 3
  *single-sample* predictions.
- `typical_action_magnitude` (used as denominator in normalize()) =
  median magnitude of *averaged* predictions across all frames.

By Jensen, `‖mean(a)‖ ≤ mean(‖a‖)`, so for stochastic models the
averaged magnitude is BIASED LOW relative to single-sample magnitude.
n_samples is computed against the larger reference, then scores are
normalized against the smaller — so reported normalized scores are
biased slightly HIGH for stochastic models.

**Fix:** use averaged-prediction magnitude in both places. Be consistent.

#### BUG #27 — GR00T language config silently sets a "task" key
File: `emboviz/models/gr00t.py:368-369`
```python
language[lk] = [[scene.instruction]]
language.setdefault("task", [[scene.instruction]])
```
Adds `"task"` key whether or not the embodiment expects it. If a future
embodiment expects a different key, this silently passes wrong shape.

**Fix:** only populate the modality's declared keys.

#### BUG #28 — GR00T silently replicates current frame as temporal context
File: `emboviz/models/gr00t.py:262-267`
```python
def _to_horizon(arr_3d: np.ndarray) -> np.ndarray:
    stacked = np.stack([arr_3d] * video_horizon, axis=0)
    return stacked[np.newaxis, ...]
```
If an embodiment declares `delta_indices=[-1, 0]` (current + previous
frame), we silently replicate the current frame in both slots — giving
the model a "no motion" signal. Models that condition on motion get a
misleading observation.

**Fix:** require the runner to pass a Trajectory (not a Scene) when the
embodiment has temporal horizon > 1, then index back into history.

#### BUG #30 — GR00T state-dim heuristic when introspection fails
File: `emboviz/models/gr00t.py:240-245`
```python
if "9d" in k: return 9
if "gripper" in k: return 1
return 7
```
Falls back to name-based heuristic (with a warning) if normalization
introspection fails. For an unknown future state key, returns 7 — which
will produce a downstream shape error somewhere, but only after a
confusing warning.

**Fix:** raise with the embodiment + key + available introspection
attempts. Don't guess dims.

---

### TIER 3 — Robustness / cosmetic

#### BUG #18 — chunk_consistency thresholds are model-agnostic constants
File: `emboviz/diagnostics/chunk_consistency.py:60-61`
```python
noise_floor: float = 0.10
grounded_threshold: float = 0.50
```
With calibration, scores are normalized so these mean "10% of typical
action" and "50% of typical action." Reasonable but arbitrary.

#### BUG #23 — formatted print rounds to 3 decimals
File: `scripts/itest_runner.py:329`
Cosmetic. Doesn't affect correctness.

#### BUG #24 — convention-mismatch heuristic uses 3× median
File: `scripts/itest_runner.py:336-339`
Heuristic threshold; informational only.

#### BUG #26 — pi0 LIBERO uses HWC but `_to_chw_uint8` exists for droid/aloha
File: `emboviz/models/pi0.py:50-55` (helper exists), `:138-155` (LIBERO ignores it)
Easy to confuse future contributors. Add a comment that LIBERO is
deliberately HWC because openpi's libero policy expects HWC.

#### BUG #29 — GR00T action_dim = 0 until first predict
File: `emboviz/models/gr00t.py:121,148`
Calibration calls predict, so action_dim is set by the time downstream
code reads it. But pre-prediction queries see 0. Minor.

#### BUG #31 — OFT inline GenerateConfig hardcodes task_suite_name
File: `emboviz/models/openvla_oft.py:117`
```python
task_suite_name: str = "libero_spatial"
```
If OFT helpers use this for task-specific normalization, the value is
wrong for non-spatial checkpoints. Verify against OFT source.

#### BUG #32 — OpenVLA action_scale degrades to None silently
File: `emboviz/models/openvla.py:95-102`
Documented warning, but downstream `compare_actions` silently falls back
to plain L2. Some metrics may use raw L2 thinking they got scale-aware.

#### BUG #33 — OpenVLA dtype = bfloat16 hardcoded
File: `emboviz/models/openvla.py:70`
Hardware without bf16 fails at load. Surface as constructor arg.

#### BUG #34 — OpenVLA `_extract_instruction_from_ids` regex anchors
File: `emboviz/models/openvla.py:526-535`
If the prompt template ever changes, we silently return raw decoded
text. Add an explicit assertion.

#### BUG #35 — OpenVLA `_resolve_query_position` returns last on unmatched word
File: `emboviz/models/openvla.py:521-524`
If the word doesn't tokenize into any matched position, we silently fall
back to the last position. Word-anchored diagnostics get a misleading
"the model attended to the word" when actually we couldn't find the word.

#### BUG #36 — pi0 GCS_PREFIX hardcoded
File: `emboviz/models/pi0.py:47`
If openpi moves checkpoints to a new bucket, breaks with a confusing
404 from `download.maybe_download`. Make it overridable + document.

---

## Part 3 — Fix order (recommended)

This isn't a piecemeal list — it's a sequence that gets us to "every
metric is honest" with the fewest interdependencies broken.

### Pass 1 — Honesty refactor (no new science, just stop lying)
1. **#1** GR00T loader: raise on missing instruction (5 min)
2. **#2** lerobot loader: raise on missing "primary" (5 min)
3. **#3** lerobot loader: dtype-explicit image conversion (5 min)
4. **#11** sensitivity_map: flip direction string to "lower_is_worse" (1 min)
5. **#15** calibration.normalize: raise on degenerate typical_action (2 min)
6. **#16** averaged_predict: assert chunk shape consistency (2 min)
7. **#19** attention_drift: raise on zero attention (2 min)
8. **#22** runner: raise on dim mismatch between pred and expert (2 min)
9. **#27** gr00t: don't setdefault "task" (2 min)
10. **#34, #35** openvla: assertions instead of silent fallthrough (5 min)

Total: ~30 min. Mechanical edits. After this pass nothing lies — many
diagnostics will now refuse to emit a verdict where they previously
gave a wrong one.

### Pass 2 — Calibration consistency (small math correction)
11. **#17** calibration: use averaged magnitude in n_samples computation
    too. (15 min)
12. **#13** sensitivity_map "consumed" threshold from calibration noise
    floor instead of `1e-9`. (15 min)

### Pass 3 — Memorization repair (the literature-backed work)
13. **#4** Replace `_pick_noun` taxonomy with GroundingDINO
    phrase-grounding on the full referring expression. Reference:
    Liu et al. 2024 "Grounding DINO" §3.2. (2-3 hr including testing
    on all 4 model trajectories)
14. **#5, #6** Add `mask_contrast` reporting + refuse CRITICAL when
    mask contrast below perceptual threshold. (1 hr)
15. **#20** Paraphrase: use `averaged_predict`. (10 min)

### Pass 4 — Modality dropout repair (the principled work)
16. **#7, #8, #9, #10, #21** Replace single-substitution with
    marginal-distribution sampling from a `state_pool` /
    `action_history_pool` / `gripper_pool` / `instruction_pool`
    constructed from broader dataset frames. Report
    `intervention_magnitude` alongside `response_magnitude`. Refuse
    "ignored" verdict when intervention is below threshold. Reference:
    Štrumbelj & Kononenko 2014, SHAP. (4-6 hr)

### Pass 5 — Long-tail hardening
17. **#12** sensitivity_map calibration-aware threshold (1 hr)
18. **#28, #30** GR00T temporal context, state-dim heuristic (1-2 hr)
19. **#31, #32, #33, #36** Model-adapter robustness tweaks (1 hr)

After all 5 passes the framework is bulletproof — every metric either
gives a real answer or refuses to give one.

---

## Part 4 — What we did NOT find

The following surfaces were read and have no bugs:
- `core/types.py` (Scene, Trajectory, AttentionMaps) — clean strict
  contracts, raises on missing keys, no silent fallbacks
- `models/protocol.py` (VLAModel ABC, RequiredInputs validation) —
  clean, raises on missing modalities
- `diagnostics/trajectory.py` (TrajectoryDiagnostic wrapper, bootstrap
  CI) — clean
- `diagnostics/base.py` — minimal, clean
- `exporters/correlation.py` (find_failure_moments) — clean
- `models/openvla_oft.py` predict() observation building — strict, no
  silent pad / truncate
- `models/pi0.py` _libero_observation_builder HWC vs CHW — correct after
  the prior fix, with comment explaining

The protocol layer (Scene, Observations, RequiredInputs, with_image,
with_images, resolve_cameras, validate) is solid — that's the load-bearing
contract that lets us identify these bugs at all, because the boundaries
*do* validate strictly.

---

## Part 5 — Empirical probes needed (Phase 2, requires VM)

These can't be proven from code alone:

A. **π0 LIBERO memorization probe** — save mask PNG + run blank-cameras
   + wrong-scene interventions. Proves/disproves whether the CRITICAL
   verdict is real OR mask is too weak (BUG #6).
B. **GR00T gripper Δ=0 probe** — print 5 frames of (model gripper,
   expert gripper) values. Three possibilities to discriminate:
   model is perfect, model emits identical-to-expert constant, or
   index mismatch.
C. **State substitution magnitude probe** — for each model on its
   trajectory, print `‖sub_state − cur_state‖ / typical_state_diff`.
   Confirms BUG #7 across all 4 models (we suspect it but haven't
   measured the actual ratios).
D. **Sensitivity-map noise floor probe** — run sensitivity_map on a
   blank image and measure the per-cell Δ distribution. Confirms BUG
   #12 (whether 25% concentration is the noise expectation).
E. **Memorization mask-contrast probe** — for each frame, compute
   `‖fill_color − mean(target_pixels)‖` and report. Confirms BUG #6.
