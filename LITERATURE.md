# Emboviz — Methodology and References

This document records the published methodology behind each Emboviz
diagnostic: the algorithm implemented, the rationale for the design
choices, the approaches excluded, and the supporting citations.

It covers the five diagnostics and the shared calibration procedure,
followed by per-model methodology notes and a master citation list.

---

## 1. Vision memorization — does the policy USE its visual input?

### Question being answered

If the manipulated target object is masked in the input image and the
policy's predicted action is unchanged, the policy is replaying a
memorized motor pattern conditioned on non-visual signals (proprio,
instruction, action history) rather than reading the scene. This
matters because such policies pass training-distribution success
metrics by surface-mimicking demonstrations, then fail catastrophically
on any visual variation.

### Method

```
Inputs:
    policy π (potentially stochastic — sample K times per call)
    instruction I (natural-language referring expression)
    observation o = {image_per_camera, state, gripper, action_history}
    detection confidence threshold τ_det     (default 0.30)
    intervention magnitude threshold τ_in    (normalized pixel-L2,
                                              default 0.05)
    response magnitude threshold τ_out       (normalized action,
                                              default 0.05 of typical)

1.  phrase ← MLLM_parse(I) -- extract the manipulated-object phrase
    if MLLM_parse cannot find a manipulated object → return INCONCLUSIVE
    (no fabricated noun; guessing from a fixed taxonomy is not permitted)

2.  per-camera detection (default detector = SAM 3, text → mask):
    for each camera c in observation:
        mask_c, score_c ← SAM3(o[c], phrase)
        if score_c < τ_det: skip camera c (record reason)
    -- the legacy GroundingDINO + SAM combo is the `--detector gd-sam`
       fallback for environments where SAM 3 isn't available.

    if no camera had a confident detection:
        return INCONCLUSIVE ("could not locate target on any view")

3.  fill ensemble — TWO independent perturbations:
        o'_mean[c] = paste(o[c], mask_c, channel_mean(o[c]))
        o'_blur[c] = paste(o[c], mask_c, gaussian_blur(o[c], σ=diag/24))
    -- the on-manifold lama_inpaint fill the literature prescribes as a
       THIRD fill is NOT implemented (see the 2-of-3 caveat below); the
       agreement gate runs over these two non-on-manifold fills only.

4.  for each fill F:
        Δ_in[F] = normalized_pixel_L2(o, o'[F]) over ∪mask_c
                  -- approximates LPIPS; a learned perceptual metric is
                     not used (no LPIPS model loaded)
        A      = mean of K samples π(o)
        A'     = mean of K samples π(o'[F])
        Δ_out[F] = normalized_L2(A − A')      -- using ModelCalibration

5.  intervention validity gate:
        if max(Δ_in[F]) < τ_in:
            return INCONCLUSIVE ("masks too low-contrast to test;
                                  target is itself near-mean colored")

6.  verdict (require agreement across ALL fills to call memorization):
        if max(Δ_out[F]) < τ_out:    return MEMORIZATION_SIGNATURE
        if min(Δ_out[F]) > τ_out:    return VISUALLY_GROUNDED
        else:                        return MIXED

7.  null controls (run on demand, not per-frame):
        mask_random_same_area → expect Δ_out ≈ noise
        mask_entire_image     → expect Δ_out large
        perturb_only_instr    → tests image-language coupling
        mask_with_perturbed_proprio → proprio-confounder check
```

### Rationale

- **Free-form referring expressions** (e.g. "the bottom-right tip of
  the duvet to the left") — GroundingDINO's block-diagonal phrase
  attention processes noun phrases independently, so passing the raw
  sentence and trusting the top box is unreliable [Liu et al. 2024].
  An MLLM phrase-extraction front-end (Qwen2.5-VL or Florence-2's
  REFERRING_EXPRESSION_SEGMENTATION token) is the SOTA pre-step.
  Florence-2 [Xiao et al. 2024] supports this end-to-end.
- **Mask-fill ensemble** — single-fill baselines suffer "baseline
  blindness": if the masked region matches the baseline color the
  model treats the mask as informative noise rather than as absent
  [Sturmfels, Lundberg & Lee 2020]. ROAR and ROAD show that single-fill
  perturbations create OOD inputs that bias verdicts
  [Hooker et al. 2019; Rong et al. 2022]. Inpainting (LaMa, BYOVLA's
  Inpaint-Anything) stays on the data manifold but is *less aggressive*,
  while channel-mean is more aggressive but OOD. Reporting BOTH and
  requiring agreement neutralizes both failure modes.
  *Shipped status (3 fills available):* the default ensemble is
  channel-mean + Gaussian-blur (both OOD-leaning, pure-numpy, no worker).
  The on-manifold `lama_inpaint` fill — the literature-prescribed THIRD,
  less-aggressive fill — is implemented via LaMa [Suvorov et al. 2022]
  and enabled per-run by adding `lama_inpaint` to `analysis.fills` (it
  runs in the isolated `emboviz-lama` ZMQ worker; deterministic and
  feed-forward, so the fill is reproducible and does not hallucinate new
  content into the hole). When enabled, the agreement gate spans the
  on-manifold/OOD axis; when not, the run is honest about it — every
  result's `fill_ensemble.on_manifold_fill_present` flag and `note`
  state which fills the agreement gate actually ran over.
- **Removal-mask dilation** — a pixel-tight detection silhouette excludes
  the object's anti-aliased boundary and any thin contact shadow. LaMa
  reconstructs that 1–2 px rim straight back into the hole (a faint ghost
  of the "removed" object); the OOD fills leave it at its original colour.
  The LaMa object-removal recipe [Suvorov et al. 2022] and the erase
  pipelines built on it (IOPaint / lama-cleaner) dilate the erase mask by
  a small margin before inpainting for exactly this reason. We grow the
  detected mask ONCE per camera — radius `max(2, round(0.03·min(H,W)))`,
  resolution-independent — and feed the SAME dilated mask to every fill
  and to the contrast gate, so the OOD↔on-manifold agreement is still
  measured over one identical region. The applied radius is surfaced in
  each result's `raw_numbers.mask_dilation_px`.
- **K-sample averaging for stochastic policies** — π0 (flow-matching),
  Diffusion Policy, GR00T's flow-matching head all have non-trivial
  sample-to-sample variance. BYOVLA [Hancock et al. 2024] samples K
  action chunks per observation; without this the metric conflates
  decoding stochasticity with intervention response.
- **Intervention magnitude as a gating condition** — the do-operator
  framing of causal mediation [Vig et al. 2020; Mueller et al. 2025]
  requires that an intervention *actually change the mediator*.
  LPIPS over the mask area answers "did the image actually change?"
  An "ignored" verdict from `Δ_out ≈ 0` is only valid when `Δ_in ≥ τ_in`.
- **Detector confidence gate** — Adebayo et al. 2018 ("Sanity Checks
  for Saliency Maps") established that interpretability methods must
  explicitly fail on null inputs. A low-confidence detection IS a null
  input; the verdict is refused rather than fabricated.
- **Proprio-confounder control** — Lin et al. 2025 ("Do You Need
  Proprioceptive States in Visuomotor Policies?", arXiv:2509.18644)
  showed that policies frequently overfit to proprio and "ignore"
  vision in a way that looks like memorization but is actually
  proprio-leakage. To distinguish, repeat the masked-image test with
  proprio perturbed; if Δ_out remains zero only with original proprio,
  the policy's invariance is proprio-driven, not target-blind.

### Excluded approaches

1. **Fixed noun taxonomy** — matching the instruction against a small
   closed set of categories silently skips any out-of-taxonomy task.
   Real robot tasks reference arbitrary objects ("the lid", "the pipe",
   "the recycling bin", "the bottom right tip of the duvet"), which a
   closed lookup table does not cover.
2. **Centered-rectangle fallback** when detection fails (silently
   masks the gripper or empty space).
3. **Black-fill masking** (triggers vision-encoder "lights-off" prior;
   inflates Δ_out spuriously).
4. **Bbox-only masking when SAM is unavailable** — without the
   pixel-accurate mask the intervention is much weaker than intended.
   Surface SAM-unavailable as `INCONCLUSIVE`, not as a quiet
   degradation.
5. **Single-trajectory or single-sample inference** for stochastic
   policies (BYOVLA precedent: K ≥ 8 chunks per evaluation).
6. **Reporting only response magnitude** without intervention
   magnitude — invariably gives false-CRITICAL verdicts when the mask
   happens to be visually invisible.

### Citations

- Liu, Zhou et al. **GroundingDINO**, ECCV 2024 (arXiv:2303.05499)
- Ren et al. **DINO-X**, 2024 (arXiv:2411.14347)
- Xiao et al. **Florence-2**, CVPR 2024
- Ravi et al. **SAM 2**, 2024; Meta **SAM 3 / Segment Anything with
  Concepts**, 2025
- Suvorov et al. **LaMa — Resolution-robust Large Mask Inpainting with
  Fourier Convolutions**, WACV 2022 (arXiv:2109.07161) — the on-manifold
  `lama_inpaint` fill (Apache-2.0)
- Hancock et al. **BYOVLA — Run-time Observation Interventions Make
  VLAs More Visually Robust**, 2024 (arXiv:2410.01971)
- Geng et al. **LIBERO-PRO**, 2025 (arXiv:2510.03827)
- Lin et al. **Do You Need Proprioceptive States in Visuomotor
  Policies?**, 2025 (arXiv:2509.18644)
- Sturmfels, Lundberg & Lee. **Visualizing the Impact of Feature
  Attribution Baselines**, Distill 2020
- Hooker, Erhan, Kindermans & Kim. **ROAR — A Benchmark for
  Interpretability Methods**, NeurIPS 2019 (arXiv:1806.10758)
- Rong et al. **ROAD**, ICML 2022 (arXiv:2202.00449)
- Adebayo et al. **Sanity Checks for Saliency Maps**, NeurIPS 2018
  (arXiv:1810.03292)
- Vig et al. **Causal Mediation Analysis for Interpreting Neural NLP**,
  NeurIPS 2020 (arXiv:2004.12265)
- Mueller et al. **The Quest for the Right Mediator**, Computational
  Linguistics 2025 (arXiv:2408.01416)
- Zhang et al. **Photorealistic Inpainting for Perturbation-based
  Explanations**, NeurIPS 2025 (arXiv:2510.03317)

---

## 2. Input modality dropout — does the policy USE each input modality?

### Question being answered

For each declared input modality M (image-per-camera, state, gripper,
action_history, instruction), replace it with a *neutral* value and
measure how much the predicted action changes. A genuinely-used
modality moves the action by a meaningful amount; an ignored modality
leaves the action unchanged. The model's `required_inputs` declaration
tells us what it CLAIMS to consume — this test verifies whether it
actually does.

### Method

```
Inputs:
    policy π (sample K=10–20 chunks for stochastic policies)
    scene s with modalities {M_1, ..., M_n}
    dataset pool D — a large set of OTHER scenes from the same dataset
        (different episodes; never the current trajectory)
    P = 50–100 substitution samples per modality per query frame
    intervention validity threshold (per modality, see below)

For each modality M_i:
    1. Build replacement pool R_i by drawing P samples of M_i from D.
       The pool must be DRAWN FROM A DIFFERENT EPISODE to avoid
       autocorrelation (Hooker & Mentch 2019).

    2. Filter near-duplicates:
       drop samples r in R_i where d_M(r, s.M_i) < 25th-percentile of
       pairwise dataset distances for M_i.

       d_M is the natural metric for that modality:
          image:   LPIPS or pixel-L2 over the masked region
          state:   geodesic-on-SO(3) for rotation components, L2 for
                   translation / joints
          gripper: |value difference| (or 1 for a flipped bit)
          action_history: per-step L2 averaged over history length
          instruction: 1 − cosine(sentence-embedding) [BGE-large
                       or sentence-transformers]

    3. For each surviving r in R_i:
         s'_r = scene with s.M_i replaced by r (other modalities held)
         a    = averaged_predict(π, s)
         a'_r = averaged_predict(π, s'_r)
         Δ_in_r = d_M(r, s.M_i)
         Δ_out_r = normalized_L2(a − a'_r)        -- via ModelCalibration

    4. Aggregate:
         mean_Δ_in   = mean over r of Δ_in_r
         mean_Δ_out  = mean over r of Δ_out_r
         sensitivity_ratio = mean_Δ_out / mean_Δ_in

    5. Validity gate:
         if mean_Δ_in < τ_in_modality:
             verdict = INTERVENTION_TOO_WEAK
             (record reason: the chosen substitutions are not
              meaningfully different from the original value)
         else:
             apply verdict thresholds below.

    6. Verdict:
         if mean_Δ_out < noise_floor:          IGNORED
         elif mean_Δ_out < grounded_threshold: PARTIAL
         else:                                  USED
```

### Per-modality replacement protocol

| Modality | Pool source | Sample primitive |
|---|---|---|
| Image-per-camera | Random in-distribution scenes from a different episode | Drop in the full image (NOT black, NOT mean fill of one camera) |
| Proprioceptive state | State vectors from random different-episode frames | Direct substitute (no zero, no midpoint) |
| Gripper (scalar) | Empirical gripper values from dataset | Sample one |
| Gripper (binary) | {0, 1} | Flip the bit |
| Action history | Action histories from different-episode frames | Direct substitute (no zero) |
| Instruction | Instructions from DIFFERENT TASKS in the same dataset | Sample one |

### Rationale

- **Marginal-distribution sampling**, not single-value substitution,
  is the unique attribution that satisfies Shapley axioms (local
  accuracy, missingness, consistency) [Lundberg & Lee 2017;
  Štrumbelj & Kononenko 2014]. With one substitution per modality you
  get a sample of size 1 — the verdict is dominated by the choice of
  substitution rather than the model's behaviour.
- **Marginal vs conditional** — Janzing, Minorics & Blöbaum 2020
  argue marginal sampling is the correct interventional notion (the
  do-operator). Conditional sampling (e.g., a state vector
  conditioned on the current image) would leak the conditioning
  information through the substitution, contaminating the verdict.
- **Different-episode pool**, NOT same-episode last-frame. For
  time-series with strong autocorrelation, last-frame state ≈
  current-frame state — intervention magnitude is essentially zero
  and any "ignored" verdict is uninterpretable (Hooker & Mentch 2019
  on "please stop permuting features": correlated features in a
  single trajectory invalidate permutation-based importance).
- **Zero-substitution is pathological for structured representations**
  — 6D rotations [Zhou et al. 2019] are continuous SO(3)
  parameterizations where zeros violate orthonormality and break
  downstream SVD; quaternions zero-fill to non-unit non-rotations;
  Euler angles zero to the identity rotation which IS a valid pose
  the model has seen and is NOT semantically "absent". Each of these
  silently corrupts the intervention.
- **Midpoint substitution is pathological for binary / saturated
  inputs** — for a gripper in [0, 1], substituting 0.5 is exactly
  "halfway between open and closed", which is precisely the model's
  most-ambiguous state and tends to produce minimum action response
  for reasons unrelated to whether the gripper signal is used.
- **Empty-string instruction is pathological** for instruction-tuned
  language backbones — they have a strong "refuse / do nothing"
  prior on empty input that the policy may inherit. Single-space is
  tokenizer-dependent. "Do nothing" is itself a valid task. The
  honest substitution is a real instruction from a different task in
  the dataset, drawing from the dataset's empirical instruction
  distribution.
- **Pool size 50–100** per query frame is the practical compromise
  derived from RISE [Petsiuk et al. 2018], which used 8000 binary
  masks for image attribution and showed variance scales as O(1/√N).
  Below ~20 samples the verdict is unstable; above ~200 is overkill
  for online use.
- **Intervention magnitude reporting and abstention** — causal
  abstraction work [Geiger et al. 2023] makes the principle explicit:
  an intervention's response is only interpretable when the
  intervention itself is large enough to matter relative to the
  modality's natural scale. Both `mean_Δ_in` and `mean_Δ_out` are
  reported, along with their ratio, and `IGNORED` is withheld when
  `mean_Δ_in` is below the 25th percentile of dataset pairwise
  distances.

### Excluded approaches

1. **Zeros for structured state** (rotations, quaternions, normalized
   embeddings) — crashes structured decoders or silently passes
   identity-rotations the model has seen.
2. **Mean / midpoint substitution** — coincides with the model's
   null prior.
3. **Empty string for instruction** — triggers the LLM's refusal
   prior; not a controlled intervention.
4. **Last-frame substitution for state in autocorrelated
   trajectories** — intervention magnitude ≈ 0.
5. **Single substitution sample** (no marginal averaging).
6. **Reporting `Δ_out` without `Δ_in`** — produces false-IGNORED
   verdicts when the intervention is too weak to register.
7. **Permuting correlated features without acknowledging
   extrapolation** (Hooker & Mentch).
8. **Reporting "do nothing" as a neutral instruction.**

### Citations

- Štrumbelj & Kononenko 2014. *Explaining prediction models and
  individual predictions with feature contributions*. KIS 41:647–665.
- Lundberg & Lee 2017. **SHAP — A Unified Approach to Interpreting
  Model Predictions**, NeurIPS (arXiv:1705.07874).
- Janzing, Minorics & Blöbaum 2020. **Feature relevance quantification
  in explainable AI: A causal problem**, AISTATS (arXiv:1910.13413).
- Hooker, Erhan, Kindermans & Kim 2019. **ROAR**, NeurIPS
  (arXiv:1806.10758).
- Hooker & Mentch 2019/2021. **Please Stop Permuting Features /
  Unrestricted Permutation Forces Extrapolation**
  (arXiv:1905.03151).
- Sundararajan, Taly & Yan 2017. **Integrated Gradients**, ICML
  (arXiv:1703.01365).
- Petsiuk, Das & Saenko 2018. **RISE — Randomized Input Sampling for
  Explanation**, BMVC (arXiv:1806.07421).
- Zhou et al. 2019. **On the Continuity of Rotation Representations
  in Neural Networks**, CVPR (arXiv:1812.07035).
- Lin et al. 2025. **Do You Need Proprioceptive States?**
  (arXiv:2509.18644).
- "When Vision Overrides Language: Counterfactual Failures in VLAs"
  (arXiv:2602.17659).
- CAST. **Counterfactual Labels Improve Instruction Following in VLAs**
  (arXiv:2508.13446).
- Geiger et al. 2023. **Causal Abstraction: A Theoretical Foundation
  for Mechanistic Interpretability** (arXiv:2301.04709).
- AR-VLA. **True Autoregressive Action Expert for VLA Models**
  (arXiv:2603.10126) — stochastic history masking precedent.
- Foerster et al. 2018. **COMA — Counterfactual Multi-Agent Policy
  Gradients**, AAAI.

---

## 3. Vision scene sensitivity — WHERE in the image does the policy look?

### Question being answered

Per-pixel saliency: which regions of each camera causally drive the
policy's action? Distinguishes "model focuses on the target" from
"model relies on background / distractor" from "model ignores camera X
entirely."

### Method

```
Inputs:
    policy π (sample K for stochastic)
    scene s with cameras C
    occluder size σ_occ = min(H, W) / 4   -- Zeiler-Fergus convention
    stride σ_str = σ_occ / 3              -- ~3x overlap

For each camera c in C:
    1. Compute baseline action a_base = averaged_predict(π, s)
    2. Slide occluder across image:
         for y in 0, σ_str, 2σ_str, ..., H - σ_occ:
           for x in 0, σ_str, 2σ_str, ...:
             o' = paste(image_c, [y, x, y+σ_occ, x+σ_occ], channel_mean)
             s' = s with image_c replaced by o'
             a' = averaged_predict(π, s')
             Δaction[y, x] = normalized_L2(a_base - a') (via calibration)

    3. Aggregate overlapping patches to per-pixel attribution map
       (mean over occluders covering each pixel).

    4. Per-cell signal above noise:
         signal = max(map - calibration.noise_floor, 0)

    5. Camera-consumed gate (calibration-aware):
         max_cell_signal = max(signal)
         cell_signal_threshold = 0.05 * typical_action_magnitude
         consumed iff max_cell_signal >= cell_signal_threshold
                      AND total signal >= noise_floor
         (cameras that never clear this are reported as
          ignored_cameras, NOT folded into the headline as a
          misleading zero)

    6. Top-K concentration scalar (per consumed camera):
         concentration = sum(top grid_side cells of signal) / sum(signal)
         scalar = mean(concentration) over consumed cameras

    7. Verdict (FIXED concentration thresholds):
         scalar > 0.5   → PASS     (focused on a few regions)
         scalar > 0.25  → INFO     (visible focus, long tail)
         else           → MODERATE (diffuse — background-statistics risk)
         no consumed camera → UNKNOWN (response below sampling noise)
```

**Design target — NOT yet implemented.** The richer
sparsity-and-calibration methodology below is the intended design and
is NOT in the shipped code today:

```
- Sparsity scalar (Hoyer) instead of top-K share:
      hoyer = (sqrt(N) - L1(signal)/L2(signal)) / (sqrt(N) - 1)
      Hoyer = 1 → spike (single-pixel focus); 0 → uniform (diffuse).
- z-score-calibrated threshold against a null grid:
      compute Hoyer on null grids = shuffled cells of the same map;
      get μ_0, σ_0 across many shuffles; z = (hoyer - μ_0)/σ_0;
      PASS if z > 3, INFO if 1 < z <= 3, MODERATE if z <= 1.
```

### Rationale

- **Sliding occluder with overlap** [Zeiler & Fergus 2014] —
  disjoint-grid is a degenerate special case (occluder = cell,
  stride = cell). Overlap is what gives the smoothing benefit that
  Captum's `Occlusion` and xaitk-saliency both inherit; without it
  the map is blocky and aliased.
- **Calibration noise subtraction PER CELL** — before computing
  concentration, subtract the model's measured noise floor. A noise-
  only cell otherwise contributes spurious "signal" to the
  concentration calculation, biasing the result toward "diffuse"
  even when the real model is focused.
- **Hoyer sparsity** (not top-K share) — Hurley & Rickard 2009
  showed Hoyer is one of only two common sparsity measures (with
  Gini) that satisfy the six axioms a sparsity metric should satisfy
  (Robin Hood, Scaling, Rising Tide, Cloning, Bill Gates,
  Babies). Top-K share has an arbitrary K and a noise-expectation
  baseline of K/N — for a 4×4 grid (N=16) and K=4 that is 0.25, so a
  pure-noise grid reads as borderline-MODERATE under a fixed 0.25
  threshold.
- **z-score calibrated threshold against null grid** — Adebayo et
  al. 2018 ("Sanity Checks for Saliency Maps") established that
  saliency methods must explicitly fail on null inputs (randomized
  weights, shuffled labels) to be trustworthy. This is incorporated:
  per `(model, grid_size, image distribution)` the null Hoyer
  distribution is computed once via cell-shuffling on a calibration
  set, and every observed score is z-scored against it.
- **Camera-consumed test in action-std units** — an absolute
  threshold such as `total_sensitivity < 1e-9` is never reached for
  real noise floors (1e-4 to 1e-2), so every camera reads as consumed
  even on a pure-noise grid. The test instead requires the maximum
  signal-above-noise per cell to exceed 5% of typical action magnitude
  (the same meaningful-intervention criterion used elsewhere).
- **For stochastic VLAs** (π0, GR00T, Diffusion Policy) the right
  target signal is the KL between predicted action distributions at
  fixed noise level, not the L2 between mean actions. It is
  approximated via K-sample averaging through `averaged_predict`
  (consistent with every other diagnostic).

### Excluded approaches

1. **Disjoint grid with no occluder overlap** (aliased / blocky
   maps).
2. **Absolute thresholds (`< 1e-9`) for "ignored"** — should be in
   action-std units from calibration.
3. **Inverted severity direction with concentration as the scalar** —
   treating higher concentration as worse ranks the most-focused frame
   as the worst.
4. **Fixed thresholds (0.5, 0.25) that collide with grid noise
   expectation** — for 4×4, top-4 / 16 = 0.25 is uniform noise, so a
   MODERATE threshold at 0.25 sits at the noise floor.
   *Current-implementation note:* the shipped diagnostic uses
   top-K concentration with FIXED thresholds (0.5 / 0.25) — the same
   class of fixed threshold described in this list. It is made
   safe in practice by the calibration-aware consumed-gate
   (`cell_signal_threshold` + noise-floor subtraction in step 5),
   which removes pure-noise cameras BEFORE the concentration is
   scored, so a noise-only grid reads UNKNOWN rather than
   borderline-MODERATE. The Hoyer + z-score-vs-null calibration
   (per the "Design target — not yet implemented" note above) is the
   intended replacement for the fixed concentration thresholds.

### Citations

- Zeiler & Fergus 2014. **Visualizing and Understanding Convolutional
  Networks**, ECCV (arXiv:1311.2901).
- Petsiuk, Das & Saenko 2018. **RISE**, BMVC (arXiv:1806.07421).
- Hancock et al. 2024. **BYOVLA** (arXiv:2410.01971).
- Sundararajan, Taly & Yan 2017. **Integrated Gradients**, ICML
  (arXiv:1703.01365).
- Smilkov et al. 2017. **SmoothGrad** (arXiv:1706.03825).
- Selvaraju et al. 2017. **Grad-CAM** (arXiv:1610.02391).
- Adebayo et al. 2018. **Sanity Checks for Saliency Maps**, NeurIPS
  (arXiv:1810.03292).
- Hurley & Rickard 2009. **Comparing Measures of Sparsity**, IEEE TIT.
- Cooper & Doshi 2022. **Metrics for saliency map evaluation**
  (arXiv:2201.13291).
- de Haan, Jayaraman & Levine 2019. **Causal Confusion in Imitation
  Learning**, NeurIPS (arXiv:1905.11979).
- "Shortcut Learning in Generalist Robot Policies", CoRL 2025
  (arXiv:2508.06426).
- "Policy Contrastive Decoding for Robotic Foundation Models"
  (arXiv:2505.13255).

---

## 4. Internal attention drift — is the model's visual focus anchored?

### Question being answered

For VLAs that expose attention (OpenVLA-7B, OpenVLA-OFT, π0 with the
PyTorch backend, and GR00T-N1.x), measure across a trajectory: does the
model's visual attention stay anchored on a coherent region, or does it
drift frame-to-frame? Drift correlates with brittle policies that grasp
adjacent to the target.

**GR00T caveat — the attention signal differs by architecture.**
OpenVLA / OFT / π0 are *single-stack*: the action is produced through the VLM's
own attention, so the last-token→image map localizes the manipulated object.
GR00T-N1.x is *dual-system* — a frozen Qwen/Eagle VLM feeds a SEPARATE
diffusion-transformer (DiT) action head (GR00T-N1, arXiv:2503.14734, which
conditions the DiT on intermediate-layer VLM embeddings via cross-attention).
Emboviz extracts GR00T's map from that **DiT action→image cross-attention** (the only
action-grounded signal; the VLM's frozen self-attention is attention-sink
dominated). That DiT signal is the **motor pathway** and is spatially
**dispersed** — it is NOT a reliable object localizer, and this is a documented
VLA property rather than an emboviz defect: ReconVLA (arXiv:2508.10333) and the
VLA survey (arXiv:2507.10672) report VLA visual attention is generally
dispersed/sink-prone, and the GR00T-N1.5 mechanistic study (arXiv:2603.19233)
shows the expert/DiT pathway encodes the *motor program* while goal/object info
lives in VLM *features* (SAEs/linear probes), not attention weights. Emboviz
therefore extracts the seeded DiT cross-attention (meaned over denoise steps,
image-cross blocks, and heads) and labels it as the dispersed motor pathway
rather than as an object localizer. See the README "GR00T attention" note and
`Gr00tAdapter.extract_attention`.

### Method

The shipped diagnostic (`emboviz/diagnostics/attention_drift.py`)
computes ONLY the per-frame attention centroid and the frame-to-frame
centroid displacement in pixels, against fixed thresholds:

```
Inputs:
    model with extract_attention()
    trajectory T with N frames
    camera (default "primary") — used only for pixel-space conversion
    drift_warn_px      = 30.0   (fixed threshold)
    drift_critical_px  = 70.0   (fixed threshold)

1. Per-frame attention extraction + cleaning (delegated to the
   adapter's extract_attention, which applies layer-adaptive
   interior-concentration selection within the model's mid-stack
   fractional band — OpenVLA 0.25–0.75; π0 0.25–0.85 — meaned over
   heads; sink handling is model-dependent, OpenVLA applies none):
   for each frame f:
       attn_v[f] = cleaned per-camera image heatmap (normalized)
       if attn_v[f].sum() <= 0: RAISE  (zero-sum is a real adapter
            bug — fabricating a (0.5, 0.5) centroid is refused)

2. Per-frame centroid (center of mass):
   centroid[f] = E_{(y,x) ~ attn_v[f]}[(y, x)]   in [0,1], then
                 scaled to the configured camera's pixel size

3. Frame-to-frame displacement (PIXELS), only between frames that are
   ADJACENT in the trajectory (no spanning of skipped frames):
   displacement_px[f] = || centroid[f] - centroid[f-1] ||_2

4. Verdict (fixed pixel thresholds):
   mean_drift_px = mean(displacement_px); max_drift_px = max(...)
   < drift_warn_px      → PASS    (anchored)
   < drift_critical_px  → MODERATE (some drift)
   ≥ drift_critical_px  → CRITICAL (unanchored)
```

**Design target — NOT yet implemented.** The richer methodology below
is the intended design; none of it is in the shipped code today. It is
recorded here so the gap is explicit and the "why" citations stay
attached to it:

```
- pointing_accuracy[f] = sum(attn_v[f] inside target_bbox[f])
      (requires per-frame target bbox from the detector — not wired)
- Frame-to-frame stability beyond centroid:
      wasserstein[f] = W_2(attn_v[f], attn_v[f+1])     (2-Wasserstein
                       on the 2D distribution)
      top_k_iou[f]   = IoU(top-K-patches(attn_v[f]),
                           top-K-patches(attn_v[f+1])), K = 10% patches
- Normalization by target scale:
      bbox_diag = mean diagonal of target_bbox across frames
      normalized_drift = centroid_drift_px / bbox_diag
- Calibrated thresholds (replacing the current fixed pixel values):
      compute drift distribution on a held-out set of SUCCESSFUL
      trajectories for this (model, dataset); anchor at the 85th /
      95th percentile of that distribution.
- Per-camera + cross-stream allocation for multi-stream models
      (OFT primary + wrist): per-stream stability conditional on the
      "active" (highest-mass) stream; report cross-stream mass
      time-series.
- Ablation sanity check (calibration set, not per frame): zero the
      attended patch and confirm the action prediction changes; if not,
      the metric is measuring image structure, not model behaviour.
```

### Rationale

- **Mid-layer visual-grounding heads, not all-layers-all-heads
  average** — "How Multimodal LLMs Solve Image Tasks" (2508.20279)
  shows LLaVA-1.5 / Qwen2-VL have a clear stage structure: early
  layers do visual grounding, mid layers do lexical-visual
  integration, later layers shift attention away from vision tokens
  toward instruction tokens. For a spatial metric, mid layers are
  the right target. For OpenVLA specifically, the symbolic-state
  probing paper (arXiv:2502.04558) finds object/spatial probes peak
  in mid layers. "Head Pursuit" (arXiv:2510.21518) and "Functional
  Roles of Attention Heads in VLMs" (arXiv:2512.10300) show
  specialization is *sparse* and head-level — ~1% of heads is
  sufficient for many behaviours.
- **Sink-token masking before normalization** — "Understanding Sink
  Tokens in MLLMs" documents that BOS / padding / image-border
  tokens absorb disproportionate attention as a softmax artifact.
  Centroids computed without masking sinks are dominated by these
  artifacts. Sink tokens are pre-identified (high baseline activation
  on a calibration set) and masked to zero before normalizing.
- **Wasserstein-2 + top-K IoU + centroid drift** (three metrics)
  rather than centroid drift alone — centroid is lossy (2D
  distribution → point) and easily corrupted by sink residuals. W_2
  measures "how much attention mass had to move" (the formal analog
  of drift) and is the standard in saliency / eye-gaze literature
  (Liu et al. PLOS ONE 2017). Top-K IoU is robust to background
  diffusion. Centroid stays for interpretability.
- **Pointing accuracy as companion metric** — a model can be
  *stably wrong*. The drift metric alone can't distinguish "stable
  on target" from "stable on distractor". Pointing accuracy is the
  fraction of attention mass inside the target's bbox; combined with
  stability it gives a 2D verdict (anchored ✓/✗ × stable ✓/✗).
- **Normalization by bbox diagonal, not image diagonal** — a 50-px
  drift on a tiny target is huge; on a large target it's nothing.
  Eye-tracking convention treats >1° visual angle as a saccade — the
  analog here is "any single-step drift >1 bbox-radius is a
  saccade."
- **Per-camera plus cross-stream allocation** for multi-stream
  models — for OFT (primary + wrist) the LLM does joint cross-stream
  attention. Track-and-handover (wrist takes over near grasp) is
  expected and shouldn't be flagged as drift. Report per-stream
  stability conditional on which stream is "active" (highest mass).
- **Adebayo-style ablation sanity check** — periodically zero the
  attended patch and confirm the prediction changes. If not, the
  metric is measuring image structure (sink positions, edge density)
  rather than model behaviour.

### Excluded approaches

1. **Treating raw attention as causal evidence** (Jain & Wallace
   2019 / Wiegreffe & Pinter 2019 debate) — always validate with an
   ablation.
2. **Averaging across ALL layers and heads** — washes out the
   specialized visual-grounding signal.
3. **Computing centroid without masking sink tokens** — centroid is
   dominated by BOS / border artifacts.
4. **Silently filling (0.5, 0.5) when attention sums to zero** — a
   zero sum is an adapter bug and is raised, not absorbed into a
   fabricated centroid.
5. **Fixed pixel thresholds** for "drift" — must be normalized
   to image or target bbox scale.
   *Current-implementation note:* the shipped diagnostic deliberately
   USES fixed pixel thresholds (`drift_warn_px=30.0`,
   `drift_critical_px=70.0`) as a documented simplification.
   Normalized / calibrated thresholds (per the "Design target —
   not yet implemented" note above) are the intended replacement;
   until then this simplification is documented here rather than
   left implicit.

### Citations

- Jain & Wallace 2019. **Attention is not Explanation**, NAACL.
- Wiegreffe & Pinter 2019. **Attention is not not Explanation**
  (arXiv:1908.04626).
- Abnar & Zuidema 2020. **Quantifying Attention Flow in Transformers**
  (arXiv:2005.00928).
- "How Multimodal LLMs Solve Image Tasks" (arXiv:2508.20279).
- "Head Pursuit: Probing Attention Specialization in Multimodal
  Transformers" (arXiv:2510.21518).
- "Functional Roles of Attention Heads in VLMs"
  (arXiv:2512.10300).
- "Mechanistic Interpretability for Steering VLA Models"
  (arXiv:2509.00328).
- "Probing a VLA for Symbolic States" (arXiv:2502.04558).
- "AVA-VLA: Active Visual Attention for VLAs" (arXiv:2511.18960).
- "PosA-VLA: Pose-Conditioned Anchor Attention" (arXiv:2512.03724).
- "Understanding Sink Tokens in Multimodal LLMs" (OpenReview 2024).
- "Attention Debiasing for Token Pruning in VLMs"
  (arXiv:2508.17807).
- Selvaraju et al. 2017. **Grad-CAM** (arXiv:1610.02391).
- Liu et al. 2017. **Saliency attention via Earth Mover's Distance**,
  PLOS ONE.

---

## 5. Internal chunk consistency — can the multi-step lookahead be trusted?

### Question being answered

For VLAs that predict ACTION CHUNKS (OpenVLA-OFT k=8, π0 k=50,
GR00T-N1.7 H=16, Diffusion Policy, ACT, RDT), the comparison is: at frame t
the model predicts `chunk = [a_t, a_{t+1}, ...]`; at frame t+1 the
model predicts `chunk' = [a'_{t+1}, ...]`. Does `a_{t+1}` (the
prediction for t+1 made at time t) match `a'_{t+1}` (the prediction
for t+1 made at time t+1)? If yes → stable multi-step planning. If
no → resampling each frame, chunks beyond first step are noise,
running multi-step controllers will hurt.

### Method

```
Inputs:
    policy π (sample N=10-20 chunks per frame for stochastic policies)
    trajectory T with frames
    decay weight ρ = 0.9
    n_steps_to_compare k (default 1 — the full curve is reported)

1. For each frame f:
       chunks[f] = N samples of π(scene_f).action_chunk
                   (averaged for stochastic, single for deterministic)
       Within-frame std σ_within[f] = mean stddev of chunks[f] across
           samples
   chunk[f] = mean of chunks[f]

2. Per-step backward coherence (Bidirectional Decoding):
   for t in range(len(T) - 1):
       for τ in range(1, chunk_len):
           δ[t, τ] = ||chunk[t][τ] - chunk[t+τ][0]||_1 weighted-by-σ_d

3. Three normalizations:
   N1 (per-dim std):     δ / σ_dim_train         -- demos' per-dim std
   N2 (per-step delta):  δ / mean(||a[t+1] - a[t]||)  -- demos' steps
   N3 (within-sample):   δ / σ_within             -- model's own jitter

4. Safely-committable horizon (Mixture-of-Horizons style):
   h* = max τ such that δ_normalized[τ] < r · mean(δ_normalized[1..5])
   default r = 1.1
   Report per-frame h*, and the distribution across trajectory.

5. Cross-model headline scalar:
   fraction_usable = h* / chunk_len
   (π0's chunk_len=50 and OFT's chunk_len=8 are not raw-comparable)

6. Severity verdict (per-frame):
   if all τ in 1..k have δ_N3[τ] < 1:    PASS (within sampling noise)
   elif δ_N2[k] < 0.5:                    PASS (small in step-delta units)
   elif δ_N2[k] < 1.0:                    MODERATE (drift comparable to
                                                    one normal step)
   else:                                   CRITICAL (drift exceeds one
                                                    normal step)
```

### Rationale

- **Backward coherence with exponential decay** — Bidirectional
  Decoding [Liu et al. ICLR 2025, arXiv:2408.17355] gives the
  cleanest formal definition for chunk-vs-chunk agreement; it is
  used directly with ρ = 0.9.
- **N samples per frame for stochastic models** — π0 (flow-matching),
  GR00T (DiT flow-matching head), Diffusion Policy, RDT all have
  non-trivial sample-to-sample variance. Without averaging, single-
  sample "drift" between frames is mostly resampling noise.
  Adaptive Action Chunking [arXiv:2604.04161] samples N=20 chunks
  per frame for the same reason.
- **Three normalizations** — combined here rather than drawn from a
  single source. Per-dim std normalizes gripper-vs-translation
  mismatches; per-step delta gives physical interpretability
  ("drift = 50% of a normal action step"); within-sample std gives
  the model's own sampling floor (drift below this is
  indistinguishable from sampling jitter, not real plan revision).
- **Safely-committable horizon `h*`** — directly answers the
  user-facing question "how many steps can I trust?" Mixture-of-
  Horizons [arXiv:2511.19433] formalizes this self-calibrating
  threshold (r=1.1 × mean of first 5 steps). It is reported as the
  headline scalar.
- **Cross-model normalization via `h*/chunk_len`** — π0's 50 and
  OFT's 8 are physical horizons; their RATIOS are comparable, their
  raw values are not.

### Excluded approaches

1. **Single-sample chunk comparison for stochastic policies** —
   conflates decoding noise with plan revision.
2. **Raw L2 thresholds in opaque action units** — not interpretable
   across models or action dims.
3. **Headline "raw mean delta" without horizon** — doesn't tell the
   user how many steps are usable.
4. **Comparing π0 chunk_len=50 to OFT chunk_len=8 with raw scores**
   — the longer chunk has more noise to accumulate; normalize by
   horizon fraction.

### Citations

- Zhao et al. 2023. **ACT — Learning Fine-Grained Bimanual
  Manipulation**, RSS (arXiv:2304.13705).
- Chi et al. 2023. **Diffusion Policy** (arXiv:2303.04137).
- Black et al. 2024. **π0 — A Vision-Language-Action Flow Model for
  General Robot Control**, Physical Intelligence.
- Kim et al. 2025. **OpenVLA-OFT — Fine-Tuning VLA Models: Optimizing
  Speed and Success** (arXiv:2502.19645).
- NVIDIA 2025. **GR00T N1** (arXiv:2503.14734).
- Liu et al. 2024. **RDT-1B** (arXiv:2410.07864).
- Liu et al. ICLR 2025. **Bidirectional Decoding**
  (arXiv:2408.17355).
- Sty et al. **Mixture of Horizons in Action Chunking**
  (arXiv:2511.19433).
- **Adaptive Action Chunking at Inference-time**
  (arXiv:2604.04161).
- Black et al. 2025. **Real-Time Chunking** (pi.website).
- **Leave No Observation Behind** (arXiv:2509.23224).
- Simchowitz et al. 2025. **Action Chunking yields Exponential
  Improvements in BC** (arXiv:2507.09061).
- **Smoothness-Driven Metrics for Imitation Learning**
  (arXiv:2604.23000).

---

## 6. Calibration (shared infrastructure)

### Question being answered

Every diagnostic produces a raw L2 distance in the model's action
units. To compare across models (OpenVLA's 7-DOF Bridge actions vs
GR00T's 17-DOF state-action vs π0's chunk[0] subset), each raw distance
is normalized against two model-specific anchors:

- **noise_floor**: the model's residual sample-to-sample variation,
  measured directly between two `n_samples`-averaged predictions on
  identical input (frame 0) — i.e. already the post-averaging floor.
  (The √N reduction appears only in the `n_samples` derivation, Step 3,
  and in `signal_threshold` as 2σ/√k — not in `noise_floor` itself.)
- **typical_action_magnitude**: the median `||action||` of the model's
  averaged predictions across the trajectory.

### Algorithm

```
1. Single-sample noise probe (≥3 paired predict() calls on frame 0):
       σ_1 = mean ||a_1 - a_2|| across pairs

2. Single-sample magnitude probe (≥3 single predict() calls):
       m_1 = mean ||a||

3. Solve for n_samples from the math:
       averaged_noise ≈ σ_1 / sqrt(N)
       want: averaged_noise <= precision_target * m_1
       => N >= (σ_1 / (precision_target * m_1))^2
       (clipped to [1, max_n_samples=64])

4. Full noise + magnitude pass with n_samples averaging:
       baseline_magnitudes = [||averaged_predict(scene)||
                              for scene in trajectory]
       typical_action_magnitude = median(baseline_magnitudes)

       noise_floor = mean ||averaged_predict(frame_0)
                          - averaged_predict(frame_0)||
                     across n_noise_probes pairs

5. Normalize:
       score = max(0, raw_delta - noise_floor) / typical_action_magnitude
```

### Rationale

- **Per-trajectory calibration**, not global — different trajectories
  have different action magnitudes (approach vs grasp vs lift). A
  single global typical would over- or under-normalize whole
  trajectory segments.
- **Median, not mean**, of baseline magnitudes — robust to a few
  outlier frames (grasp transients).
- **`n_samples` derived from the math, not picked** — `n` is exactly
  enough to bound averaged noise below `precision_target × typical`.
  Deterministic models get `n=1` automatically (zero noise → divide
  blows up only if mag is also zero, which is checked). Highly
  stochastic models get whatever the math says, capped at 64.
- **Raise on degenerate `typical_action_magnitude`** — for a model
  that produces near-zero actions on every baseline frame, there's
  no scale to normalize against. This raises rather than silently
  returning 0, so a degenerate calibration is surfaced rather than
  hidden.
- **Single-sample magnitude in `n_samples` derivation**, averaged
  magnitude in the denominator — derivation is in terms of single
  samples; downstream comparisons are between averaged actions; the
  precision target absorbs the Jensen factor
  (`E[||avg||] ≤ E[||single||]`).

---

## 7. Per-model methodology notes

### OpenVLA-7B
- **Output**: 7-DOF discrete tokens (256 bins per dim) →
  continuous action via `bin_centers`. Deterministic at inference
  (`do_sample=False`).
- **Capabilities**: full mechanistic-interp suite — attention,
  hidden states, FFN activations, residual patching, neuron
  ablation. All four VLA adapters expose ATTENTION; OpenVLA's
  distinction is the full mechanistic-interp suite (hidden states,
  FFN activations, residual patching, neuron ablation), not
  attention alone.
- **Calibration**: `n_samples = 1` always (deterministic).
- **Chunk consistency**: N/A (no chunk).
- **Attention drift**: applies. Layer-adaptive selection within a
  mid-stack fractional band (OpenVLA 0.25–0.75; π0 0.25–0.85), meaned
  over heads; sink handling is model-dependent (OpenVLA: none —
  LLaMA-2 has no image-patch spatial sinks).
- **Action space**: bridge_orig 7-DOF [dx, dy, dz, drx, dry, drz,
  gripper], gripper ∈ {0, 1} after un-normalization.

### OpenVLA-OFT
- **Output**: 7-DOF chunk of 8 steps, parallel decoding,
  L1-regression head. Deterministic.
- **Capabilities**: INFERENCE + ATTENTION.
- **Calibration**: `n_samples = 1`.
- **Chunk consistency**: chunk_len=8, comparison meaningful per BID.
- **Attention drift**: exposed (multi-stream — primary + wrist).
- **Action space**: libero_spatial_no_noops 7-DOF, gripper ∈
  {0, 1}.

### π0 (Physical Intelligence openpi)
- **Output**: chunk of 10-50 steps via flow-matching DiT.
  STOCHASTIC — every `predict()` call yields a different chunk.
  Recommended `n_samples = 4-8` per calibration math (typical noise
  σ ≈ 0.08, typical magnitude ≈ 1.2 → n ≈ 4 for 5% precision).
- **Capabilities**: INFERENCE + ATTENTION (when use_pytorch=True).
- **Chunk consistency**: stochastic — average N samples first then
  compare averaged chunks.
- **Attention drift**: exposed when use_pytorch=True (PyTorch-converted
  checkpoint; PaliGemma/Gemma backbone); not on the JAX backend.
- **Action space**: pi0_libero 7-DOF, gripper ∈ {-1, +1}.
- **State convention**: pi_libero state is 8-dim
  [x, y, z, roll, pitch, yaw, gripper_l, gripper_r]. Distinct from
  community LIBERO splits (which use quaternion).

### GR00T-N1.7-3B (NVIDIA)
- **Output**: 17-dim chunk of 16 steps via DiT flow-matching head
  (4 Euler steps). STOCHASTIC.
- **Capabilities**: INFERENCE + ATTENTION (DiT motor-pathway
  cross-attention — dispersed; see §4 caveat).
- **Action layout**: per `_action_keys` = [eef_9d (0:9),
  gripper_position (9:10), joint_position (10:17)]. The 6D rotation
  inside eef_9d uses the continuous representation of Zhou et al.
  2019 — zeros BREAK SVD inside the decoder.
- **State layout** (droid_sample): same 17-dim layout as actions.
- **Chunk consistency**: stochastic — N-sample averaging required.
- **Modality dropout caveat**: state substitution MUST come from
  the dataset pool (not zeros, not last-frame). Zero state crashes
  the 6D-rotation decoder.

### ACT (lerobot ACTPolicy)
- **Output**: action chunk (default `chunk_size = 100`) from a
  DETR-style CVAE decoder. DETERMINISTIC at inference (the CVAE latent
  is the zero prior; the reparameterization sampling is training-only).
- **Inputs**: per-camera images + a proprioceptive-state token. NO
  language — `required_inputs.instruction = False`, so instruction
  perturbations auto-skip.
- **Capabilities**: INFERENCE + ATTENTION.
- **Calibration**: `n_samples = 1` (deterministic).
- **Chunk consistency**: applies (native chunk).
- **Attention drift**: the decoder cross-attention from the first
  action query to the encoder's image tokens (DETR-style; Carion et al.
  2020). The image tokens are a flattened ResNet feature grid
  (`H/stride × W/stride`, generally non-square — reported with an
  explicit `(h, w)` grid shape). This is the action pathway's
  attention, not a language-anchored object localizer. ACT has a SINGLE
  decoder layer (the original's 7 are a documented no-op bug; lerobot
  sets `n_decoder_layers = 1`) with 8 heads that **specialise** — verified
  per-head on the reference checkpoint, grounding heads concentrate on the
  end-effectors / contact point (interior-fraction ~0.85–1.0) while others
  are spatial sinks on the frame border (~0.42). So the clean map selects
  the single most interior-concentrated HEAD (`head_reduction =
  "select_interior"`) rather than averaging heads, which would blend the
  sink in. The image-attention fraction is ~1.0 across heads (the decoder
  query attends to image tokens, not the proprio/latent tokens).
- **Normalization**: lerobot has two checkpoint layouts. v0.5+ ships a
  saved processor pipeline (`policy_preprocessor.json`); pre-v0.5 bakes the
  normalization stats into `model.safetensors` buffers, which the current
  policy class discards as unexpected keys. The adapter handles both: it
  loads the saved pipeline when present, else reconstructs `dataset_stats`
  from the baked buffers and builds the pipeline from those — and RAISES if
  a feature that needs normalization has neither, rather than running
  un-normalized (lerobot's normalizer skips un-statted features silently).

### SmolVLA (lerobot SmolVLAPolicy)
- **Output**: action chunk (default `chunk_size = 50`) from a
  flow-matching action expert. STOCHASTIC — every `predict()` call
  samples noise and denoises; `n_samples` averaging applies (same
  treatment as π0 / GR00T).
- **Inputs**: per-camera images + language instruction + state.
- **Capabilities**: INFERENCE + ATTENTION.
- **Attention drift**: the SmolVLM2 prefix self-attention — the last
  instruction token's attention over the image patches, read from the
  KV-cache-fill forward that precedes denoising (the same
  visual-grounding signal as OpenVLA / π0, localization-head literature
  arXiv:2503.06287). The action expert's suffix→prefix attention is the
  action pathway and is not used for this map. Mid-stack fractional band
  0.25–0.85, mean over heads, query-averaged sink removal.
- **Normalization / tokenization**: the checkpoint's own pre/post-
  processor pipeline tokenizes the instruction and applies the model's
  stats; nothing is reconstructed by the adapter.

---

## 8. Master citation list (sorted by topic, then year)

### Causal mediation & SHAP foundations
- Pearl, J. **Causality** (2nd ed.), 2009.
- Štrumbelj, E. & Kononenko, I. 2014. *Explaining prediction models
  and individual predictions with feature contributions*. KIS.
- Lundberg, S. & Lee, S.-I. 2017. **SHAP**, NeurIPS
  (arXiv:1705.07874).
- Vig, J. et al. 2020. **Causal Mediation Analysis for Interpreting
  Neural NLP**, NeurIPS (arXiv:2004.12265).
- Janzing, D., Minorics, L. & Blöbaum, P. 2020. **Feature relevance
  quantification: a causal problem**, AISTATS (arXiv:1910.13413).
- Geiger, A. et al. 2023. **Causal Abstraction** (arXiv:2301.04709).
- Mueller, A. et al. 2025. **The Quest for the Right Mediator**,
  Computational Linguistics (arXiv:2408.01416).

### Feature attribution & saliency
- Zeiler, M. & Fergus, R. 2014. **Visualizing and Understanding
  Convolutional Networks**, ECCV (arXiv:1311.2901).
- Selvaraju, R. et al. 2017. **Grad-CAM** (arXiv:1610.02391).
- Sundararajan, M., Taly, A. & Yan, Q. 2017. **Integrated Gradients**
  (arXiv:1703.01365).
- Smilkov, D. et al. 2017. **SmoothGrad** (arXiv:1706.03825).
- Petsiuk, V., Das, A. & Saenko, K. 2018. **RISE**, BMVC
  (arXiv:1806.07421).
- Adebayo, J. et al. 2018. **Sanity Checks for Saliency Maps**,
  NeurIPS (arXiv:1810.03292).
- Hooker, S. et al. 2019. **ROAR**, NeurIPS (arXiv:1806.10758).
- Hooker, G. & Mentch, L. 2019. **Please Stop Permuting Features**
  (arXiv:1905.03151).
- Sturmfels, P., Lundberg, S. & Lee, S.-I. 2020. **Visualizing the
  Impact of Feature Attribution Baselines**, Distill.
- Rong, Y. et al. 2022. **ROAD** (arXiv:2202.00449).
- Suvorov, R. et al. 2022. **LaMa — Resolution-robust Large Mask
  Inpainting with Fourier Convolutions**, WACV (arXiv:2109.07161).
- Hurley, N. & Rickard, S. 2009. **Comparing Measures of Sparsity**,
  IEEE TIT.
- Cooper et al. 2022. **Metrics for saliency map evaluation**
  (arXiv:2201.13291).

### Attention interpretation
- Jain, S. & Wallace, B. 2019. **Attention is not Explanation**,
  NAACL.
- Wiegreffe, S. & Pinter, Y. 2019. **Attention is not not
  Explanation** (arXiv:1908.04626).
- Abnar, S. & Zuidema, W. 2020. **Quantifying Attention Flow in
  Transformers** (arXiv:2005.00928).
- Carion, N. et al. 2020. **DETR — End-to-End Object Detection with
  Transformers**, ECCV (arXiv:2005.12872). Decoder-query cross-attention
  visualization — the basis for the ACT attention map.

### Detection / segmentation for phrase grounding
- Liu, S. et al. 2024. **GroundingDINO**, ECCV (arXiv:2303.05499).
- Ren, J. et al. 2024. **DINO-X** (arXiv:2411.14347).
- Xiao, B. et al. 2024. **Florence-2**, CVPR.
- Ravi, N. et al. 2024. **SAM 2**.
- Meta 2025. **SAM 3 / Segment Anything with Concepts**.

### Robot policy chunking & evaluation
- Zhao, T. et al. 2023. **ACT**, RSS (arXiv:2304.13705).
- Chi, C. et al. 2023. **Diffusion Policy** (arXiv:2303.04137).
- Black, K. et al. 2024. **π0 — A VLA Flow Model**, Physical
  Intelligence.
- Liu, S. et al. 2024. **RDT-1B** (arXiv:2410.07864).
- NVIDIA 2025. **GR00T N1** (arXiv:2503.14734).
- Kim, J. et al. 2025. **OpenVLA-OFT** (arXiv:2502.19645).
- Liu, Z. et al. ICLR 2025. **Bidirectional Decoding**
  (arXiv:2408.17355).
- Sty et al. **Mixture of Horizons** (arXiv:2511.19433).
- **Adaptive Action Chunking** (arXiv:2604.04161).
- Black, K. et al. 2025. **Real-Time Chunking**.
- Simchowitz, M. et al. 2025. **Action Chunking yields Exponential
  Improvements in BC** (arXiv:2507.09061).

### VLA-specific interpretability & robustness
- Shukor, M. et al. 2025. **SmolVLA — A Vision-Language-Action Model for
  Affordable and Efficient Robotics** (arXiv:2506.01844).
- Hancock, A. et al. 2024. **BYOVLA** (arXiv:2410.01971).
- "Mechanistic Interpretability for Steering VLA Models"
  (arXiv:2509.00328).
- "Probing a VLA for Symbolic States" (arXiv:2502.04558).
- Lin, B. et al. 2025. **Do You Need Proprioceptive States?**
  (arXiv:2509.18644).
- Geng, X. et al. 2025. **LIBERO-PRO** (arXiv:2510.03827).
- "LIBERO-Plus" (arXiv:2510.13626).
- "Head Pursuit: Probing Attention Specialization"
  (arXiv:2510.21518).
- "Functional Roles of Attention Heads in VLMs"
  (arXiv:2512.10300).
- "AVA-VLA: Active Visual Attention" (arXiv:2511.18960).
- "PosA-VLA: Pose-Conditioned Anchor Attention" (arXiv:2512.03724).
- de Haan, P., Jayaraman, D. & Levine, S. 2019. **Causal Confusion
  in Imitation Learning**, NeurIPS (arXiv:1905.11979).
- "Shortcut Learning in Generalist Robot Policies" (CoRL 2025,
  arXiv:2508.06426).
- CAST (arXiv:2508.13446).
- "When Vision Overrides Language" (arXiv:2602.17659).
- AR-VLA (arXiv:2603.10126).

### Structured representations
- Zhou, Y. et al. 2019. **On the Continuity of Rotation
  Representations in Neural Networks**, CVPR (arXiv:1812.07035).
