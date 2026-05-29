"""Modality dropout diagnostic — does the policy USE each input modality?

For each declared input modality (image-per-camera, state, gripper,
action_history, instruction), draw K samples from a marginal-distribution
pool (built from OTHER episodes in the dataset), replace the modality with
each sample, and measure two quantities:

  • **intervention magnitude** Δ_in — distance between the substitute
    and the original value in the modality's natural metric (L2 for
    state / action_history, abs-diff for gripper, Jaccard for
    instruction, pixel-L2 for image).
  • **response magnitude** Δ_out — normalized L2 change in the model's
    action under the substitution.

The verdict combines them per the causal-mediation principle:

  • If mean Δ_in is below the pool's "minimum meaningful intervention"
    threshold (25th percentile of intra-pool pairwise distances), the
    test is UNTESTABLE — the substitutes were too similar to the
    current value to count as a real intervention. We refuse to call
    "ignored."
  • Otherwise the verdict is USED / PARTIAL / IGNORED based on the
    normalized Δ_out vs noise-floor and grounded thresholds.

Implementation is faithful to the literature:
  - SHAP / marginal sampling (Janzing-Minorics-Blöbaum 2020,
    arXiv:1910.13413; Lundberg & Lee 2017, arXiv:1705.07874)
  - Cross-episode sampling avoids Hooker-Mentch extrapolation
    (arXiv:1905.03151)
  - Per-modality natural distance metrics (state on SO(3) → L2 in joint
    coords; instructions → Jaccard token overlap as a lightweight
    embedding-free proxy)
  - K=20 default per RISE convention scaled down (RISE used N=8000 for
    binary masks; for full-modality substitution K=20 gives
    Monte-Carlo SE ≈ 1/√20 ≈ 22 %)
  - Intervention validity gate per Geiger et al. 2023 "Causal
    Abstraction" (arXiv:2301.04709) — refuse verdict when intervention
    magnitude is below the modality's natural scale.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from emboviz.calibration import ModelCalibration, averaged_predict, averaged_predict_batch
from emboviz.core.observations import (
    Proprioception,
)
from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import ActionResult, Scene
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel
from emboviz.modality_pools import ModalityPool, _distance


class ModalityDropoutDiagnostic(Diagnostic):
    """Marginal-distribution modality dropout (the proper SHAP one).

    Args:
        pool: ModalityPool built from OTHER episodes in the dataset.
            Required.
        calibration: per-model anchors (typical_action_magnitude,
            noise_floor, n_samples). Required for normalized scoring
            and stochastic averaging.
        k_samples: substitutions per modality per query frame.
            Default 20 (Monte-Carlo SE ≈ 22 %, manageable inference cost).
        cameras: which cameras to test individually. None = every
            declared camera.
        noise_floor_score: normalized Δ_out below which the modality is
            "ignored" (after intervention-validity gate passes).
        grounded_threshold: normalized Δ_out above which the modality is
            "genuinely used".
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        pool: ModalityPool,
        calibration: ModelCalibration,
        k_samples: int = 20,
        cameras: Optional[list[str]] = None,
        noise_floor_score: float = 0.05,
        grounded_threshold: float = 0.30,
        seed: int = 0,
    ):
        if pool is None:
            raise ValueError(
                "ModalityDropoutDiagnostic: pool is required. The "
                "diagnostic refuses to invent substitutions — call "
                "build_modality_pool(dataset, current_episode, ...) "
                "and pass it in."
            )
        if calibration is None:
            raise ValueError(
                "ModalityDropoutDiagnostic: calibration is required. "
                "Normalized scoring + multi-sample averaging both need "
                "the model calibration."
            )
        self.pool = pool
        self.calibration = calibration
        self.k_samples = int(k_samples)
        self.cameras = cameras
        self.noise_floor_score = float(noise_floor_score)
        self.grounded_threshold = float(grounded_threshold)
        self.seed = int(seed)
        self.name = "modality_dropout"
        self.axis = "input.modality_dropout"

    def run(
        self, model: VLAModel, scene: Scene,
        *, baseline: Optional[ActionResult] = None,
    ) -> DiagnosticResult:
        """Run marginal-distribution modality dropout for ``scene``.

        ``baseline`` is an optional precomputed unperturbed prediction —
        the runner shares it across diagnostics so we don't re-pay
        ``n_samples`` forward passes for the same baseline in every one.
        """
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        rng = np.random.default_rng(self.seed)
        n_samples = self.calibration.n_samples
        if baseline is None:
            baseline = averaged_predict(model, scene, n_samples)

        reqs = model.required_inputs
        per_modality: dict[str, dict] = {}

        # 1. Image per camera
        requested_cams = (
            sorted(self.cameras) if self.cameras is not None
            else sorted(reqs.cameras)
        )
        for cam in requested_cams:
            modality = f"image:{cam}"
            if cam not in scene.observations.images:
                per_modality[modality] = self._skip(
                    f"camera '{cam}' not present in scene "
                    f"(scene has {sorted(scene.observations.images)}); "
                    "load it in the dataset adapter."
                )
                continue
            if not self.pool.has(modality):
                per_modality[modality] = self._skip(
                    f"pool has no image samples for camera '{cam}'."
                )
                continue
            current = np.asarray(scene.observations.images[cam].data)
            per_modality[modality] = self._run_one(
                modality, model, scene, baseline, current, n_samples, rng,
                apply_fn=lambda s, v: s.with_image(_as_pil(v), camera=cam),
            )

        # 2. State
        if reqs.state and scene.observations.state is not None:
            if not self.pool.has("state"):
                per_modality["state"] = self._skip("pool has no state samples")
            else:
                current = np.asarray(scene.observations.state.values, dtype=np.float32)
                def apply_state(s, v):
                    new_state = Proprioception(
                        values=np.asarray(v, dtype=current.dtype).reshape(current.shape),
                        convention=s.observations.state.convention,
                    )
                    return replace(s, observations=replace(s.observations, state=new_state))
                per_modality["state"] = self._run_one(
                    "state", model, scene, baseline, current, n_samples, rng,
                    apply_fn=apply_state,
                )

        # 3. Gripper
        if reqs.gripper and scene.observations.gripper is not None:
            if not self.pool.has("gripper"):
                per_modality["gripper"] = self._skip("pool has no gripper samples")
            else:
                current = float(scene.observations.gripper.value)
                def apply_grip(s, v):
                    new_grip = replace(s.observations.gripper, value=float(v))
                    return replace(s, observations=replace(s.observations, gripper=new_grip))
                per_modality["gripper"] = self._run_one(
                    "gripper", model, scene, baseline, current, n_samples, rng,
                    apply_fn=apply_grip,
                )

        # 4. Action history
        if reqs.action_history and scene.observations.action_history is not None:
            if not self.pool.has("action_history"):
                per_modality["action_history"] = self._skip("pool has no action_history samples")
            else:
                current = np.asarray(scene.observations.action_history.actions, dtype=np.float32)
                def apply_hist(s, v):
                    new_hist = replace(
                        s.observations.action_history,
                        actions=np.asarray(v, dtype=current.dtype).reshape(current.shape),
                    )
                    return replace(s, observations=replace(s.observations, action_history=new_hist))
                per_modality["action_history"] = self._run_one(
                    "action_history", model, scene, baseline, current, n_samples, rng,
                    apply_fn=apply_hist,
                )

        # 5. Instruction
        if reqs.instruction and scene.instruction is not None:
            if not self.pool.has("instruction"):
                per_modality["instruction"] = self._skip(
                    "pool has no different-task instruction samples"
                )
            else:
                current = str(scene.instruction)
                def apply_instr(s, v):
                    return s.with_instruction(str(v))
                per_modality["instruction"] = self._run_one(
                    "instruction", model, scene, baseline, current, n_samples, rng,
                    apply_fn=apply_instr,
                )

        if not per_modality:
            return self._not_applicable(
                model, scene,
                "no testable modality found — model declares no inputs OR the scene "
                "is missing all declared modality fields."
            )

        # Roll up verdicts
        verdicts = {m: r["verdict"] for m, r in per_modality.items()}
        n_used        = sum(1 for v in verdicts.values() if v == "USED")
        n_ignored     = sum(1 for v in verdicts.values() if v == "IGNORED")
        n_partial     = sum(1 for v in verdicts.values() if v == "PARTIAL")
        n_below_noise = sum(1 for v in verdicts.values() if v == "BELOW_NOISE")
        n_untestable  = sum(1 for v in verdicts.values() if v == "UNTESTABLE")

        # Severity priority (worst → best). Severity is internal sort
        # only — the user-facing Finding below explains in plain English.
        #   IGNORED      → CRITICAL   (demonstrably unused real input)
        #   PARTIAL      → MODERATE   (real response, but weak)
        #   USED only    → PASS
        #   BELOW_NOISE  → UNKNOWN    (response within sampling noise —
        #                 cannot distinguish ignored from jitter; try
        #                 a more dynamic frame or larger K)
        #   UNTESTABLE   → UNKNOWN    (intervention too similar to
        #                 current value — dataset lacks variety)

        def _list(group: str) -> list[str]:
            return [m for m, v in verdicts.items() if v == group]

        ignored      = _list("IGNORED")
        partial      = _list("PARTIAL")
        used         = _list("USED")
        below_noise  = _list("BELOW_NOISE")
        untestable   = _list("UNTESTABLE")

        raw_numbers = {
            "n_used":        n_used,
            "n_ignored":     n_ignored,
            "n_partial":     n_partial,
            "n_below_noise": n_below_noise,
            "n_untestable":  n_untestable,
            "k_samples":     self.k_samples,
            "per_modality_verdict":          verdicts,
            "per_modality_intervention_mag": {
                m: r.get("mean_intervention_mag", float("nan"))
                for m, r in per_modality.items()
            },
            "per_modality_response_norm":    {
                m: r.get("mean_response_normalized", float("nan"))
                for m, r in per_modality.items()
            },
        }

        if n_ignored > 0:
            sev = Severity.CRITICAL
            finding = Finding(
                observed=(
                    f"When we replaced {', '.join(ignored)} with samples "
                    f"drawn from other episodes, the model's predicted "
                    f"action barely moved — under "
                    f"{self.noise_floor_score:.0%} of typical action "
                    f"magnitude (but above the model's per-call sampling "
                    f"noise, so this is a real signal, just small). "
                    f"K={self.k_samples} substitutions per modality."
                ),
                meaning=(
                    f"The policy is largely not using "
                    f"{', '.join(ignored)} to decide what action to take "
                    f"on this frame. If your task requires that input "
                    f"(e.g. instruction for a language-conditioned task), "
                    f"that's a problem."
                ),
                next_step=(
                    "Check the same modality across more frames. If it "
                    "stays ignored on every frame, your fine-tune may "
                    "have collapsed that modality. If it varies, the "
                    "input only matters at certain phases."
                ),
                raw_numbers=raw_numbers,
            )
        elif n_partial > 0:
            sev = Severity.MODERATE
            finding = Finding(
                observed=(
                    f"The model responds to {', '.join(partial)} but "
                    f"only weakly — between sampling noise and "
                    f"{self.grounded_threshold:.0%} of typical action "
                    f"magnitude. "
                    + (f"{', '.join(used)} produced a strong response. "
                       if used else "")
                ),
                meaning=(
                    "The policy reads these inputs but they're not the "
                    "primary driver of its actions on this frame."
                ),
                next_step=(
                    "Compare strong vs weak modalities across the "
                    "trajectory. If a modality you EXPECT to matter "
                    "(instruction, target camera) is consistently weak, "
                    "the model is leaning on other inputs more than "
                    "intended."
                ),
                raw_numbers=raw_numbers,
            )
        elif n_used > 0 and n_below_noise == 0 and n_untestable == 0:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"All {n_used} testable modality(ies) "
                    f"({', '.join(used)}) produced strong responses "
                    f"when intervened on — the model's action shifted "
                    f"substantially when each was replaced."
                ),
                meaning=(
                    "The policy is consuming every declared input on "
                    "this frame. Healthy modality usage."
                ),
                next_step="No action needed for this frame.",
                raw_numbers=raw_numbers,
            )
        elif n_used > 0:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"Strong response on {', '.join(used)}. "
                    + (f"{', '.join(below_noise)} response was within the "
                       f"model's per-call sampling noise. " if below_noise else "")
                    + (f"{', '.join(untestable)} substitutions were too "
                       f"similar to the current frame to test." if untestable else "")
                ),
                meaning=(
                    "The modalities we could test are being used. The "
                    "others are inconclusive — not necessarily ignored, "
                    "just not testable on this frame."
                ),
                next_step=(
                    "Re-run on a more dynamic mid-episode frame, or "
                    "supply a more varied dataset, to test the "
                    "inconclusive modalities."
                ),
                raw_numbers=raw_numbers,
            )
        elif n_below_noise > 0 or n_untestable > 0:
            sev = Severity.UNKNOWN
            finding = Finding(
                observed=(
                    (f"{n_below_noise} modality(ies) ({', '.join(below_noise)}) "
                     f"produced responses within the model's per-call "
                     f"sampling noise. " if below_noise else "")
                    + (f"{n_untestable} ({', '.join(untestable)}) had "
                       f"substitutions too similar to the current frame "
                       f"to count as a real intervention." if untestable else "")
                ),
                meaning=(
                    "We cannot tell on this frame whether these inputs "
                    "are used or ignored — the diagnostic refuses to "
                    "guess."
                ),
                next_step=(
                    "Use a more dynamic frame (mid-trajectory during "
                    "active manipulation) or a more varied dataset; "
                    "increase K samples for tighter statistics."
                ),
                raw_numbers=raw_numbers,
            )
        else:
            sev = Severity.UNKNOWN
            finding = Finding(
                observed="No modality produced a usable verdict on this frame.",
                meaning="The diagnostic could not test any input.",
                next_step="Check that the model's required_inputs match what the scene actually carries.",
                raw_numbers=raw_numbers,
            )
        verdict_str = f"{finding.observed} {finding.meaning} {finding.next_step}"

        scalar = float(n_ignored)   # higher = worse

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=scalar,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict_str,
            finding=finding,
            per_variant={
                f"{m}:Δ_in":  r.get("mean_intervention_mag", float("nan"))
                for m, r in per_modality.items()
            } | {
                f"{m}:Δ_out": r.get("mean_response_normalized", float("nan"))
                for m, r in per_modality.items()
            } | {
                f"{m}:verdict": r["verdict"] for m, r in per_modality.items()
            },
            raw={
                "per_modality":           per_modality,
                "n_used":                 n_used,
                "n_ignored":              n_ignored,
                "n_partial":              n_partial,
                "n_below_noise":          n_below_noise,
                "n_untestable":           n_untestable,
                "k_samples":              self.k_samples,
                "pool_size":              {
                    "state":           len(self.pool.state_samples),
                    "gripper":         len(self.pool.gripper_samples),
                    "action_history":  len(self.pool.action_history_samples),
                    "instruction":     len(self.pool.instruction_samples),
                    "image":           {c: len(s) for c, s in self.pool.image_samples.items()},
                },
                "pool_ref_distance":      self.pool.ref_distance,
                "pool_sampled_episodes":  self.pool.metadata.get("sampled_episodes"),
                "calibration_used":       self.calibration.to_summary(),
                "noise_floor_score":      self.noise_floor_score,
                "grounded_threshold":     self.grounded_threshold,
            },
        )

    # ----------------------------------------------------------------
    def _skip(self, reason: str) -> dict:
        return {"verdict": "UNTESTABLE", "skip_reason": reason}

    def _run_one(
        self, modality: str, model, scene, baseline, current_value,
        n_samples: int, rng: np.random.Generator,
        *, apply_fn,
    ) -> dict:
        """Run K-sample marginal-dropout for one modality on one scene."""
        samples = self.pool.sample(
            modality, self.k_samples, rng, current_value=current_value,
        )
        if not samples:
            return self._skip(
                f"pool returned 0 samples for {modality} after filtering "
                "duplicates of current value."
            )

        # Build every substituted scene first — the K marginal substitutions
        # are mutually independent, so we submit them as ONE batch instead of
        # K sequential forwards. apply_fn failure skips the whole modality
        # (it means the substitute is incompatible with this scene); a batch
        # predict failure does too — identical outcome to the old per-sample
        # early-return, just discovered once instead of on the first sample.
        perturbed_scenes = []
        for sub in samples:
            try:
                perturbed_scenes.append(apply_fn(scene, sub))
            except Exception as e:
                return self._skip(
                    f"apply_fn raised on {modality}: {type(e).__name__}: {e}"
                )
        try:
            preds = averaged_predict_batch(model, perturbed_scenes, n_samples)
        except Exception as e:
            return self._skip(
                f"model.predict raised on {modality} substitute: "
                f"{type(e).__name__}: {e}"
            )

        intervention_mags = []
        response_norms    = []
        response_raws     = []
        for sub, pred in zip(samples, preds):
            d_in = _distance(modality, sub, current_value)
            d_out_raw = float(np.linalg.norm(pred.action - baseline.action))
            d_out_norm = self.calibration.normalize(d_out_raw)

            intervention_mags.append(d_in)
            response_norms.append(d_out_norm)
            response_raws.append(d_out_raw)

        mean_in       = float(np.mean(intervention_mags))
        median_in     = float(np.median(intervention_mags))
        mean_out_norm = float(np.mean(response_norms))
        mean_out_raw  = float(np.mean(response_raws))
        std_out_norm  = float(np.std(response_norms))
        ratio = (mean_out_norm / mean_in) if mean_in > 1e-12 else float("inf")

        # Intervention-validity gate.
        ref = self.pool.ref_distance.get(modality, 0.0)
        if mean_in < ref:
            return {
                "verdict":                       "UNTESTABLE",
                "skip_reason":                   (
                    f"mean intervention magnitude {mean_in:.4f} is below "
                    f"the pool's 25th-percentile pairwise distance "
                    f"({ref:.4f}). The substitutions are too similar to "
                    "the current value to count as a real intervention. "
                    "More variety in the dataset would help."
                ),
                "n_samples_used":                int(len(samples)),
                "mean_intervention_mag":         mean_in,
                "median_intervention_mag":       median_in,
                "ref_min_intervention":          float(ref),
                "mean_response_normalized":      mean_out_norm,
                "mean_response_raw":             mean_out_raw,
                "std_response_normalized":       std_out_norm,
                "sensitivity_ratio":             ratio,
                "intervention_magnitudes":       intervention_mags,
                "response_normalized_per_sample": response_norms,
            }

        # Real intervention happened. Two-stage verdict:
        #
        #   1) Is the response statistically distinguishable from the
        #      model's own sampling noise? If not, we cannot tell
        #      "model ignored input" apart from "model jitters by this
        #      magnitude on every call regardless of input." → BELOW_NOISE
        #
        #   2) If yes, compare to the strength thresholds:
        #      IGNORED  (real but tiny — < noise_floor_score of typical)
        #      PARTIAL  (real, modest)
        #      USED     (real, strong — > grounded_threshold of typical)
        #
        # The noise threshold is derived from the model's calibrated
        # noise floor (per-call sigma) divided by sqrt(K) — the SE of
        # the K-sample mean. See ModelCalibration.signal_threshold_normalized.
        signal_threshold = self.calibration.signal_threshold_normalized(
            k_samples=int(len(samples)),
        )
        if mean_out_norm < signal_threshold:
            verdict = "BELOW_NOISE"
        elif mean_out_norm < self.noise_floor_score:
            verdict = "IGNORED"
        elif mean_out_norm < self.grounded_threshold:
            verdict = "PARTIAL"
        else:
            verdict = "USED"

        return {
            "verdict":                       verdict,
            "n_samples_used":                int(len(samples)),
            "mean_intervention_mag":         mean_in,
            "median_intervention_mag":       median_in,
            "ref_min_intervention":          float(ref),
            "mean_response_normalized":      mean_out_norm,
            "mean_response_raw":             mean_out_raw,
            "std_response_normalized":       std_out_norm,
            "signal_threshold_normalized":   signal_threshold,
            "sensitivity_ratio":             ratio,
            "intervention_magnitudes":       intervention_mags,
            "response_normalized_per_sample": response_norms,
        }


def _as_pil(v):
    """Tolerant: accept PIL.Image, ndarray, or whatever and return PIL."""
    from PIL import Image
    if isinstance(v, Image.Image):
        return v
    arr = np.asarray(v, dtype=np.uint8)
    return Image.fromarray(arr)
