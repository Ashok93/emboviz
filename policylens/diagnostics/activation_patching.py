"""Activation patching — the gold-standard causal mediation diagnostic.

Method (Heimersheim & Nanda 2024, arXiv 2404.15255):

  1. **Clean run**: run model on the baseline scene; cache residual-stream
     output at each layer (at the action-prediction position).
  2. **Corrupted run**: run model on a perturbed scene (e.g., noun_swap).
     Note the corrupted action.
  3. **Patched runs**: for each layer L in turn, take the *clean cached
     residual at L* and inject it into the *corrupted run* at the same
     position. Record the resulting action.
  4. **Recovery curve**: for each L, measure how much the patched action
     'recovers' toward the clean action vs the corrupted action.

The layer where recovery is highest is where the perturbed signal 'lives' —
the layer that, when fixed, restores baseline behaviour. This is the
strongest mechanistic claim available short of full circuit analysis.

Use cases:
  • "We patched layer 14 from a clean run into the noun-swapped run, and
    the action recovered 87% of the way toward clean. The noun is being
    routed at L14."
  • "No single layer recovers >20%, so the routing is distributed across
    many layers."
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from policylens.core.results import DiagnosticResult, Severity
from policylens.core.types import Scene, TokenSelector
from policylens.diagnostics.base import Diagnostic
from policylens.models.protocol import Capability, NotSupported, VLAModel
from policylens.perturb.base import Perturber


class ActivationPatchingDiagnostic(Diagnostic):
    """Layer-by-layer patching to localize where a perturbation 'lives'.

    Pair this with a perturber (typically NounSwapPerturber). For each
    variant the perturber yields, we patch each layer in turn and measure
    action recovery.
    """

    required_capabilities = (
        Capability.INFERENCE
        | Capability.HIDDEN_STATES
        | Capability.ACTIVATION_PATCHING
    )

    def __init__(
        self,
        perturber: Perturber,
        layer_indices: Optional[list[int]] = None,
        # Severity thresholds calibrated against literature reports
        # of fusion-band layers carrying 15–40 % of signal individually
        # (MINT 2025, "Few Heads for Visual Grounding" 2025).
        strong_threshold: float = 0.40,
        moderate_threshold: float = 0.15,
    ):
        self.perturber = perturber
        self.layer_indices = layer_indices         # None → use evenly-spaced selection
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.name = f"activation_patching.{perturber.name}"
        self.axis = f"internal.activation_patching"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks ACTIVATION_PATCHING")

        # Pick layers to patch: by default a dense sweep across the middle
        # third of the network — the documented fusion band where vision /
        # language integration happens (MINT, arXiv 2503.06287).
        n_layers = model.num_layers or 32
        if self.layer_indices is None:
            lo = n_layers // 4
            hi = (3 * n_layers) // 4
            # 12 layers in the fusion band — finer-grained than 8.
            layers_to_patch = sorted(set(int(x) for x in np.linspace(lo, hi, num=12)))
        else:
            layers_to_patch = list(self.layer_indices)

        # 1. Baseline (clean) action + cached residuals
        baseline = model.predict(scene.image, scene.instruction)
        try:
            clean_hs = model.extract_hidden_states(
                scene.image, scene.instruction,
                layers_to_patch,
                TokenSelector(relative="before_action"),
            )
        except NotSupported as e:
            return self._not_applicable(model, scene, str(e))

        clean_residuals = {
            li: clean_hs.states[idx]
            for idx, li in enumerate(layers_to_patch)
        }

        # 2. For each perturbation variant, patch each layer and measure recovery
        per_variant_records = []
        variant_count = 0
        # Aggregate per-layer recovery across all variants.
        per_layer_recoveries: dict[int, list[float]] = {li: [] for li in layers_to_patch}

        for variant in self.perturber.variants(scene):
            v_scene = variant.scene
            corrupted = (
                model.predict_with_image(v_scene.image, v_scene.instruction)
                if self.perturber.domain == "image"
                else model.predict(v_scene.image, v_scene.instruction)
            )
            d_clean_corrupt = float(model.compare_actions(baseline, corrupted))
            if d_clean_corrupt < 1e-6:
                # Corrupted action is identical to clean — no perturbation to recover from.
                continue

            per_layer_for_variant = {}
            for li in layers_to_patch:
                patched = model.predict_with_residual_patch(
                    v_scene.image, v_scene.instruction,
                    patches={li: clean_residuals[li]},
                )
                d_clean_patched = float(model.compare_actions(baseline, patched))
                # Recovery: 1.0 = action fully matches clean, 0.0 = no recovery.
                recovery = 1.0 - (d_clean_patched / d_clean_corrupt)
                # Clip to [-0.5, 1.5] so a "worse than corrupted" patch shows as < 0.
                recovery = float(np.clip(recovery, -0.5, 1.5))
                per_layer_for_variant[li] = recovery
                per_layer_recoveries[li].append(recovery)

            per_variant_records.append({
                "variant_id": variant.variant_id,
                "description": variant.description,
                "instruction": v_scene.instruction,
                "d_clean_corrupt": d_clean_corrupt,
                "per_layer_recovery": per_layer_for_variant,
            })
            variant_count += 1

        if variant_count == 0:
            return self._not_applicable(model, scene,
                "perturber produced no variants with corrupted action different from clean")

        # 3. Aggregate per-layer mean recovery
        mean_recovery_per_layer = {
            li: float(np.mean(per_layer_recoveries[li]))
            for li in layers_to_patch
        }
        best_layer = max(mean_recovery_per_layer, key=mean_recovery_per_layer.get)
        best_recovery = mean_recovery_per_layer[best_layer]

        if best_recovery >= self.strong_threshold:
            sev = Severity.INFO
            verdict = (
                f"Layer L{best_layer} is the principal locus of {self.perturber.name} "
                f"signal — patching clean→corrupted at L{best_layer} recovers "
                f"{best_recovery:.0%} of the action by itself. Strong single-layer localization."
            )
        elif best_recovery >= self.moderate_threshold:
            sev = Severity.INFO
            verdict = (
                f"Layer L{best_layer} carries the largest single-layer share of "
                f"{self.perturber.name} signal ({best_recovery:.0%} recovery). The signal is "
                f"primarily there but also distributed across other layers. Consistent with "
                f"the fusion-band finding (MINT 2025)."
            )
        else:
            sev = Severity.MODERATE
            verdict = (
                f"No single layer recovers ≥{self.moderate_threshold:.0%} of the action. "
                f"The {self.perturber.name} signal is highly distributed; multi-layer or "
                f"head-level interventions would be needed to localize it."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=best_recovery,
            severity=sev,
            direction="higher_is_worse",   # higher = stronger localization signal
            explanation=verdict,
            per_variant={f"L{li}": mean_recovery_per_layer[li] for li in layers_to_patch},
            raw={
                "layers_tested": layers_to_patch,
                "mean_recovery_per_layer": {str(k): v for k, v in mean_recovery_per_layer.items()},
                "best_layer": best_layer,
                "best_recovery": best_recovery,
                "per_variant": per_variant_records,
            },
        )
