"""Cross-modal attention diagnostic.

Compares image-token attention under two instructions (typically noun A
vs noun B). High Jensen-Shannon divergence per (layer, head) means the
input pathway DOES route on the noun; low JS means the noun is being
ignored by the attention machinery.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from emboviz.core.divergences import jensen_shannon
from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, TokenSelector
from emboviz.diagnostics.base import Diagnostic
from emboviz.metrics.attention_js import AttentionJSMetric
from emboviz.models.protocol import Capability, NotSupported, VLAModel


class CrossModalAttentionDiagnostic(Diagnostic):
    """How differently does the model attend when a noun is swapped?"""

    required_capabilities = Capability.INFERENCE | Capability.ATTENTION

    def __init__(self, noun_a: str, noun_b: str, top_n_heads: int = 10):
        self.noun_a = noun_a
        self.noun_b = noun_b
        self.top_n_heads = top_n_heads
        self.name = f"attention.{noun_a}_vs_{noun_b}"
        self.axis = "vision.binding_grounding"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks ATTENTION capability")

        # Build the counterfactual instruction (replace noun_a with noun_b).
        instr_a = scene.instruction or ""
        if not instr_a:
            return self._not_applicable(model, scene, "scene has no instruction")
        instr_b = re.sub(rf"\b{re.escape(self.noun_a)}\b", self.noun_b, instr_a,
                         flags=re.IGNORECASE)
        if instr_a == instr_b:
            return self._not_applicable(model, scene,
                f"'{self.noun_a}' not found in instruction")

        try:
            attn_a = model.extract_attention(
                scene.with_instruction(instr_a), TokenSelector(word=self.noun_a),
            )
            attn_b = model.extract_attention(
                scene.with_instruction(instr_b), TokenSelector(word=self.noun_b),
            )
        except NotSupported as e:
            return self._not_applicable(model, scene, str(e))

        metric = AttentionJSMetric()
        js_per_head = metric.compute(attn_a, attn_b)   # (L, H)

        # Rank heads
        flat = js_per_head.flatten()
        order = np.argsort(-flat)[: self.top_n_heads]
        top_heads = [
            {"layer": int(idx // js_per_head.shape[1]),
             "head": int(idx % js_per_head.shape[1]),
             "js": float(flat[idx])}
            for idx in order
        ]
        max_js = float(flat.max())
        mean_js = float(flat.mean())

        # Severity: a high max JS means the input pathway IS routing — the
        # *interesting* finding is when this is non-zero but downstream
        # action divergence is small. That's discovered by combining with a
        # CounterfactualDiagnostic; this diagnostic by itself only reports
        # input-pathway routing.
        if max_js >= 0.30:
            sev = Severity.INFO
            verdict = (
                f"Input pathway routes strongly on noun choice (max head JS = {max_js:.3f}). "
                f"If action divergence is small despite this, the downstream FFN is ignoring "
                f"the routing — that's the smoking gun for binding-without-following."
            )
        else:
            sev = Severity.MODERATE
            verdict = (
                f"Input pathway barely routes on noun choice (max head JS = {max_js:.3f}). "
                f"Attention is similar regardless of the noun — the model isn't even "
                f"distinguishing them at the attention level."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=max_js,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant={"max_js": max_js, "mean_js": mean_js},
            raw={
                "noun_a": self.noun_a,
                "noun_b": self.noun_b,
                "instr_a": instr_a,
                "instr_b": instr_b,
                "top_heads": top_heads,
                "js_per_head": js_per_head.tolist(),
                "attn_a_image_grid": attn_a.image_weights().mean(axis=(0, 1)).tolist(),
                "attn_b_image_grid": attn_b.image_weights().mean(axis=(0, 1)).tolist(),
            },
        )
