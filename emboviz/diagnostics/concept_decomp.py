"""Concept decomposition via FFN logit-lens.

Based on Häon et al. 2025 (arXiv 2509.00328): in a fine-tuned VLA, <25% of
FFN neurons get rewired for action prediction. The remainder retain
semantically interpretable directions inherited from VLM pretraining.

The diagnostic:

  1. At each FFN layer in the late-layer band, capture per-neuron
     activations at the action-prediction position.
  2. Rank neurons by *contribution* = |activation| × ||value_vector||.
  3. (Optional) Project the top neurons' value vectors onto the
     vocabulary embedding to read their 'logit-lens label.'

For a Trajectory, anomaly detection over time finds *which neurons fire
unusually at which frames* — the 'smoking gun' for failure moments.

Single-scene diagnostic produces a `DiagnosticResult` with `raw` payload:
  - `top_neurons`: list of {layer, neuron, contribution, label_tokens?}

A trajectory wrapper (TrajectoryDiagnostic over this) gives you the
cross-frame anomaly story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, TokenSelector
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, NotSupported, VLAModel


@dataclass
class _NeuronHit:
    layer: int
    neuron: int
    activation: float                # raw signed
    contribution: float              # |activation| × value-vector norm
    label_tokens: list[str]          # may be empty if VOCAB_LOGIT_LENS unsupported


class ConceptDecompositionDiagnostic(Diagnostic):
    """Top-K FFN neurons driving this frame's action.

    Capability-gated. Requires FFN_ACTIVATIONS + FFN_VALUE_VECTORS.
    VOCAB_LOGIT_LENS is optional — without it, top neurons are unlabeled.
    """

    required_capabilities = Capability.INFERENCE | Capability.FFN_ACTIVATIONS | Capability.FFN_VALUE_VECTORS

    def __init__(
        self,
        top_k: int = 12,
        layer_fraction: tuple[float, float] = (0.5, 1.0),  # late-layer band
        label_top: int = 8,                                 # vocab labels for top-N
        label_tokens_per_neuron: int = 8,
    ):
        self.top_k = top_k
        self.layer_fraction = layer_fraction
        self.label_top = label_top
        self.label_tokens_per_neuron = label_tokens_per_neuron
        self.name = "concept_decomposition"
        self.axis = "internal.concept_decomp"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(
                model, scene,
                "model lacks FFN_ACTIVATIONS or FFN_VALUE_VECTORS capability",
            )

        n_layers = model.num_layers or 32
        lo = int(n_layers * self.layer_fraction[0])
        hi = int(n_layers * self.layer_fraction[1])
        layer_indices = list(range(lo, hi))

        try:
            ffn = model.extract_ffn_activations(
                scene.image, scene.instruction, layer_indices,
                TokenSelector(relative="before_action"),
            )
            norms = model.get_ffn_value_vector_norms(layer_indices)
        except NotSupported as e:
            return self._not_applicable(model, scene, str(e))

        can_label = Capability.VOCAB_LOGIT_LENS in model.capabilities

        # Compute per-(layer, neuron) contribution; collect top-K across all layers.
        candidates: list[_NeuronHit] = []
        for li in layer_indices:
            acts = ffn.by_layer[li]                          # (intermediate,)
            ns = norms[li]                                   # (intermediate,)
            contrib = np.abs(acts) * ns                      # (intermediate,)
            # Top per-layer first to keep memory bounded
            k = min(self.top_k, contrib.size)
            top = np.argpartition(-contrib, k - 1)[:k]
            for ni in top:
                ni = int(ni)
                candidates.append(_NeuronHit(
                    layer=li,
                    neuron=ni,
                    activation=float(acts[ni]),
                    contribution=float(contrib[ni]),
                    label_tokens=[],
                ))

        # Global top-K
        candidates.sort(key=lambda h: -h.contribution)
        top_hits = candidates[: self.top_k]

        # Optional: label the very top
        if can_label and self.label_top > 0:
            for h in top_hits[: self.label_top]:
                try:
                    vec = model.get_ffn_value_vectors(h.layer)[h.neuron]
                    tokens = model.project_to_vocab(vec, top_k=self.label_tokens_per_neuron)
                    h.label_tokens = [
                        t for t, _ in tokens
                        if t and len(t) >= 3 and t.isascii() and t.isalpha()
                    ][: self.label_tokens_per_neuron] or [t for t, _ in tokens[:3]]
                except Exception:
                    pass

        # Headline score: total contribution captured by top-K — useful as
        # a per-frame "intensity" measure for trajectory analysis.
        total = float(sum(h.contribution for h in top_hits))

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=total,
            severity=Severity.INFO,                  # descriptive, not failure-detecting
            direction="higher_is_worse",             # higher = more concentrated activity
            explanation=(
                f"Top {len(top_hits)} FFN neurons contribute {total:.2f} total to "
                f"the action-prediction residual at this frame "
                f"(labels via vocab logit lens: {'on' if can_label else 'off'})."
            ),
            per_variant={
                f"L{h.layer}.N{h.neuron}": h.contribution for h in top_hits
            },
            raw={
                "top_hits": [
                    {
                        "layer": h.layer,
                        "neuron": h.neuron,
                        "activation": h.activation,
                        "contribution": h.contribution,
                        "label_tokens": h.label_tokens,
                    }
                    for h in top_hits
                ],
                "layer_range": [lo, hi],
            },
        )


def find_anomalous_neurons(
    trajectory_result,           # TrajectoryDiagnosticResult — typed loosely to avoid cycle
    z_threshold: float = 2.0,
    top_n: int = 10,
) -> list[dict]:
    """Find neurons whose contribution at one frame is z>threshold above its baseline.

    Aggregates `raw['top_hits']` across all frames in a TrajectoryDiagnosticResult
    from a ConceptDecompositionDiagnostic. Returns a list of dicts with:
      {layer, neuron, label_tokens, anomalous_frame_idx, frame_contribution,
       baseline_mean, baseline_std, z_score}
    """
    # Build per-(layer, neuron) time-series of contributions.
    timeseries: dict[tuple[int, int], list[float]] = {}
    labels_by_key: dict[tuple[int, int], list[str]] = {}
    for r in trajectory_result.per_frame:
        seen = set()
        for h in r.raw.get("top_hits", []):
            key = (int(h["layer"]), int(h["neuron"]))
            timeseries.setdefault(key, [0.0] * len(trajectory_result.per_frame))
            seen.add(key)
            if not labels_by_key.get(key):
                labels_by_key[key] = h.get("label_tokens", [])
    # Second pass fills the values at the correct frame index.
    for i, r in enumerate(trajectory_result.per_frame):
        for h in r.raw.get("top_hits", []):
            timeseries[(int(h["layer"]), int(h["neuron"]))][i] = float(h["contribution"])

    anomalies: list[dict] = []
    for key, vals in timeseries.items():
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size < 3:
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            continue
        for i, v in enumerate(arr):
            z = (v - mean) / std
            if z >= z_threshold:
                anomalies.append({
                    "layer": key[0],
                    "neuron": key[1],
                    "label_tokens": labels_by_key.get(key, []),
                    "anomalous_frame_idx": trajectory_result.frame_indices[i],
                    "frame_contribution": float(v),
                    "baseline_mean": mean,
                    "baseline_std": std,
                    "z_score": float(z),
                })

    anomalies.sort(key=lambda a: -a["z_score"])
    return anomalies[:top_n]
