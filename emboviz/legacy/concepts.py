"""Concept decomposition for OpenVLA — the moat layer of Stage B.

The core idea is from Häon et al. 2025 (arXiv 2509.00328), "Mechanistic
Interpretability for Steering Vision-Language-Action Models" — they show
that <25% of OpenVLA's FFN neurons get rewired for action prediction during
VLA fine-tuning, and the remainder retain semantically interpretable
directions inherited from VLM pre-training (concepts like "fast", "slow",
"up", "lift", "careful", "grasp").

We turn that finding into a *product*:

  1. **Concept dictionary** (one-time, cached). For each FFN neuron in the
     Llama backbone, project its value vector (column of `down_proj.weight`)
     onto the LLM's vocabulary embedding via logit-lens
     `score_t = w_i · E[t]`. Top-K tokens are that neuron's semantic label.

  2. **Per-frame concept extraction** (online). Hook the input to each
     layer's `down_proj` (the per-neuron activation `intermediate = gate*up`)
     at the action-token prediction position. Per neuron contribution to
     the residual stream is `|a_i| · ||w_i||`. Rank — top-K = "concepts the
     model is using to choose this action."

  3. **Anomaly detection** (the punchline). Across an episode, compute the
     normal activation profile per concept. At a failure frame, find concepts
     whose activations are unusually high (z-score) — those are the
     "smoking gun" driving the unusual behavior.

This is a vocabulary for VLA debugging that doesn't exist as a tool today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from emboviz.openvla import OpenVLAInference, VLAPrediction


# ---------------------------------------------------------------------------
# Concept dictionary (offline, cached)
# ---------------------------------------------------------------------------


def build_concept_dictionary(
    vla: OpenVLAInference,
    cache_path: Path | None = None,
    top_k: int = 20,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run logit lens on every FFN value vector.

    Returns a JSON-serializable dict:
        {
          "top_k": int,
          "layers": {
            layer_idx (str): {
              "neuron_idx (str)": [token, token, ...],   # length top_k
            },
            ...
          },
          "value_norms": {layer_idx (str): [float, ...]}, # len = intermediate
        }

    If `cache_path` exists, loads from disk instead of recomputing.

    Cost: 32 layers × 11008 neurons × (4096-dim dot product against 32k-vocab
    matrix) ≈ a few seconds on a 3090. We chunk by layer to avoid building
    a 32×11008×32000 matrix in memory.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    model = vla.model
    llm = model.language_model
    layers = llm.model.layers  # Llama decoder layers
    embed_weight = llm.get_input_embeddings().weight  # (vocab, hidden)
    tokenizer = vla.processor.tokenizer

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    out_layers: dict[str, dict[str, list[str]]] = {}
    out_norms: dict[str, list[float]] = {}

    for li in tqdm(layer_indices, desc="dict build", unit="layer"):
        down_proj = layers[li].mlp.down_proj.weight  # (hidden, intermediate)
        # Normalize the value vectors so the "top tokens" score reflects
        # direction, not magnitude. Keep magnitudes separately for the
        # contribution computation later.
        norms = down_proj.norm(dim=0).detach().float().cpu().numpy()  # (intermediate,)
        out_norms[str(li)] = norms.tolist()

        # Logit lens: each column w_i of down_proj is a residual-stream
        # direction. Its "vocabulary projection" is E · w_i, where E is the
        # token-embedding matrix. Top tokens by this score = w_i's meaning.
        with torch.no_grad():
            # (vocab, intermediate) — one column per neuron, scores against vocab
            scores = embed_weight.to(down_proj.dtype) @ down_proj  # (vocab, intermediate)
            topk_ids = scores.topk(top_k, dim=0).indices.cpu().numpy()  # (top_k, intermediate)

        # We also keep a *larger* top-K internally (5× what we display) so the
        # English-filter has enough material to work with — a neuron whose
        # top-3 are foreign tokens may still have clean English at top-10..20.
        wide_k = top_k * 5
        wide_top = scores.topk(wide_k, dim=0).indices.cpu().numpy()  # (wide_k, intermediate)
        del scores

        layer_dict: dict[str, list[str]] = {}
        for ni in range(wide_top.shape[1]):
            token_ids = wide_top[:, ni].tolist()
            tokens = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(token_ids)]
            # Promote clean English-ish tokens to the front while keeping order
            # within each category so the projection ranking is preserved.
            english_first = [t for t in tokens if _is_clean_english(t)]
            others = [t for t in tokens if not _is_clean_english(t)]
            layer_dict[str(ni)] = (english_first + others)[:top_k]
        out_layers[str(li)] = layer_dict

    result = {
        "top_k": top_k,
        "n_layers": len(layers),
        "intermediate_dim": down_proj.shape[1],
        "layers": out_layers,
        "value_norms": out_norms,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Per-frame concept extraction (online)
# ---------------------------------------------------------------------------


@dataclass
class ConceptHit:
    layer: int
    neuron: int
    contribution: float       # |activation| * ||value_vector||
    activation: float         # raw signed activation
    label_tokens: list[str]   # top vocabulary tokens for this neuron

    def short_label(self) -> str:
        # English first, then anything else.
        words = [t for t in self.label_tokens if _is_clean_english(t)][:3]
        if not words:
            words = [t for t in self.label_tokens if t and t != "<none>"][:2]
        return "/".join(words) if words else "?"


@dataclass
class FrameConcepts:
    """Per-frame snapshot of which FFN neurons fired during action prediction."""

    top_hits: list[ConceptHit]                # top-K across all layers
    per_layer_max: dict[int, float] = field(default_factory=dict)
    # Per-(layer, neuron) raw contributions — needed for anomaly detection.
    full: dict[tuple[int, int], float] = field(default_factory=dict)


def extract_frame_concepts(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    dictionary: dict,
    top_k_per_frame: int = 15,
    layer_indices: list[int] | None = None,
    min_layer_fraction: float = 0.5,
) -> FrameConcepts:
    """Hook every FFN's `down_proj` input to capture per-neuron activations
    at the action-token-prediction position, rank by contribution, return
    the top hits with their dictionary labels.

    `min_layer_fraction`: only consider layers in the back half of the network
    by default — that's where action-selection happens (per Häon et al.).
    """
    model = vla.model
    llm = model.language_model
    layers = llm.model.layers
    n_layers = len(layers)
    if layer_indices is None:
        start = int(n_layers * min_layer_fraction)
        layer_indices = list(range(start, n_layers))

    # Position to read activations from: the LAST token of the prompt — that
    # is the position that predicts the first action token (predict_action
    # appends a space + 7 action tokens after the prompt). We sample only
    # this one position; sampling all 7 action positions is also reasonable
    # but adds noise from late tokens whose action is largely set.
    read_position = pred.prompt_len - 1

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int) -> Callable:
        def hook(module, inputs, output):  # input to down_proj == intermediate
            # `inputs[0]` is (B, seq, intermediate)
            captured[layer_idx] = inputs[0].detach()
            return None
        return hook

    for li in layer_indices:
        h = layers[li].mlp.down_proj.register_forward_hook(make_hook(li))
        handles.append(h)

    try:
        with torch.inference_mode():
            vla.model(
                input_ids=pred.full_input_ids,
                pixel_values=pred.pixel_values,
            )
    finally:
        for h in handles:
            h.remove()

    # Rank all (layer, neuron) by contribution.
    full: dict[tuple[int, int], float] = {}
    per_layer_max: dict[int, float] = {}
    candidates: list[ConceptHit] = []

    value_norms = dictionary["value_norms"]
    layer_label_dict = dictionary["layers"]

    for li, act_tensor in captured.items():
        # act_tensor: (B=1, seq, intermediate)
        acts = act_tensor[0, read_position].float().cpu().numpy()  # (intermediate,)
        norms = np.array(value_norms[str(li)], dtype=np.float32)   # (intermediate,)
        contributions = np.abs(acts) * norms                       # (intermediate,)

        per_layer_max[li] = float(contributions.max())

        # Pick top-K per layer first so candidates stay small.
        top_neurons = np.argpartition(-contributions, kth=min(top_k_per_frame, contributions.size - 1))[:top_k_per_frame]
        for ni in top_neurons:
            ni = int(ni)
            c = float(contributions[ni])
            full[(li, ni)] = c
            candidates.append(ConceptHit(
                layer=li,
                neuron=ni,
                contribution=c,
                activation=float(acts[ni]),
                label_tokens=layer_label_dict[str(li)][str(ni)],
            ))

    candidates.sort(key=lambda h: -h.contribution)
    return FrameConcepts(
        top_hits=candidates[:top_k_per_frame],
        per_layer_max=per_layer_max,
        full=full,
    )


# ---------------------------------------------------------------------------
# Anomaly detection — "smoking gun" concepts at a failure frame
# ---------------------------------------------------------------------------


@dataclass
class ConceptAnomaly:
    layer: int
    neuron: int
    label_tokens: list[str]
    failure_contribution: float
    baseline_mean: float
    baseline_std: float
    z_score: float

    def short_label(self) -> str:
        words = [t for t in self.label_tokens if _is_clean_english(t)][:3]
        if not words:
            words = [t for t in self.label_tokens if t and t != "<none>"][:2]
        return "/".join(words) if words else "?"


def find_anomalous_concepts(
    per_frame_concepts: dict[int, FrameConcepts],
    failure_idx: int,
    dictionary: dict,
    z_threshold: float = 2.0,
    top_n: int = 10,
) -> list[ConceptAnomaly]:
    """Compare failure-frame contributions to the baseline (other frames).

    A concept is "anomalous" if its contribution at the failure frame is
    `z_threshold`+ standard deviations above the mean of the other frames.
    This is the smoking-gun finder: it tells you *which named neuron is
    behaving unusually right when the model misbehaves*.
    """
    baseline_indices = [i for i in per_frame_concepts if i != failure_idx]
    if len(baseline_indices) < 2:
        return []

    # Union of all (layer, neuron) keys observed at any frame.
    all_keys = set()
    for c in per_frame_concepts.values():
        all_keys.update(c.full.keys())

    layer_label_dict = dictionary["layers"]
    anomalies: list[ConceptAnomaly] = []

    for (li, ni) in all_keys:
        baseline = np.array([
            per_frame_concepts[i].full.get((li, ni), 0.0) for i in baseline_indices
        ], dtype=np.float32)
        failure_val = per_frame_concepts[failure_idx].full.get((li, ni), 0.0)

        mean, std = float(baseline.mean()), float(baseline.std())
        if std < 1e-6:
            continue
        z = (failure_val - mean) / std
        if z >= z_threshold:
            anomalies.append(ConceptAnomaly(
                layer=li,
                neuron=ni,
                label_tokens=layer_label_dict[str(li)][str(ni)],
                failure_contribution=failure_val,
                baseline_mean=mean,
                baseline_std=std,
                z_score=z,
            ))

    anomalies.sort(key=lambda a: -a.z_score)
    return anomalies[:top_n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_token(t) -> str:
    """Llama BPE: '▁' marks a word boundary. Tokenizer may return None for
    out-of-range ids — coerce to a safe placeholder."""
    if t is None:
        return "<none>"
    return t.replace("▁", " ").strip() or t


def _is_clean_english(t: str) -> bool:
    """True if `t` is a plausibly-meaningful English word fragment.

    Llama2's vocab is multilingual; for surfacing concept labels to a
    robotics engineer, foreign-script and short subword tokens are noise.
    Keep tokens that are pure ASCII letters of length >= 3.
    """
    if not t or len(t) < 3:
        return False
    # Allow internal hyphen for compounds like "long-horizon" if they appear.
    return all(c.isalpha() and c.isascii() for c in t)
