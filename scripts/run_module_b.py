"""Module B — Language-Grounding Diagnostic.

The product question: *"Did your VLA actually listen to the instruction?"*

Run OpenVLA on a Bridge episode with the baseline instruction AND a set of
counterfactual instructions, score (a) action divergence per variant, (b)
attention-head sensitivity to a noun swap. Render a single verdict card +
write a report.

Usage:
    uv run python scripts/run_module_b.py --episode 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emboviz.attention_grounding import (
    extract_attention_to_image,
    find_noun_token_positions,
    score_head_language_sensitivity,
)
from emboviz.counterfactual import classify_grounding, run_counterfactuals
from emboviz.dataset_bridge import load_bridge_episode
from emboviz.openvla import OpenVLAInference
from emboviz.visualize_verdict import VerdictPayload, render_verdict_card


# Counterfactual instruction variants for testing the spoon/fork-style failure.
# The strategy mirrors LIBERO-Plus / IGAR — pair a baseline with: an object swap,
# a direction reversal, a different verb, and an empty prompt.
DEFAULT_VARIANTS_TEMPLATE = [
    # Object swap: targets the noun. If model is language-blind it ignores this.
    "put small fork from basket to tray",
    # Direction reversal: instruction-following test.
    "put small spoon from tray to basket",
    # Different verb / task: completely different action.
    "lift the basket",
    # Empty prompt: pure vision.
    "",
    # Wildcard distractor task: vision-aligned but semantically unrelated.
    "press the red button",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Emboviz Module B — language-grounding diagnostic")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4,
                        help="Subsample factor across the episode for counterfactual rollouts.")
    parser.add_argument("--noun-a", type=str, default="spoon",
                        help="Noun in the baseline instruction whose grounding we probe.")
    parser.add_argument("--noun-b", type=str, default="fork",
                        help="Counterfactual noun to compare attention with.")
    parser.add_argument("--outdir", type=str, default="outputs/module_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Module B needs a GPU (OpenVLA-7B in bf16)"

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[module-b] device={device}  episode={args.episode}")

    print("[module-b] loading OpenVLA-7B...")
    vla = OpenVLAInference(device=device)

    print("[module-b] loading bridge episode...")
    ep = load_bridge_episode(episode_idx=args.episode)
    print(f'           instruction: "{ep.instruction}"  ·  T={ep.num_frames}')

    frame_indices = list(range(0, ep.num_frames, args.frame_stride))
    print(f"[module-b] sampling {len(frame_indices)} frames at stride {args.frame_stride}: {frame_indices}")

    print(f"[module-b] running baseline + {len(DEFAULT_VARIANTS_TEMPLATE)} counterfactuals...")
    cf = run_counterfactuals(vla, ep, DEFAULT_VARIANTS_TEMPLATE, frame_indices=frame_indices)
    print(f"           {cf.headline()}")
    for variant, iss in cf.instruction_sensitivity.items():
        marker = "GROUNDED" if iss >= 0.30 else ("partial" if iss >= 0.05 else "BLIND")
        print(f"           ISS={iss:.3f} [{marker:>8}]  ·  variant: '{variant}'")

    verdict_tag, verdict_text = classify_grounding(cf)
    print(f"[module-b] verdict: {verdict_tag.upper()} — {verdict_text}")

    # --- Attention grounding diagnostic (noun A vs noun B) ---
    print(f"[module-b] attention grounding diagnostic for nouns '{args.noun_a}' vs '{args.noun_b}'...")
    # Use the middle of the episode for visualization (stable framing).
    viz_idx = frame_indices[len(frame_indices) // 2]
    print(f"           viz frame: t={viz_idx}")

    # Prompt A: baseline (the noun is naturally in it, e.g., "spoon")
    pred_a = vla.predict(ep.images[viz_idx], ep.instruction)
    pos_a = find_noun_token_positions(vla, pred_a, args.noun_a)
    print(f"           '{args.noun_a}' token positions in baseline prompt: {pos_a}")
    attn_a = extract_attention_to_image(vla, pred_a, pos_a)
    if attn_a is not None:
        attn_a.noun = args.noun_a

    # Prompt B: counterfactual with noun B in place of noun A.
    cf_instruction_b = _swap_word(ep.instruction, args.noun_a, args.noun_b)
    pred_b = vla.predict(ep.images[viz_idx], cf_instruction_b)
    pos_b = find_noun_token_positions(vla, pred_b, args.noun_b)
    print(f"           '{args.noun_b}' token positions in cf prompt:       {pos_b}")
    attn_b = extract_attention_to_image(vla, pred_b, pos_b)
    if attn_b is not None:
        attn_b.noun = args.noun_b

    head_sens = []
    if attn_a is not None and attn_b is not None:
        head_sens = score_head_language_sensitivity(attn_a, attn_b)
        head_sens.sort(key=lambda h: -h.js_divergence)
        print(f"[module-b] top 5 language-sensitive heads (high JS = different attention for A vs B):")
        for h in head_sens[:5]:
            print(f"             L{h.layer:>2}.H{h.head:>2}  JS={h.js_divergence:.3f}  "
                  f"A focus@{h.primary_focus_a}  B focus@{h.primary_focus_b}")

    recommendation = _make_recommendation(verdict_tag, cf, head_sens)

    payload = VerdictPayload(
        base_image=ep.images[viz_idx],
        base_action=cf.actions[0, len(frame_indices) // 2],
        cf_result=cf,
        verdict_tag=verdict_tag,
        verdict_text=verdict_text,
        attn_noun_a=attn_a,
        attn_noun_b=attn_b,
        noun_a=args.noun_a,
        noun_b=args.noun_b,
        head_sensitivities=head_sens,
        recommendation=recommendation,
    )
    render_verdict_card(payload, outdir / "verdict_card.png")
    _write_report(outdir / "MODULE_B_REPORT.md", payload, viz_idx, ep, args)

    print(f"[module-b] done in {(time.time() - t0)/60:.1f} min  →  {outdir}")
    return 0


def _swap_word(text: str, a: str, b: str) -> str:
    """Case-sensitive whole-word swap (simple — Bridge instructions are lowercase)."""
    import re
    return re.sub(rf"\b{re.escape(a)}\b", b, text)


def _make_recommendation(verdict_tag, cf, head_sens) -> str:
    if verdict_tag == "language_blind":
        return (
            "This is the canonical 'vision-override-language' failure mode (LIBERO-Plus "
            "arXiv 2510.13626; IGAR arXiv 2603.06001; 'Robust Skills, Brittle Grounding' "
            "arXiv 2602.24143). The model is producing actions from visual priors alone.\n\n"
            "Fix recipe (graded by impact, cheapest first):\n"
            "  1. CONTRASTIVE OBJECT DEMOS — record 20-50 demos where ≥2 plausibly-graspable\n"
            "     objects co-occur and the instruction is the only disambiguator. Examples:\n"
            "     {spoon+fork}, {bowl+plate}, {cup+mug} with both target instructions for each pair.\n"
            "  2. REFUSAL/SEARCH DEMOS — 10-20 demos where the instruction names an object\n"
            "     that is NOT present; the correct action is to search/refuse, not grasp\n"
            "     something else. This directly trains the 'check before act' behavior.\n"
            "  3. INSTRUCTION REPHRASING — paraphrase each existing demo's instruction\n"
            "     2-3 times during fine-tuning to discourage memorization of exact wordings.\n\n"
            "Validation: re-run this diagnostic after fine-tuning. ISS should climb from\n"
            "<0.05 to >0.20 to indicate meaningful improvement."
        )
    if verdict_tag == "partial":
        return (
            "Partial grounding — the model responds to some but not all instruction "
            "variants. Focus on the LOW-ISS variants (the ones it ignores) — those are "
            "the binding gaps. Recommend recording contrastive demos along that specific "
            "axis (e.g., if 'spoon→fork' is ignored but 'lift→put' is followed, the gap "
            "is utensil-object disambiguation, not verb following)."
        )
    if verdict_tag == "grounded":
        return (
            "Grounded behavior — the model uses the instruction. If you are still seeing "
            "rollout failures, the issue is likely perception or motor execution rather "
            "than language grounding. Check failure detector signals + per-frame action "
            "deviation against expert demonstrations."
        )
    return "Unable to issue recommendation — verdict was unavailable."


def _write_report(path, payload, viz_idx, ep, args) -> None:
    cf = payload.cf_result
    rows = "\n".join(
        f"  • ISS={iss:.3f}  ·  '{variant}'"
        for variant, iss in cf.instruction_sensitivity.items()
    )
    head_rows = "\n".join(
        f"  • L{h.layer}.H{h.head}  JS={h.js_divergence:.3f}  "
        f"A focus@{h.primary_focus_a}  B focus@{h.primary_focus_b}"
        for h in payload.head_sensitivities[:10]
    ) if payload.head_sensitivities else "  (no head data — noun positions not found in prompts)"

    body = f"""# Emboviz Module B — Language-Grounding Diagnostic

**Episode**: {args.episode}
**Baseline instruction**: "{ep.instruction}"
**Frames sampled**: stride {args.frame_stride}, viz frame t={viz_idx}
**Noun probe**: A=`{args.noun_a}`  B=`{args.noun_b}`

## Verdict

**{payload.verdict_tag.upper()}**

{payload.verdict_text}

## Counterfactual Instruction Sensitivity (per variant)

{rows}

Interpretation: ISS < 0.05 = vision-blind on that variant. ISS > 0.30 = grounded.

## Top language-sensitive attention heads

{head_rows}

## Recommendation

{payload.recommendation}

## Method (one paragraph)

Per LIBERO-Plus (arXiv 2510.13626), IGAR (arXiv 2603.06001), and 'When Vision
Overrides Language' (arXiv 2602.17659), the cleanest test of VLA language
grounding is to hold the scene fixed and swap the instruction. Action
divergence is measured per-frame in 7-DOF Bridge action space. The
attention diagnostic identifies heads whose attention distribution (from
the noun token, to image-token positions) changes between two nouns — those
are the heads doing the noun-to-region routing. Heads with low JS
divergence treat both nouns identically (visual default). The verdict
combines both signals.
"""
    path.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
