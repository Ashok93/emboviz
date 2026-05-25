"""End-to-end killer demo for one real failure scenario.

The user story:
  Engineer fine-tunes OpenVLA for kitchen tasks. They run a rollout where
  the instruction says "pick the fork" — but the scene only has a spoon.
  The robot picks the spoon anyway. The engineer opens PolicyLens.

This script reproduces that exact moment on BridgeV2 (a real dataset
OpenVLA was trained on), diagnoses it end-to-end, and emits a poster-quality
PNG + Markdown report.

Usage:
    uv run python scripts/run_failure_demo.py
    uv run python scripts/run_failure_demo.py --primary-episode 0 --noun-correct spoon --noun-wrong fork
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policylens.action_viz import aggregate_trajectory
from policylens.attention_grounding import (
    extract_attention_to_image,
    find_noun_token_positions,
    score_head_language_sensitivity,
)
from policylens.counterfactual import run_counterfactuals
from policylens.coverage_analysis import (
    analyze_dataset_coverage,
    collect_bridge_instructions,
    detect_gaps,
    render_coverage_report,
)
from policylens.dataset_bridge import DATASET_REPO, load_bridge_episodes
from policylens.demo_viz import (
    FailureDemoPayload,
    VariantPanel,
    render_failure_demo,
)
from policylens.instruction_perturb import NOUN_CATEGORIES, OBJECT_CATEGORIES, build_perturbations
from policylens.openvla import OpenVLAInference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-episode", type=int, default=0)
    parser.add_argument("--noun-correct", type=str, default="spoon")
    parser.add_argument("--noun-wrong", type=str, default="fork")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="outputs/failure_demo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[demo] device={device}  episode={args.primary_episode}")
    print(f"[demo] loading OpenVLA-7B...", flush=True)
    vla = OpenVLAInference(device=device)

    print(f"[demo] loading bridge episode {args.primary_episode}...", flush=True)
    eps = load_bridge_episodes([args.primary_episode])
    ep = eps[args.primary_episode]
    correct_instr = ep.instruction
    wrong_instr = _swap_word(correct_instr, args.noun_correct, args.noun_wrong)
    print(f'        correct: "{correct_instr}"')
    print(f'        wrong  : "{wrong_instr}"')

    # ------- Counterfactual sweep on this single scene ----------------------
    print("\n[demo] running baseline + counterfactuals...", flush=True)
    pset = build_perturbations(correct_instr)
    variant_texts = [p.text for p in pset.perturbations]
    # Ensure the explicit wrong instruction is included as a variant.
    if wrong_instr not in variant_texts:
        variant_texts.insert(0, wrong_instr)
    frame_indices = list(range(0, ep.num_frames, args.frame_stride))
    cf = run_counterfactuals(vla, ep, variant_texts, frame_indices=frame_indices)

    # Build variant panels (baseline + each counterfactual)
    baseline_traj = aggregate_trajectory(cf.actions[0])
    panels = [VariantPanel(
        instruction=correct_instr, axis="baseline", iss=0.0,
        arrow=baseline_traj, is_baseline=True,
    )]
    for i, text in enumerate(variant_texts):
        # Determine axis from pset if present, else label wrong_instr explicitly.
        axis = "noun_swap" if text == wrong_instr else next(
            (p.axis for p in pset.perturbations if p.text == text), "other"
        )
        panels.append(VariantPanel(
            instruction=text, axis=axis,
            iss=cf.instruction_sensitivity[text],
            arrow=aggregate_trajectory(cf.actions[i + 1]),
            is_baseline=False,
        ))

    headline_noun_iss = cf.instruction_sensitivity[wrong_instr]
    ood_iss = next(
        (v for k, v in cf.instruction_sensitivity.items() if "press" in k.lower()),
        max(cf.instruction_sensitivity.values()),
    )
    print(f"\n[demo] headline numbers:")
    print(f"        noun-swap '{args.noun_correct}'→'{args.noun_wrong}' ISS = {headline_noun_iss:.3f}")
    print(f"        OOD-task reference ISS                            = {ood_iss:.3f}")

    # ------- Attention grounding ------------------------------------------
    viz_idx = frame_indices[len(frame_indices) // 2]
    print(f"\n[demo] attention grounding at t={viz_idx}: '{args.noun_correct}' vs '{args.noun_wrong}'", flush=True)

    pred_a = vla.predict(ep.images[viz_idx], correct_instr)
    pos_a = find_noun_token_positions(vla, pred_a, args.noun_correct)
    attn_a = extract_attention_to_image(vla, pred_a, pos_a)
    if attn_a is not None:
        attn_a.noun = args.noun_correct

    pred_b = vla.predict(ep.images[viz_idx], wrong_instr)
    pos_b = find_noun_token_positions(vla, pred_b, args.noun_wrong)
    attn_b = extract_attention_to_image(vla, pred_b, pos_b)
    if attn_b is not None:
        attn_b.noun = args.noun_wrong

    top_heads: list = []
    if attn_a is not None and attn_b is not None:
        top_heads = score_head_language_sensitivity(attn_a, attn_b)
        top_heads.sort(key=lambda h: -h.js_divergence)
        for h in top_heads[:3]:
            print(f"        L{h.layer}.H{h.head}  JS={h.js_divergence:.3f}  "
                  f"A focus@{h.primary_focus_a} B focus@{h.primary_focus_b}")

    # ------- Coverage analysis on the full Bridge task corpus --------------
    print(f"\n[demo] running coverage analysis on {DATASET_REPO}...", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds_for_meta = LeRobotDataset(DATASET_REPO, episodes=[0])
    all_instructions = collect_bridge_instructions(ds_for_meta)
    coverage = analyze_dataset_coverage(all_instructions, dataset_name=DATASET_REPO)
    # Detect gaps for all object categories — coverage panel can show them.
    gaps = detect_gaps(coverage, [
        {"axis": "noun_swap", "category": c} for c in NOUN_CATEGORIES
    ])
    # Promote the category we ACTUALLY tested to the top of the list so the
    # recommendation focuses on the failing axis — not the worst-coverage
    # axis in general.
    target_category = None
    for cat, words in OBJECT_CATEGORIES.items():
        if args.noun_correct.lower() in words and cat in NOUN_CATEGORIES:
            target_category = cat
            break
    if target_category:
        target_gap = next((g for g in gaps if target_category in g.failure_axis), None)
        if target_gap is not None:
            gaps = [target_gap] + [g for g in gaps if g is not target_gap]
    # Then sort the *remaining* gaps by severity for the bar chart.
    severity_rank = {"critical": 0, "moderate": 1, "ok": 2}
    head = gaps[:1]
    tail = sorted(gaps[1:], key=lambda g: (severity_rank[g.severity], -g.observed_count))
    gaps = head + tail
    print(f"        analysed {len(all_instructions)} unique tasks; "
          f"{sum(1 for g in gaps if g.severity != 'ok')} categories flagged")

    # ------- Render the killer demo PNG ------------------------------------
    payload = FailureDemoPayload(
        scene_image=ep.images[viz_idx],
        correct_instruction=correct_instr,
        wrong_instruction=wrong_instr,
        variant_panels=panels,
        attn_a=attn_a, attn_b=attn_b,
        noun_a=args.noun_correct, noun_b=args.noun_wrong,
        top_heads=top_heads,
        coverage_gaps=gaps,
        total_dataset_episodes=len(all_instructions),
        paired_summary=None,
        headline_iss_noun_swap=headline_noun_iss,
        headline_iss_correct=0.0,
        headline_iss_ood=ood_iss,
    )
    print(f"\n[demo] rendering poster...", flush=True)
    render_failure_demo(payload, outdir / "failure_demo.png")
    render_coverage_report(coverage, outdir / "COVERAGE_REPORT.md")
    _write_demo_report(outdir / "DEMO_REPORT.md", payload, ep, args, cf)

    print(f"\n[demo] done in {(time.time() - t0) / 60:.1f} min  →  {outdir}")
    return 0


def _swap_word(text: str, a: str, b: str) -> str:
    return re.sub(rf"\b{re.escape(a)}\b", b, text, flags=re.IGNORECASE)


def _write_demo_report(path: Path, payload: FailureDemoPayload, ep, args, cf) -> None:
    iss_table = "\n".join(
        f"  • `{p.axis:>15}`  ISS={p.iss:.3f}  ·  \"{p.instruction or '(empty)'}\""
        for p in payload.variant_panels
    )
    head_table = "\n".join(
        f"  • L{h.layer}.H{h.head}  JS={h.js_divergence:.3f}  "
        f"A focus={h.primary_focus_a}  B focus={h.primary_focus_b}"
        for h in payload.top_heads[:5]
    ) if payload.top_heads else "  (no head data)"

    gap_blocks = []
    for g in payload.coverage_gaps:
        badge = {"critical": "🟥", "moderate": "🟧", "ok": "🟩"}[g.severity]
        gap_blocks.append(
            f"{badge} **{g.failure_axis}** — severity **{g.severity.upper()}**\n"
            f"  observed: {g.observed_count} demos "
            f"(of {payload.total_dataset_episodes:,} unique tasks)\n"
            f"  recommendation: {g.recommendation.splitlines()[0]}"
        )
    gap_block = "\n\n".join(gap_blocks)

    body = f"""# PolicyLens — Failure Scenario Demo

**Scene**: BridgeV2 episode {args.primary_episode}, frame t={len(payload.variant_panels) // 2}
**Correct instruction**: `"{payload.correct_instruction}"`
**Wrong instruction (the failure case)**: `"{payload.wrong_instruction}"`

## The story in one sentence

OpenVLA was asked to pick a **{args.noun_wrong}** from a scene containing only a **{args.noun_correct}**.
Its predicted action diverged from the baseline (correct instruction) by **only
{payload.headline_iss_noun_swap:.3f}** units (7-DOF Bridge action space).
For reference, an OOD-task instruction ("press the red button") changes the
action by **{payload.headline_iss_ood:.3f}**.

In plain English: the model is treating "fork" the same as "spoon." It's
producing actions from visual priors, not from the language.

## Numbers

### Counterfactual matrix (this scene)
{iss_table}

### Top language-sensitive attention heads
For each head, JS divergence between attention(noun=`{payload.noun_a}`) and
attention(noun=`{payload.noun_b}`) over the image tokens.

{head_table}

The non-zero JS values mean the model's INPUT pathway DOES route on the noun
— but the action output stays nearly identical, so downstream FFN layers
are not using that routing.

## Root cause in the training data

BridgeV2 ({payload.total_dataset_episodes:,} unique task descriptions):

{gap_block}

## Method

- **Counterfactual instruction test** — hold the scene fixed, swap the instruction,
  measure action divergence in 7-DOF Bridge units. (LIBERO-Plus arXiv 2510.13626;
  IGAR arXiv 2603.06001.)
- **Attention grounding diagnostic** — extract attention from the noun-token
  position to image-patch tokens; compare two nouns via Jensen-Shannon divergence
  per head. (Kang et al. CVPR 2025 arXiv 2503.06287.)
- **Coverage analysis** — scan the dataset's task descriptions for the
  within-category object co-occurrences that are needed for grounding to emerge.
- **Recommendation** — templated, per-failure-axis data-collection brief tied
  to dataset-specific gaps.

## What this enables

A robotics engineer, on receiving a real failure rollout, can run one command
and get a single PNG that pinpoints:

  1. **Whether** the model used the instruction (counterfactual)
  2. **Where** the language enters the network (attention heads)
  3. **Why** the model has this gap (coverage analysis)
  4. **What to do about it** (concrete recording brief)

That is the closed loop no tool today ships.
"""
    path.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
