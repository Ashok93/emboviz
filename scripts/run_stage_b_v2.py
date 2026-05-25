"""Stage B v2 — Concept Decomposition of OpenVLA-7B.

What it does, in one sentence:
  Identifies *which named neurons in OpenVLA's backbone* are driving the
  model's chosen action at each frame, and flags neurons that fire
  unusually at failure moments.

This is fundamentally different from Stage B v1 (which produced heatmaps).
A heatmap tells you "the pixels of the spoon matter." Concept decomposition
tells you "the model engaged its 'grasp', 'lift', 'left' concepts to make
this choice — and at the failure frame, the 'release' concept fired 3×
above baseline. *That's* probably your bug."

Mechanism (from Häon et al. 2025, arXiv 2509.00328):
  • In each Llama FFN, the value vectors (columns of `down_proj.weight`) are
    residual-stream directions. Each direction has semantic meaning we can
    read off via logit lens: top vocabulary tokens that align with it.
  • A neuron's contribution to this frame's action = its activation at the
    action-prediction position × its value-vector magnitude.
  • Top-K contributing neurons = "concepts the model is using right now."

Usage:
    uv run python scripts/run_stage_b_v2.py --episode 0 --max-frames 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policylens.concepts import (
    build_concept_dictionary,
    extract_frame_concepts,
    find_anomalous_concepts,
)
from policylens.dataset_bridge import load_bridge_episode
from policylens.openvla import OpenVLAInference
from policylens.replay_vla import pick_keyframes, replay_vla
from policylens.visual_attribution import (
    compute_attention_rollout,
    compute_per_neuron_image_attribution,
)
from policylens.visualize_concepts import (
    render_anomaly_comparison,
    render_concept_timeline,
    render_cross_modal_binding,
    render_top_concepts_bars,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="PolicyLens Stage B v2 — concept decomposition")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--keyframes", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=12,
                        help="Number of episode frames to extract concepts from. Each frame is "
                             "a quick forward pass; 12 is enough to compute meaningful "
                             "baselines for anomaly detection.")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top neurons (per layer) to surface per frame")
    parser.add_argument("--dict-top-k", type=int, default=20,
                        help="Top vocabulary tokens per neuron in the dictionary")
    parser.add_argument("--z-threshold", type=float, default=1.5,
                        help="Z-score threshold for the smoking-gun anomaly detector")
    parser.add_argument("--vis-grid", type=int, default=16,
                        help="Image-attribution patch grid (NxN). 16 ≈ 75s per neuron.")
    parser.add_argument("--vis-top-neurons", type=int, default=5,
                        help="How many top smoking-gun neurons to compute image maps for.")
    parser.add_argument("--outdir", type=str, default="outputs/stage_b_v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Stage B v2 needs a GPU"

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    dict_cache = REPO_ROOT / "models_cache" / f"concept_dict_top{args.dict_top_k}.json"

    t0 = time.time()
    print(f"[stage-b-v2] device={device}  episode={args.episode}")

    print("[stage-b-v2] loading OpenVLA-7B...")
    vla = OpenVLAInference(device=device)

    print(f"[stage-b-v2] building/loading concept dictionary (cache: {dict_cache.name})...")
    dictionary = build_concept_dictionary(
        vla,
        cache_path=dict_cache,
        top_k=args.dict_top_k,
    )
    print(f"             {dictionary['n_layers']} layers × "
          f"{dictionary['intermediate_dim']} neurons = "
          f"{dictionary['n_layers'] * dictionary['intermediate_dim']:,} concepts indexed")

    # Show 5 random concepts as a sanity check the dictionary is meaningful.
    print("[stage-b-v2] sample concepts (top-5 tokens each):")
    import random
    rng = random.Random(0)
    for _ in range(5):
        li = rng.randrange(dictionary["n_layers"])
        ni = rng.randrange(dictionary["intermediate_dim"])
        tokens = dictionary["layers"][str(li)][str(ni)][:5]
        print(f"             L{li}.N{ni}: {tokens}")

    print("[stage-b-v2] loading bridge episode...")
    ep = load_bridge_episode(episode_idx=args.episode)
    print(f'             instruction: "{ep.instruction}"  ·  T={ep.num_frames}')

    print("[stage-b-v2] replaying OpenVLA over episode...")
    replay = replay_vla(vla, ep, max_frames=args.max_frames, warmup_frames=2)
    failure_idx = replay.failure_frame_idx
    print(f"             max-divergence frame: t={failure_idx}  "
          f"deviation={replay.action_deviations[failure_idx]:.3f}")

    print(f"[stage-b-v2] extracting concept activations for {len(replay.predictions)} frames...")
    per_frame = {}
    for ti, pred in enumerate(replay.predictions):
        per_frame[ti] = extract_frame_concepts(
            vla, pred, dictionary,
            top_k_per_frame=args.top_k,
        )

    keyframes = pick_keyframes(replay, args.keyframes)
    print(f"[stage-b-v2] keyframes for rendering: {keyframes}")

    print(f"[stage-b-v2] finding anomalous concepts (z >= {args.z_threshold}) at failure frame...")
    anomalies = find_anomalous_concepts(
        per_frame, failure_idx, dictionary,
        z_threshold=args.z_threshold,
        top_n=12,
    )
    print(f"             {len(anomalies)} smoking-gun concepts:")
    for a in anomalies[:5]:
        print(f"             • L{a.layer}.{a.neuron}  z={a.z_score:.2f}  "
              f"failure={a.failure_contribution:.3f}  baseline={a.baseline_mean:.3f}±{a.baseline_std:.3f}  "
              f"tokens={a.label_tokens[:5]}")

    # --- Cross-modal visual attribution for the top smoking-gun neurons ---
    failure_pred = replay.predictions[failure_idx]
    failure_image = ep.images[failure_idx]
    top_neurons = [(a.layer, a.neuron) for a in anomalies[:args.vis_top_neurons]]

    print(f"[stage-b-v2] computing attention rollout for failure frame t={failure_idx}...")
    att_rollout = compute_attention_rollout(vla, failure_pred)
    print(f"             rollout shape: {att_rollout.shape}")

    print(f"[stage-b-v2] computing per-neuron image attribution "
          f"({args.vis_grid}×{args.vis_grid} grid, top {len(top_neurons)} neurons)...")
    neuron_maps = compute_per_neuron_image_attribution(
        vla, failure_pred, top_neurons, dictionary, grid_side=args.vis_grid,
    )

    print("[stage-b-v2] rendering outputs...")
    render_cross_modal_binding(
        failure_image,
        anomalies[:args.vis_top_neurons],
        neuron_maps,
        att_rollout,
        outdir / "cross_modal_binding.png",
        title=f'Cross-modal binding · failure frame t={failure_idx} · "{ep.instruction}"',
    )
    render_top_concepts_bars(
        per_frame, keyframes, outdir / "top_concepts_per_keyframe.png",
        title=f'OpenVLA · "{ep.instruction}"',
    )
    render_concept_timeline(
        per_frame, outdir / "concept_timeline.png",
        failure_idx=failure_idx,
    )
    render_anomaly_comparison(
        anomalies, failure_idx, outdir / "smoking_gun_concepts.png",
        title=f'Smoking-gun concepts at failure frame t={failure_idx}  ·  "{ep.instruction}"',
    )

    _write_hypothesis_v2(
        outdir / "HYPOTHESIS_STAGE_B_V2.md",
        episode_idx=args.episode,
        instruction=ep.instruction,
        failure_idx=failure_idx,
        max_dev=float(replay.action_deviations[failure_idx]),
        anomalies=anomalies,
        top_concepts_at_failure=per_frame[failure_idx].top_hits[:10],
        keyframes=keyframes,
    )

    print(f"[stage-b-v2] done in {(time.time() - t0)/60:.1f} min  →  {outdir}")
    return 0


def _write_hypothesis_v2(
    path: Path, *, episode_idx, instruction, failure_idx, max_dev,
    anomalies, top_concepts_at_failure, keyframes,
) -> None:
    sg = "\n".join(
        f"  {i+1}. **L{a.layer}.N{a.neuron}** "
        f"(`{' · '.join(t for t in a.label_tokens[:6] if t.isalpha())}`)  "
        f"  z={a.z_score:.2f},  "
        f"  failure={a.failure_contribution:.3f} vs baseline {a.baseline_mean:.3f}±{a.baseline_std:.3f}"
        for i, a in enumerate(anomalies)
    ) or "  *(none above threshold)*"

    top_now = "\n".join(
        f"  {i+1}. **L{h.layer}.N{h.neuron}**  contribution={h.contribution:.3f}  "
        f"tokens=`{' · '.join(t for t in h.label_tokens[:6] if t.isalpha())}`"
        for i, h in enumerate(top_concepts_at_failure)
    )

    body = f"""# PolicyLens Stage B v2 — Concept Decomposition of OpenVLA-7B

**Episode**: {episode_idx}
**Instruction**: "{instruction}"
**Max-divergence frame**: t={failure_idx}  (||policy − expert|| = {max_dev:.3f})
**Keyframes rendered**: {keyframes}

## What this is

A different kind of interpretability than v1. Rather than highlighting
*which pixels mattered*, we ask: **which named neurons in OpenVLA's
backbone are doing the work, and which one is misbehaving at the failure
frame?**

The method (per Häon et al. 2025, arXiv 2509.00328): every FFN value vector
in Llama-7B has a semantic direction we can read off via logit lens. We
label every neuron in every layer with its top vocabulary tokens, hook the
FFN activations at the action-prediction position, then rank neurons by
their contribution to *this frame's* action choice.

## Smoking-gun analysis (the headline)

At the failure frame, these neurons fire **unusually** compared to the
rest of the episode (z-score over the baseline frames):

{sg}

Read these as the model's working vocabulary. If a neuron whose label
includes words like *release / drop / stop* fires high at a failure frame,
that's a directly readable bug.

## All top concepts at the failure frame

{top_now}

## Verdict
- [ ] Names line up with the failure modality (e.g. premature release,
  wrong-object grasp, drift) → genuine product value, escalate to a user
  test.
- [ ] Names look like noise (Unicode garbage, function-word soup) → the
  logit-lens labels aren't semantic enough; switch to SAE-based labels.

## Why this is the moat (vs. v1)

v1 showed "the pixels of the spoon matter" and "the word 'spoon' matters."
Both are *true and obvious*. Engineers can read them off the input.

v2 shows: *what the model is doing internally to produce this action*, in
named-concept terms. That's a vocabulary that doesn't exist in the input
and that the engineer can't deduce from data alone — it's a *property of
the trained network's internals* that we surface directly.

## Honest limits

1. **Logit-lens labels are noisy.** A neuron's top tokens may be
   approximate or polysemous. The literature recommends SAEs for cleaner
   features; we picked logit-lens because it requires no extra training
   data and works zero-shot on any OpenVLA checkpoint.
2. **Per-neuron analysis ignores neuron clusters.** A concept may be
   spread across multiple neurons; we surface the top individual ones.
3. **One episode, one instruction.** A robust "smoking-gun" claim
   requires sweeping several episodes and showing the same neuron flags.
"""
    path.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
