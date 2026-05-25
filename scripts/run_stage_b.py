"""End-to-end Stage B: OpenVLA-7B on a real BridgeV2 scene, with image +
token attribution AND a causal faithfulness check (the moat).

Usage:
    uv run python scripts/run_stage_b.py --episode 0 --keyframes 4 --max-frames 24
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

from emboviz.attribute_vla import attribute_image, attribute_tokens
from emboviz.dataset_bridge import load_bridge_episode
from emboviz.faithfulness import image_occlusion_curve
from emboviz.openvla import OpenVLAInference
from emboviz.replay_vla import pick_keyframes, replay_vla
from emboviz.visualize_vla import render_frame_grid, render_image_faithfulness


def main() -> int:
    parser = argparse.ArgumentParser(description="Emboviz Stage B")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--keyframes", type=int, default=4,
                        help="Number of keyframes to render & attribute (kept small — IG on 7B is expensive)")
    parser.add_argument("--max-frames", type=int, default=24,
                        help="Cap on frames replayed (a 50-frame Bridge episode is fine; we cap for speed)")
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="outputs/stage_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Stage B (OpenVLA-7B) needs a GPU"

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[stage-b] device={device}  episode={args.episode}")

    print("[stage-b] loading OpenVLA-7B (bf16, eager-attn for gradients)...")
    vla = OpenVLAInference(device=device)
    print(f"           vram: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("[stage-b] loading bridge episode...")
    ep = load_bridge_episode(episode_idx=args.episode)
    print(f"           T={ep.num_frames}  fps={ep.fps}")
    print(f'           instruction: "{ep.instruction}"')

    print("[stage-b] replaying OpenVLA over episode...")
    # Skip first 2 frames when picking max-divergence — open-loop first frames
    # have nothing to anchor on so deviation is often artificially large.
    replay = replay_vla(vla, ep, max_frames=args.max_frames, warmup_frames=2)
    print(
        f"           failure frame: t={replay.failure_frame_idx}  "
        f"deviation={replay.action_deviations[replay.failure_frame_idx]:.3f}"
    )

    keyframes = pick_keyframes(replay, args.keyframes)
    print(f"[stage-b] keyframes: {keyframes}")

    print("[stage-b] computing image + token attributions (this is the expensive step)...")
    image_attrs = {}
    token_attrs = {}
    for k in keyframes:
        pred = replay.predictions[k]
        image_attrs[k] = attribute_image(vla, pred, ig_steps=args.ig_steps)
        token_attrs[k] = attribute_tokens(vla, pred, ig_steps=args.ig_steps)
        ta = token_attrs[k]
        # Top-3 tokens by attribution magnitude — quick eyeball check.
        top3 = sorted(zip(ta.tokens[1:], ta.scores[1:]), key=lambda x: -abs(x[1]))[:3]
        print(f"           t={k} done  ·  top tokens: " +
              ", ".join(f"{t.replace(chr(9601), ' ').strip()!r}={s:.3f}" for t, s in top3))

    print("[stage-b] running faithfulness checks (the moat bit)...")
    image_faith = {}
    for k in keyframes:
        pred = replay.predictions[k]
        image_faith[k] = image_occlusion_curve(vla, pred, image_attrs[k])
        print(f"           t={k} occlusion: AUC ratio = {image_faith[k].auc_ratio:.2f}×")

    # Token attribution IS the causal ablation (see attribute_vla.py). The
    # bars in the frame grid are themselves the moat — they're causally
    # meaningful by construction. No separate faithfulness step needed.

    print("[stage-b] rendering outputs...")
    render_frame_grid(
        ep.images, keyframes, image_attrs, token_attrs,
        failure_idx=replay.failure_frame_idx,
        instruction=ep.instruction,
        out_path=outdir / "frame_grid.png",
    )
    render_image_faithfulness(image_faith, replay.failure_frame_idx, outdir / "faithfulness_image.png")

    # Top tokens per keyframe — quick text summary for the writeup.
    top_tokens_per_kf = {
        k: _top_tokens(token_attrs[k], n=3) for k in keyframes
    }
    _write_hypothesis_b(
        outdir / "HYPOTHESIS_STAGE_B.md",
        episode_idx=args.episode,
        instruction=ep.instruction,
        failure_idx=replay.failure_frame_idx,
        max_dev=float(replay.action_deviations[replay.failure_frame_idx]),
        keyframes=keyframes,
        image_auc_ratios={k: float(image_faith[k].auc_ratio) for k in keyframes},
        top_tokens_per_kf=top_tokens_per_kf,
    )

    print(f"[stage-b] done in {(time.time() - t0)/60:.1f} min")
    print(f"           → {outdir}")
    return 0


def _write_hypothesis_b(
    path: Path, *, episode_idx, instruction, failure_idx, max_dev, keyframes,
    image_auc_ratios, top_tokens_per_kf,
) -> None:
    mean_ratio = sum(image_auc_ratios.values()) / max(1, len(image_auc_ratios))
    top_tok_md = chr(10).join(
        f"  • t={k}: " + ", ".join(f"**{t}**={v:.3f}" for t, v in toks)
        for k, toks in top_tokens_per_kf.items()
    )
    body = f"""# Emboviz Stage B — OpenVLA-7B on BridgeV2

**Episode**: {episode_idx}
**Instruction**: "{instruction}"
**Max-divergence frame**: t={failure_idx}  (||policy − expert|| = {max_dev:.3f})
**Keyframes attributed**: {keyframes}

Tests three claims, two of them *causal* (measured, not just descriptive).

## Claim 1 — Image attribution highlights task-relevant regions
See `frame_grid.png`, column "IG over image". Gradient-based heatmap;
descriptive, not causal. Hot zones should sit on the objects named in the
instruction, the gripper, and the target.

- [ ] PASS / FAIL (eyeball)

## Claim 2 — Token attribution by causal ablation surfaces content words

We dropped embedding-IG (gradients underflowed through OpenVLA-7B in bf16)
and switched to **direct ablation**: for each token we silence it (replace
with BOS) and *measure* how much the predicted action changes. The bar
heights in `frame_grid.png` are ||Δaction||₂ — causal scores, not gradients.

Top tokens per keyframe (excluding BOS):
{top_tok_md}

Expected: instruction nouns and verbs dominate; "the"/"to"/etc near zero.

- [ ] PASS / FAIL (eyeball)

## Claim 3 — Image attribution is CAUSALLY FAITHFUL (the moat)

Tests whether the gradient-IG heatmap actually predicts where the model is
sensitive. Method: sort 32×32 patches by IG score, mask top-k%, measure
||Δaction||₂. Compare to masking *random* patches. AUC ratio of the two
curves is our headline metric.

Per-keyframe AUC ratio (IG / random):
{chr(10).join(f"  • t={k}: {r:.2f}×" for k, r in image_auc_ratios.items())}
  • **mean: {mean_ratio:.2f}×**

Interpretation:
  • **≥ 2.0×** — strong causal coupling; clean moat story.
  • **1.3–2.0×** — heatmap carries real signal; report honestly.
  • **≤ 1.1×** — heatmap is cosmetic.

See `faithfulness_image.png`.

## Verdict
- [ ] PASS — all three claims hold → real, defensible Stage B demo.
- [ ] MIXED — partial; refine before showing users.
- [ ] FAIL — step back; the story needs rethinking.

## Honest framing for any demo
1. **Attribution target** = single-step log-probability of the chosen action
   tokens, not the multi-step trajectory.
2. **Token attribution by single-token ablation** — joint effects of pairs
   of tokens aren't captured.
3. **Image faithfulness** is measured at 32×32 patch granularity because
   single-pixel perturbations don't cross OpenVLA's 256-bin action
   discretization.
4. **One episode, one instruction.** Don't generalize without ≥5 more.
"""
    path.write_text(body)


def _top_tokens(ta, n: int = 3):
    """Top-N most causally important tokens, excluding BOS."""
    ranked = sorted(
        [(t, s) for t, s in zip(ta.tokens[1:], ta.scores[1:])],
        key=lambda x: -abs(x[1]),
    )[:n]
    return [(t.replace("▁", " ").strip(), float(s)) for t, s in ranked]


if __name__ == "__main__":
    raise SystemExit(main())
