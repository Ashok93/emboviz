"""Train a failure-prediction probe on N Bridge episodes.

Pipeline:
  1. For each episode, run the model on every frame, extract hidden states
     at the action-prediction position, and compute the predicted action.
  2. Compare predictions to expert actions in the dataset metadata.
  3. Label-spread around the largest-deviation frame.
  4. Train logistic regression on (hidden_states → failure label).
  5. Save the trained probe to disk.

Usage:
    uv run python -m policylens.cli.train_failure_probe \
        --model openvla-7b \
        --episodes 0 1 2 3 4 5 \
        --layers 14 22 30 \
        --outdir probes_trained/openvla_failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from policylens.cli._loaders import load_model
from policylens.core.types import TokenSelector
from policylens.datasets.lerobot_bridge import BridgeEpisodeSource
from policylens.probes.base import ProbeSpec
from policylens.probes.presets.failure_predictor import (
    FAILURE_PROBE_NAME,
    label_frames_from_deviation,
)
from policylens.probes.store import save_probe
from policylens.probes.trainer import train_linear_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[14, 22, 30],
                        help="Which decoder layers to source hidden states from")
    parser.add_argument("--failure-threshold", type=float, default=0.30,
                        help="Per-episode max-deviation cutoff (Bridge action units)")
    parser.add_argument("--spread", type=int, default=3,
                        help="Mark this many frames around the spike as 'failure'")
    parser.add_argument("--outdir", default="probes_trained/failure_predictor")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[train] model: {args.model}", flush=True)
    model = load_model(args.model)
    print(f"[train] layers: {args.layers}")
    print(f"[train] episodes: {args.episodes}")

    src = BridgeEpisodeSource()
    per_ep_predicted: list[np.ndarray] = []
    per_ep_expert: list[np.ndarray] = []
    per_ep_hidden: list[np.ndarray] = []

    for ep_idx in args.episodes:
        traj = src.load_trajectory(ep_idx)
        print(f"\n[train] ep {ep_idx}: T={len(traj)}  instr=\"{traj.frames[0].instruction}\"", flush=True)
        pred_actions = []
        expert_actions = []
        hidden_per_frame = []
        for scene in tqdm(traj.frames, desc=f"ep{ep_idx}", unit="frame", leave=False):
            ar = model.predict(scene.image, scene.instruction)
            hs = model.extract_hidden_states(
                scene.image, scene.instruction, args.layers,
                TokenSelector(relative="before_action"),
            )
            pred_actions.append(ar.action)
            expert_actions.append(np.asarray(scene.metadata.get("expert_action", np.zeros(7)),
                                              dtype=np.float32))
            hidden_per_frame.append(hs.states)
        per_ep_predicted.append(np.stack(pred_actions))
        per_ep_expert.append(np.stack(expert_actions))
        per_ep_hidden.append(np.stack(hidden_per_frame))

    # Labelling
    labels, _idx = label_frames_from_deviation(
        per_ep_predicted, per_ep_expert,
        failure_threshold=args.failure_threshold,
        spread_frames=args.spread,
    )
    failure_count = sum(labels)
    success_count = len(labels) - failure_count
    print(f"\n[train] labeling: {failure_count} failure / {success_count} success "
          f"frames across {len(args.episodes)} episodes")
    if failure_count < 3 or success_count < 3:
        print("[train] ERROR: not enough samples in both classes; lower --failure-threshold "
              "or include more episodes.")
        return 1

    # Flatten hidden states across episodes
    all_hidden = np.concatenate(per_ep_hidden, axis=0)
    print(f"[train] flattened hidden states: {all_hidden.shape}")

    label_strings = ["success" if y == 0 else "failure" for y in labels]
    spec = ProbeSpec(
        name=FAILURE_PROBE_NAME,
        target_description="P(this frame is in a failing rollout episode)",
        model_id=model.model_id,
        layer_indices=list(args.layers),
        classes=["success", "failure"],
        metadata={
            "episodes_trained_on": list(args.episodes),
            "failure_threshold": args.failure_threshold,
            "spread_frames": args.spread,
        },
    )
    probe = train_linear_probe(all_hidden, label_strings, spec)
    print(f"[train] probe accuracy: train={probe.spec.train_accuracy:.3f}  "
          f"val={probe.spec.val_accuracy:.3f}")

    out_path = outdir / f"{FAILURE_PROBE_NAME}_{model.model_id}"
    save_probe(probe, out_path)
    print(f"[train] saved → {out_path}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
