"""Per-model probe for the new ModalityDropoutDiagnostic.

Verifies on a real model + dataset that:
  • The marginal pool builds from OTHER episodes successfully.
  • K-sample marginal sampling produces meaningful intervention magnitudes.
  • Per-modality verdicts honor the intervention-validity gate.

Run inside the model's venv:
    /root/venvs/openvla/bin/python scripts/probe_modality_dropout.py \\
        --model openvla --dataset bridge --episode 0 --frame 0 \\
        --pool-size 10 --k-samples 5 --out /root/probes/dropout_openvla
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emboviz.calibration import calibrate_model
from emboviz.diagnostics.modality_dropout import ModalityDropoutDiagnostic
from emboviz.modality_pools import build_modality_pool


def load_model(name: str):
    if name == "openvla":
        from emboviz.models.openvla import OpenVLAAdapter
        return OpenVLAAdapter()
    if name == "oft":
        from emboviz.models.openvla_oft import OpenVLAOFTAdapter
        return OpenVLAOFTAdapter()
    if name == "pi0":
        from emboviz.models.pi0 import Pi0Adapter
        return Pi0Adapter(config_name="pi0_libero", use_pytorch=True)
    if name == "gr00t":
        from emboviz.models.gr00t import Gr00tAdapter
        return Gr00tAdapter(camera_mapping={
            "primary":    "exterior_image_1_left",
            "wrist_left": "wrist_image_left",
        })
    raise SystemExit(f"unknown model {name!r}")


def load_dataset(name: str):
    if name == "bridge":
        from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
        return BridgeEpisodeSource()
    if name == "libero-spatial":
        from emboviz.datasets.lerobot_libero import LiberoSpatialSource
        return LiberoSpatialSource()
    if name == "pi-libero":
        from emboviz.datasets.lerobot_libero import PhysicalIntelligenceLiberoSource
        return PhysicalIntelligenceLiberoSource()
    if name == "droid-sample":
        from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
        return GR00TDroidSampleSource()
    raise SystemExit(f"unknown dataset {name!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["openvla", "oft", "pi0", "gr00t"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=10)
    p.add_argument("--k-samples", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--pool-cache-dir", default=None,
        help="Directory for ModalityPool disk cache. Shared between the "
             "probe and the runner so the runner skips the rebuild "
             "(otherwise pool reconstruction triggers a second HF API "
             "round and can hit the rate limit).",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading dataset: {args.dataset}", flush=True)
    dataset = load_dataset(args.dataset)
    traj = dataset.load_trajectory(args.episode)
    scene = traj.frames[args.frame]
    print(f"  instruction: {scene.instruction!r}")
    print(f"  cameras: {sorted(scene.observations.images)}")

    print(f"[2/4] loading model: {args.model}", flush=True)
    model = load_model(args.model)
    print(f"  caps: {model.capabilities}")
    print(f"  required_inputs.cameras: {sorted(model.required_inputs.cameras)}")
    print(f"  required_inputs.state={model.required_inputs.state}  "
          f"gripper={model.required_inputs.gripper}  "
          f"action_history={model.required_inputs.action_history}  "
          f"instruction={model.required_inputs.instruction}")

    print(f"[3/4] calibrating + building pool", flush=True)
    calibration = calibrate_model(model, traj, n_noise_probes=3)
    print(f"  noise_floor={calibration.noise_floor:.4f} "
          f"typical={calibration.typical_action_magnitude:.4f} "
          f"n_samples={calibration.n_samples}")

    declared = {
        "state":          model.required_inputs.state,
        "gripper":        model.required_inputs.gripper,
        "action_history": model.required_inputs.action_history,
        "instruction":    model.required_inputs.instruction,
        "images":         sorted(model.required_inputs.cameras),
    }
    print(f"  declared modalities: {declared}")

    pool = build_modality_pool(
        dataset, current_episode=args.episode,
        declared_modalities=declared,
        n_samples=args.pool_size,
        instruction_must_differ_from_task=scene.instruction,
        cache_dir=args.pool_cache_dir,
    )
    print(f"  pool sampled from episodes: {pool.metadata.get('sampled_episodes')}")
    print(f"  pool sizes: state={len(pool.state_samples)} "
          f"gripper={len(pool.gripper_samples)} "
          f"action_history={len(pool.action_history_samples)} "
          f"instruction={len(pool.instruction_samples)} "
          f"images={ {c: len(s) for c, s in pool.image_samples.items()} }")
    print(f"  ref_distance: {pool.ref_distance}")

    print(f"[4/4] running ModalityDropoutDiagnostic (K={args.k_samples})", flush=True)
    diag = ModalityDropoutDiagnostic(
        pool=pool, calibration=calibration,
        k_samples=args.k_samples,
    )
    result = diag.run(model, scene)
    print(f"\n=== VERDICT ===")
    print(f"  severity: {result.severity.value}")
    print(f"  explanation: {result.explanation}")
    print()
    pm = result.raw.get("per_modality", {})
    for modality, r in pm.items():
        if "skip_reason" in r:
            print(f"  {modality:<25} → {r['verdict']:>12}  ({r['skip_reason']})")
        else:
            print(f"  {modality:<25} → {r['verdict']:>12}  "
                  f"Δ_in={r['mean_intervention_mag']:.4f} "
                  f"Δ_out={r['mean_response_normalized']:.4f} "
                  f"ratio={r['sensitivity_ratio']:.4f}")

    # Save full result
    serializable = {
        "model": args.model, "model_id": model.model_id,
        "dataset": args.dataset, "episode": args.episode, "frame": args.frame,
        "instruction": scene.instruction,
        "calibration": calibration.to_summary(),
        "pool_metadata": {
            "sampled_episodes": pool.metadata.get("sampled_episodes"),
            "ref_distance":     pool.ref_distance,
            "sizes":            {
                "state":           len(pool.state_samples),
                "gripper":         len(pool.gripper_samples),
                "action_history":  len(pool.action_history_samples),
                "instruction":     len(pool.instruction_samples),
                "images":          {c: len(s) for c, s in pool.image_samples.items()},
            },
        },
        "severity":    result.severity.value,
        "explanation": result.explanation,
        "scalar":      float(result.scalar_score),
        "per_modality": pm,
        "k_samples":   args.k_samples,
    }
    (out / "result.json").write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n=== saved {out}/result.json ===")


if __name__ == "__main__":
    main()
