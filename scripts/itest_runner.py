"""Generalized multi-camera trajectory-story runner.

Same scaffold the previous OpenVLA-only script used, but parameterised so
every model + dataset can run through it. Honours the post-audit strict
contract: per-camera attention/sensitivity/target overlays in Rerun;
per-camera modality dropout; no silent primary-only fallback.

Invocation (called once per (model, scenario)):

    uv run python scripts/itest_runner.py \
        --story-id     openvla:bridge:ep0:clutter \
        --model-builder emboviz.models.registry:get_model:openvla-7b \
        --dataset-builder emboviz.datasets.lerobot_bridge:BridgeEpisodeSource \
        --episode-idx 0 \
        --frame-start 8 --n-frames 8 \
        --out-dir /root/itest/openvla/bridge_clutter_ep0 \
        --extra-camera-mapping ''           # for adapters that need one

The runner deliberately stays slim: it imports a builder, asks the
dataset for a trajectory, and runs the enhanced battery. All Tier 1 +
Tier 2 + post-audit fixes are exercised.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from emboviz.calibration import calibrate_model
from emboviz.core.results import Severity
from emboviz.core.types import TokenSelector, resolve_cameras
from emboviz.diagnostics.attention_drift import AttentionDriftDiagnostic
from emboviz.diagnostics.chunk_consistency import ChunkConsistencyDiagnostic
from emboviz.diagnostics.memorization import MemorizationDiagnostic
from emboviz.diagnostics.modality_dropout import ModalityDropoutDiagnostic
from emboviz.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from emboviz.diagnostics.trajectory import TrajectoryDiagnostic
from emboviz.exporters.correlation import find_failure_moments, format_failure_moments
from emboviz.exporters.rerun import export_rerun
from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.models.protocol import Capability
from emboviz.perturb._target_detection import GroundingDINOSAMDetector
from emboviz.perturb.instruction import PromptParaphrasePerturber


def _resolve(spec: str, kwargs_json: str = ""):
    """Resolve ``module.path:attr[:arg]`` into a callable that returns an instance.

    Examples:
      ``emboviz.models.registry:get_model:openvla-7b``
          → ``get_model("openvla-7b")()``  (the registry returns a class)
      ``emboviz.datasets.lerobot_bridge:BridgeEpisodeSource``
          → ``BridgeEpisodeSource()``
      ``emboviz.models.lerobot_policy:LeRobotPolicyAdapter`` with
        kwargs_json={"repo_id": "lerobot/smolvla_base", "camera_mapping": {...}}
          → ``LeRobotPolicyAdapter(repo_id=..., camera_mapping=...)``
    """
    parts = spec.split(":")
    module = importlib.import_module(parts[0])
    obj = getattr(module, parts[1])
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if len(parts) == 2:
        return obj(**kwargs)
    # registry-style: obj(arg) returns a class; then instantiate
    intermediate = obj(parts[2])
    if isinstance(intermediate, type):
        return intermediate(**kwargs)
    return intermediate


def run_story(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[runner] story={args.story_id}", flush=True)
    print(f"[runner] out_dir={out_dir}", flush=True)

    # --- 1. model + dataset ----------------------------------------------
    print(f"[1/7] load model: {args.model_builder} kwargs={args.model_kwargs_json}", flush=True)
    model = _resolve(args.model_builder, args.model_kwargs_json)
    print(f"      model_id={model.model_id}  caps={model.capabilities}", flush=True)
    print(f"      required_inputs.cameras={sorted(model.required_inputs.cameras)}", flush=True)

    print(f"[2/7] load dataset: {args.dataset_builder} kwargs={args.dataset_kwargs_json}", flush=True)
    dataset = _resolve(args.dataset_builder, args.dataset_kwargs_json)
    full_traj = dataset.load_trajectory(int(args.episode_idx))
    window_frames = full_traj.frames[args.frame_start : args.frame_start + args.n_frames]
    window_indices = list(
        full_traj.frame_indices[args.frame_start : args.frame_start + args.n_frames]
    )
    trajectory = replace(full_traj, frames=window_frames, frame_indices=window_indices)
    print(f"      trajectory: {len(trajectory.frames)} frames source={trajectory.source}", flush=True)
    print(f"      cameras in scene: {sorted(trajectory.frames[0].observations.images)}", flush=True)
    print(f'      instruction: "{trajectory.frames[0].instruction}"', flush=True)

    # --- 3a. CALIBRATION: noise floor + typical action magnitude --------
    print(f"[3a/7] calibrating model on this trajectory "
          f"(noise-floor + typical action magnitude)...", flush=True)
    calibration = calibrate_model(model, trajectory, n_noise_probes=5)
    print(f"      noise_floor              = {calibration.noise_floor:.4f}", flush=True)
    print(f"      typical_action_magnitude = {calibration.typical_action_magnitude:.4f}", flush=True)
    print(f"      n_samples (averaging)    = {calibration.n_samples}", flush=True)
    if calibration.single_sample_noise_floor is not None:
        print(f"      (single-sample noise floor was "
              f"{calibration.single_sample_noise_floor:.4f}; "
              f"averaging reduces it to {calibration.noise_floor:.4f})", flush=True)
    print(f"      → diagnostic scores reported on a 0-1 anchored scale", flush=True)

    # --- 3b. per-frame diagnostics with calibration ----------------------
    print(f"[3b/7] per-frame diagnostics across the window...", flush=True)
    gd_sam = GroundingDINOSAMDetector(device="cuda")
    per_axis: dict = {}

    # Track diagnostics that SKIP for architectural reasons (model doesn't
    # support what the axis measures). These don't appear in per_axis at all
    # — they're listed separately in summary.json as "not_applicable" so
    # the user knows why a given axis isn't reported.
    not_applicable: dict[str, str] = {}

    # Trajectory-level scalars (attention_drift, chunk_consistency) live in
    # ``trajectory_axes`` and follow the same shape as per_axis but they're
    # not wrapped in TrajectoryDiagnosticResult.
    trajectory_axes: dict = {}

    if model.capabilities & Capability.ATTENTION:
        drift = AttentionDriftDiagnostic().run_trajectory(model, trajectory)
        if drift.severity == Severity.UNKNOWN:
            not_applicable["internal.attention_drift"] = drift.explanation
        else:
            trajectory_axes["internal.attention_drift"] = {
                "severity":      drift.severity.value,
                "scalar_score":  drift.scalar_score,
                "explanation":   drift.explanation,
            }
            print(f"      attention_drift: {drift.severity.value} "
                  f"({drift.scalar_score:.1f}px)", flush=True)
    else:
        not_applicable["internal.attention_drift"] = (
            f"model {model.model_id} does not expose Capability.ATTENTION"
        )

    chunk = ChunkConsistencyDiagnostic(calibration=calibration).run_trajectory(model, trajectory)
    if chunk.severity == Severity.UNKNOWN:
        not_applicable["internal.chunk_consistency"] = chunk.explanation
    else:
        trajectory_axes["internal.chunk_consistency"] = {
            "severity":     chunk.severity.value,
            "scalar_score": chunk.scalar_score,
            "explanation":  chunk.explanation,
            "raw_mean_delta": chunk.raw.get("raw_mean_delta"),
        }
        print(f"      chunk_consistency: {chunk.severity.value} "
              f"(normalized mean_delta={chunk.scalar_score:.3f}, "
              f"raw={chunk.raw['raw_mean_delta']:.3f})", flush=True)

    print(f"      memorization (GD+SAM per camera, calibrated) ...", flush=True)
    memo = TrajectoryDiagnostic(
        MemorizationDiagnostic(target_detector=gd_sam, calibration=calibration),
        progress=True,
    )
    per_axis["vision.memorization"] = memo.run(model, trajectory)

    # Compute trajectory-level substitution values for state + action_history
    # so ModalityDropoutDiagnostic can substitute with a from-distribution
    # valid sample instead of zeros (zeros break structured representations
    # like GR00T's 6D rotation in eef_9d). Use the LAST frame's state as the
    # substitution for every frame — it's a real recorded state, guaranteed
    # valid, and uninformative for any non-last frame.
    last_scene = trajectory.frames[-1]
    sub_state = (
        np.asarray(last_scene.observations.state.values, dtype=np.float32)
        if last_scene.observations.state is not None else None
    )
    sub_hist = (
        np.asarray(last_scene.observations.action_history.actions, dtype=np.float32)
        if last_scene.observations.action_history is not None else None
    )
    print(f"      modality dropout (per camera + per modality, calibrated) ...", flush=True)
    md = TrajectoryDiagnostic(
        ModalityDropoutDiagnostic(
            calibration=calibration,
            substitution_state=sub_state,
            substitution_action_history=sub_hist,
        ),
        progress=True,
    )
    per_axis["input.modality_dropout"] = md.run(model, trajectory)

    print(f"      [3c] sensitivity map ({args.sensitivity_grid_side}x{args.sensitivity_grid_side}, per camera) ...", flush=True)
    sm = TrajectoryDiagnostic(
        SensitivityMapDiagnostic(
            grid_side=args.sensitivity_grid_side, calibration=calibration,
        ),
        progress=True,
    )
    per_axis["vision.scene_sensitivity"] = sm.run(model, trajectory)

    # --- 4. prompt paraphrase on frame 0 ---------------------------------
    print(f"[4/7] prompt paraphrase on frame 0...", flush=True)
    pp = PromptParaphrasePerturber()
    paraphrase_deltas = {}
    baseline_action = model.predict(trajectory.frames[0]).action
    for variant in pp.variants(trajectory.frames[0]):
        try:
            pred = model.predict(variant.scene).action
        except Exception as e:
            print(f"      paraphrase {variant.variant_id} failed: {type(e).__name__}: {e}", flush=True)
            continue
        paraphrase_deltas[variant.variant_id] = float(np.linalg.norm(pred - baseline_action))
        print(f'      "{variant.scene.instruction}" -> Δ={paraphrase_deltas[variant.variant_id]:.4f}', flush=True)
    paraphrase_mean = float(np.mean(list(paraphrase_deltas.values()))) if paraphrase_deltas else float("nan")

    # --- 5. per-camera Rerun overlays ------------------------------------
    print(f"[5/7] collect per-camera overlays for Rerun...", flush=True)
    attention_per_frame: dict[int, dict[str, np.ndarray]] = {}
    sensitivity_per_frame: dict[int, dict[str, np.ndarray]] = {}
    target_mask_per_frame: dict[int, dict[str, np.ndarray]] = {}

    if model.capabilities & Capability.ATTENTION:
        for i, scene in enumerate(trajectory.frames):
            try:
                am = model.extract_attention(scene, TokenSelector(relative="before_action"))
                img_attn = am.image_weights().mean(axis=(0, 1))
                # Attention is over a single image stream — log under "primary".
                attention_per_frame[trajectory.frame_indices[i]] = {"primary": img_attn}
            except Exception as e:
                print(f"      attn frame {i} failed: {type(e).__name__}: {e}", flush=True)
                break

    for i, r in enumerate(per_axis["vision.scene_sensitivity"].per_frame):
        grids = (r.raw or {}).get("sensitivity_grid_per_camera", {})
        if not grids:
            continue
        sensitivity_per_frame[trajectory.frame_indices[i]] = {
            cam: np.asarray(g, dtype=np.float32) for cam, g in grids.items()
        }

    for i, r in enumerate(per_axis["vision.memorization"].per_frame):
        raw = r.raw or {}
        detected = raw.get("detected_cameras") or []
        if not detected:
            continue
        scene = trajectory.frames[i]
        cam_masks = {}
        for cam in detected:
            probe = scene.with_image(scene.observations.images[cam].data, camera="primary") \
                    if cam != "primary" and "primary" in scene.observations.images \
                    else scene.with_image(scene.observations.images[cam].data, camera=cam)
            det = gd_sam(probe)
            if det is not None and det.mask is not None:
                cam_masks[cam] = det.mask
        if cam_masks:
            target_mask_per_frame[trajectory.frame_indices[i]] = cam_masks

    print(f"      attention frames: {len(attention_per_frame)}", flush=True)
    print(f"      sensitivity frames: {len(sensitivity_per_frame)}", flush=True)
    print(f"      target-mask frames: {len(target_mask_per_frame)}", flush=True)

    # --- 6. expert delta (imitation-accuracy only) + Rerun + failure moments
    # expert_delta is an IMITATION-ACCURACY metric, not a deployment-debugging
    # metric. It compares the model's prediction to the action a human
    # demonstrator recorded for the SAME frame. Only meaningful when the
    # dataset has recorded expert actions (training/validation demos) —
    # NOT for raw deployment rollouts where the robot was driven by the
    # model and no human ever teleoperated those frames.
    print(f"[6/7] expert delta (imitation accuracy) + Rerun export...", flush=True)
    has_expert = any(
        scene.metadata.get("expert_action") is not None
        for scene in trajectory.frames
    )
    profile = trajectory.frames[0].profile
    dim_names: list[str] = (
        list(profile.action.dim_names)
        if profile is not None and profile.action is not None
           and profile.action.dim_names is not None
        else []
    )

    expert_delta_per_frame: list[float] = []
    expert_delta_per_frame_per_dim: list[list[float]] = []
    per_dim_mean_abs: list[float] = []
    per_dim_max_abs: list[float] = []
    dim_label: list[str] = []
    mean_delta: float = float("nan")

    if not has_expert:
        print(f"      expert_delta SKIPPED — dataset has no recorded "
              f"expert actions (typical for deployment rollouts).", flush=True)
        not_applicable["evaluation.expert_delta"] = (
            "dataset has no recorded expert action column "
            "(scene.metadata['expert_action'] is None on every frame). "
            "expert_delta only applies to training/validation demos where "
            "a human demonstrator's action was recorded per frame; raw "
            "deployment rollouts can't be evaluated against an expert "
            "they don't have."
        )
    else:
        n_compared_dims = 0
        for scene in trajectory.frames:
            pred = model.predict(scene).action
            expert = scene.metadata.get("expert_action")
            if expert is None:
                # Mixed case: this frame lacks expert; skip cleanly
                continue
            expert_arr = np.asarray(expert, dtype=np.float32)
            n = min(len(pred), len(expert_arr))
            n_compared_dims = n
            per_dim = (pred[:n] - expert_arr[:n]).tolist()
            expert_delta_per_frame_per_dim.append(per_dim)
            expert_delta_per_frame.append(float(np.linalg.norm(per_dim)))
        if expert_delta_per_frame:
            mean_delta = float(np.mean(expert_delta_per_frame))
            arr = np.array(expert_delta_per_frame_per_dim, dtype=np.float32)
            per_dim_mean_abs = np.abs(arr).mean(axis=0).tolist()
            per_dim_max_abs  = np.abs(arr).max(axis=0).tolist()
            dim_label = (
                dim_names[:n_compared_dims]
                if dim_names and len(dim_names) >= n_compared_dims
                else [f"d{i}" for i in range(n_compared_dims)]
            )
            print(f"      imitation L2 per-frame: "
                  f"{[f'{d:.3f}' for d in expert_delta_per_frame]}", flush=True)
            print(f"      mean L2: {mean_delta:.3f}", flush=True)
            print(f"      per-dim |Δ| (NOTE: imitation accuracy on demo "
                  f"data; not a deployment-failure signal):", flush=True)
            for nm, m, mx in zip(dim_label, per_dim_mean_abs, per_dim_max_abs):
                print(f"        {nm:>10s} : mean={m:.3f}  max={mx:.3f}", flush=True)
            med = float(np.median(per_dim_mean_abs))
            suspects = [
                (nm, m) for nm, m in zip(dim_label, per_dim_mean_abs)
                if med > 1e-6 and m >= 3 * med
            ]
            if suspects:
                print(f"      ⚠️ convention-mismatch suspects (dim |Δ| >= 3× median):", flush=True)
                for nm, m in suspects:
                    print(f"        {nm} mean|Δ|={m:.3f} vs median={med:.3f}", flush=True)

    moments = find_failure_moments(
        per_axis,
        expert_delta_per_frame=expert_delta_per_frame if has_expert else None,
        min_critical_axes=2,
    )
    print(f"      failure moments (≥2 critical):", flush=True)
    print(format_failure_moments(moments, max_show=10), flush=True)

    rrd_path = out_dir / "rollout.rrd"
    export_rerun(
        trajectory, per_axis, rrd_path,
        application_id=f"emboviz:{args.story_id}",
        attention_per_frame=attention_per_frame,
        sensitivity_per_frame=sensitivity_per_frame,
        target_mask_per_frame=target_mask_per_frame,
    )
    print(f"      wrote {rrd_path} ({rrd_path.stat().st_size:,} bytes)", flush=True)

    # --- 7. JSON summary -------------------------------------------------
    summary = {
        "story_id":          args.story_id,
        "model":             model.model_id,
        "required_cameras":  sorted(model.required_inputs.cameras),
        "scene_cameras":     sorted(trajectory.frames[0].observations.images),
        "trajectory_source": trajectory.source,
        "n_frames":          len(trajectory.frames),
        "frame_indices":     list(trajectory.frame_indices),
        "instruction":       trajectory.frames[0].instruction,
        "calibration":       calibration.to_summary(),
        "per_axis":          {axis: tr.to_summary() for axis, tr in per_axis.items()},
        "trajectory_axes":   trajectory_axes,
        "not_applicable":    not_applicable,
        "paraphrase_deltas": paraphrase_deltas,
        "paraphrase_mean_delta": paraphrase_mean,
        # Imitation-accuracy block — only populated when the dataset has
        # recorded expert actions. Absent when running on deployment rollouts.
        "imitation_accuracy": (
            {
                "expert_delta_per_frame":         expert_delta_per_frame,
                "expert_delta_mean":              mean_delta,
                "action_dim_names":               dim_label,
                "expert_delta_per_dim_mean_abs":  per_dim_mean_abs,
                "expert_delta_per_dim_max_abs":   per_dim_max_abs,
                "expert_delta_per_frame_per_dim": expert_delta_per_frame_per_dim,
                "note": (
                    "Imitation-accuracy is per-frame distance between the "
                    "model's prediction and the dataset's recorded expert "
                    "action. Useful for validation runs against training/demo "
                    "data; NOT a deployment-debug signal."
                ),
            } if has_expert else None
        ),
        "failure_moments": [
            {
                "frame_idx": fm.frame_idx,
                "n_critical_axes": fm.n_critical_axes,
                "critical_axes": fm.critical_axes,
                "expert_delta": fm.expert_delta,
                "notes": fm.notes,
            }
            for fm in moments
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] total {time.time() - t0:.1f}s", flush=True)
    print(f"  rerun {rrd_path}", flush=True)
    print(f"  cat {out_dir}/summary.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--story-id", required=True)
    p.add_argument("--model-builder", required=True,
                   help="module.path:attr[:arg] resolving to a VLAModel instance")
    p.add_argument("--model-kwargs-json", default="",
                   help="JSON dict of kwargs passed to the model builder")
    p.add_argument("--dataset-builder", required=True,
                   help="module.path:attr[:arg] resolving to an EpisodeSource instance")
    p.add_argument("--dataset-kwargs-json", default="",
                   help="JSON dict of kwargs passed to the dataset builder")
    p.add_argument("--episode-idx", type=int, required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--sensitivity-grid-side", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    try:
        run_story(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
