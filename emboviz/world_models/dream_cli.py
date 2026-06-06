"""Closed-loop world-model stress test — run a policy inside the Cosmos simulator.

Given a recorded episode, this finds the decisive instants, optionally perturbs
each seed frame with an editing instruction ("rotate the cup 90 degrees", "replace
the cup with a rubber duck"), then flies the user's policy inside Cosmos step by
step from that perturbed scene and asks the reasoner what happened. The simulator
*is* Cosmos; the policy is the thing under test.

Everything comes from the run config's ``analysis.cosmos_stress`` section (the
server, the policy, the perturbations, the camera maps). Output per clip is the
**dream video** + the **reasoner verdict** — no pixel divergence, because a
perturbed scene never happened in reality and has nothing to compare against.

For the unperturbed recorded-action *faithfulness* check (does Cosmos reproduce
reality on real actions), use ``stress_cli`` instead.

Run::

    uv run python -m emboviz.world_models.dream_cli --config configs/droid.yaml \
        --episode 0 --out outputs/cosmos_dream
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from emboviz.adapters import connect_world_model
from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import Keyframe, detect_keyframes
from emboviz.world_models.simulate import closed_loop_rollout
from emboviz.world_models.dream_rerun import export_dream_rerun
from emboviz.world_models.viz import frames_to_arrays


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "edit"


def _img(scene, role: str) -> np.ndarray:
    if role not in scene.observations.images:
        raise SystemExit(
            f"camera role '{role}' (from cosmos_stress.concat_cameras) is not in the "
            f"episode (available: {sorted(scene.observations.images)}). Fix the mapping."
        )
    return np.asarray(scene.observations.images[role].data, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--out", default="outputs/cosmos_dream")
    p.add_argument(
        "--max-clips", type=int, default=None,
        help="Stop after this many clips are produced (a cheap smoke run before "
             "the full keyframe sweep). Default: all keyframes x perturbations.",
    )
    p.add_argument(
        "--keyframe-kinds", default=None,
        help="Comma-separated keyframe kinds to keep (e.g. 'gripper_change' for "
             "the grasp/release only, skipping the many 'settle' keyframes). "
             "Default: all kinds.",
    )
    p.add_argument(
        "--near-frame", type=int, default=None,
        help="Keep only the single keyframe nearest this frame index (applied "
             "after --keyframe-kinds). Use to dream one decisive moment — e.g. "
             "--keyframe-kinds gripper_change --near-frame 60 picks the grasp.",
    )
    p.add_argument(
        "--seed-frames", default=None,
        help="Comma-separated frame indices to dream, bypassing keyframe "
             "detection — one clip per index, each seeded lead_s before it. Use "
             "to tile a spread of moments across the whole task in ONE run "
             "(workers load once), e.g. '20,40,61,80,100,120'.",
    )
    args = p.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        raise SystemExit(f"--max-clips must be >= 1, got {args.max_clips}.")

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    if cs is None:
        raise SystemExit("config has no analysis.cosmos_stress section — nothing to run.")
    if cs.policy_adapter is None:
        raise SystemExit(
            "dream_cli runs the closed-loop *policy* simulator, so cosmos_stress."
            "policy_adapter is required. For the recorded-action faithfulness "
            "baseline (no policy), use stress_cli --source recorded."
        )

    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[dream] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    keyframes = detect_keyframes(real)
    print(f"[dream] {len(real.frames)} frames, fps {real.fps:g}; {len(keyframes)} keyframes")

    # Optional keyframe selection: dream a chosen set of moments instead of
    # sweeping every keyframe. --seed-frames overrides detection with an explicit
    # list (one clip per frame); otherwise --keyframe-kinds filters by kind and
    # --near-frame keeps the single nearest detected keyframe.
    if args.seed_frames:
        idxs = [int(x) for x in args.seed_frames.split(",") if x.strip()]
        bad = [i for i in idxs if not 0 <= i < len(real.frames)]
        if bad:
            raise SystemExit(f"--seed-frames out of range {bad} (episode has {len(real.frames)} frames).")
        keyframes = [Keyframe(i, "manual", 0.0, 0.0) for i in idxs]
        print(f"[dream] --seed-frames: dreaming {len(keyframes)} explicit frames {idxs}")
    else:
        if args.keyframe_kinds:
            kinds = {k.strip() for k in args.keyframe_kinds.split(",") if k.strip()}
            keyframes = [kf for kf in keyframes if kf.kind in kinds]
            print(f"[dream] kept {len(keyframes)} keyframes of kind(s) {sorted(kinds)}")
        if args.near_frame is not None:
            if not keyframes:
                raise SystemExit("--near-frame: no keyframes left to choose from after --keyframe-kinds.")
            chosen = min(keyframes, key=lambda kf: abs(kf.index - args.near_frame))
            keyframes = [chosen]
            print(f"[dream] --near-frame {args.near_frame}: dreaming keyframe "
                  f"{chosen.index} ({chosen.kind})")
    if not keyframes:
        raise SystemExit("no keyframes selected — relax --keyframe-kinds / --near-frame / --seed-frames.")

    # fps is the conditioning frame rate Cosmos reads — it MUST equal the rate at
    # which the action deltas are sampled, or the model misreads the motion
    # dynamics. For droid_lerobot the dataset is 15 Hz (one generated frame per
    # control step), so fps = control_hz; the model only ever saw this domain at
    # 15 fps and the adapter default (10) is off-distribution. guardrails=False
    # matches NVIDIA's robotics forward-dynamics cookbook (run_fd_with_vllm.ipynb),
    # which disables the safety filter for the autoregressive DROID rollout.
    wm = connect_world_model("cosmos3", world_model_kwargs={
        "server_url": cs.server_url, "domain_name": cs.domain,
        "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
        "fps": int(round(cs.control_hz)), "guardrails": False,
    })

    # Adapter-side pieces (Cosmos-specific) — lazily imported on this driver path.
    from emboviz.adapters import connect
    from emboviz.config import _JOINT_ACTION_CONVENTIONS
    from emboviz_cosmos3.bridge import make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
    from emboviz_cosmos3.dream_step import PolicyDreamStepper
    from emboviz_cosmos3.perturb import CosmosImageEditor

    editor = CosmosImageEditor(cs.server_url)
    reasoner = None
    if cs.reasoner_url:
        from emboviz_cosmos3.reason import CosmosReasoner
        reasoner = CosmosReasoner(cs.reasoner_url)

    print(f"[dream] connecting policy '{cs.policy_adapter}' (kwargs: {cs.policy_kwargs or '{}'}) ...")
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)

    # Joint-space conventions need forward kinematics (joints -> EE pose); build
    # the robot once. Cartesian conventions track the pose directly (kinematics
    # stays None). emboviz-robot is imported only on this branch.
    kinematics = None
    if cs.action_convention in _JOINT_ACTION_CONVENTIONS:
        from emboviz_robot import load_kinematics
        if cs.robot is not None:
            print(f"[dream] forward kinematics: preconfigured robot '{cs.robot}'")
            kinematics = load_kinematics(cs.robot)
        else:
            print(f"[dream] forward kinematics: custom URDF {cs.robot_urdf}")
            kinematics = load_kinematics(
                urdf=cs.robot_urdf, ee_frame=cs.robot_ee_frame, joint_names=cs.robot_joint_names
            )

    def seed_concat(seed_index: int) -> np.ndarray:
        frame = real.frames[seed_index]
        return build_concat_view(
            _img(frame, cs.concat_cameras["wrist"]),
            _img(frame, cs.concat_cameras["exterior_left"]),
            _img(frame, cs.concat_cameras["exterior_right"]),
            wrist_size=cs.concat_resolution,
        )

    def stepper_for(seed_index: int) -> PolicyDreamStepper:
        frame = real.frames[seed_index]
        if frame.observations.state is None or frame.observations.gripper is None:
            raise SystemExit(
                f"seed frame {seed_index} lacks state/gripper needed to anchor the "
                "policy's actions. Map the policy's state (joint_position for a "
                "joint convention, cartesian_position for a cartesian one) + gripper "
                "in the config."
            )
        tracker = make_state_tracker(
            np.asarray(frame.observations.state.values, dtype=np.float32),
            float(frame.observations.gripper.value),
            action_convention=cs.action_convention,
            state_convention=cs.state_convention,
            kinematics=kinematics,
            control_hz=cs.control_hz,
        )
        return PolicyDreamStepper(
            policy.client.predict,
            tracker=tracker,
            camera_map=cs.camera_map,
            instruction=frame.instruction,
            n_actions=cs.n_actions,
            execute_steps=cs.execute_steps,
        )

    instructions = list(cs.perturbations) if cs.perturbations else [None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    lead = int(round(cs.lead_s * real.fps))

    for kf in keyframes:
        seed_index = max(0, kf.index - lead)
        if seed_index >= len(real.frames) - 1:
            continue  # no room to seed from this keyframe
        for instruction in instructions:
            tag = _slug(instruction) if instruction else "unperturbed"
            clip_dir = out / f"clip_{kf.index:04d}_{kf.kind}__{tag}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            seed = seed_concat(seed_index)
            if instruction:
                print(f"  [edit] frame {kf.index}: {instruction!r}")
                seed = editor.edit(seed, instruction)

            def on_step(i: int, traj) -> None:
                print(f"    step {i}: {len(traj.frames)} frame(s) committed", flush=True)

            commit = cs.execute_steps if cs.execute_steps is not None else cs.n_actions
            print(f"  [dream] frame {kf.index} ({kf.kind}) / {tag}: "
                  f"{cs.n_loop_steps} turns x dream {cs.n_actions}, commit {commit} ...")
            dream = closed_loop_rollout(
                wm, seed, stepper_for(seed_index),
                n_steps=cs.n_loop_steps, conditioning_camera=cs.conditioning_camera,
                instruction=real.frames[seed_index].instruction,
                execute_steps=cs.execute_steps,
                on_step=on_step,
            )

            # The dream frames are the full concat; split out ALL three regions so
            # the viewer shows every camera. Each region's left panel is the SAME
            # physical camera from the recorded episode (concat_cameras maps the
            # region to its episode role), over the same time span, aligned frame-
            # for-frame (one committed frame per timestep).
            dream_concat = frames_to_arrays(dream.trajectory, cs.conditioning_camera)
            split = [split_concat_view(f) for f in dream_concat]
            original_window = [
                real.frames[seed_index + i]
                for i in range(len(split))
                if seed_index + i < len(real.frames)
            ]
            dream_views: dict[str, list] = {}
            original_views: dict[str, list] = {}
            for region in ("wrist", "exterior_left", "exterior_right"):
                view_role = cs.concat_cameras[region]
                dream_views[region] = [s[region] for s in split]
                original_views[region] = frames_to_arrays(original_window, view_role)

            rrd_path = export_dream_rerun(
                clip_dir / "dream.rrd",
                original_views=original_views,
                dream_views=dream_views,
                seed_concat=seed,
                instruction=real.frames[seed_index].instruction,
                perturbation=instruction,
                fps=real.fps,
                policy_name=cs.policy_adapter,
                recording_id=f"dream_{episode}_{kf.index:04d}_{tag}",
            )
            print(f"    saved {rrd_path}")

            verdict = None
            if reasoner is not None:
                verdict = reasoner.judge(dream_concat, cs.reasoner_question)
                print(f"    verdict: {verdict}")

            record = {
                "keyframe_index": kf.index, "kind": kf.kind, "seed_index": seed_index,
                "perturbation": instruction, "n_loop_steps": cs.n_loop_steps,
                "n_actions": cs.n_actions, "execute_steps": commit, "n_frames": len(split),
                "reasoner_question": cs.reasoner_question, "verdict": verdict,
            }
            (clip_dir / "verdict.json").write_text(json.dumps(record, indent=2))
            summary.append(record)
            (out / "summary.json").write_text(json.dumps(
                {"episode": episode, "policy": cs.policy_adapter, "clips": summary}, indent=2
            ))

            if args.max_clips is not None and len(summary) >= args.max_clips:
                print(f"[dream] reached --max-clips {args.max_clips}; stopping early.")
                break
        else:
            continue   # inner loop finished without hitting the cap → next keyframe
        break          # inner loop broke on the cap → stop the keyframe sweep too

    print(f"\n[dream] DONE: {len(summary)} clips -> {out}/")


if __name__ == "__main__":
    main()
