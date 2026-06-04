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
from emboviz.world_models.keyframes import detect_keyframes
from emboviz.world_models.simulate import closed_loop_rollout
from emboviz.world_models.viz import frames_to_arrays, save_video


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
    args = p.parse_args()

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

    wm = connect_world_model("cosmos3", world_model_kwargs={
        "server_url": cs.server_url, "domain_name": cs.domain,
        "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
    })

    # Adapter-side pieces (Cosmos-specific) — lazily imported on this driver path.
    from emboviz.adapters import connect
    from emboviz.config import _JOINT_ACTION_CONVENTIONS
    from emboviz_cosmos3.bridge import make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view
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
        )
        return PolicyDreamStepper(
            policy.client.predict,
            tracker=tracker,
            camera_map=cs.camera_map,
            instruction=frame.instruction,
            n_actions=cs.n_actions,
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
            from PIL import Image
            Image.fromarray(seed, mode="RGB").save(clip_dir / "seed.png")

            def on_step(i: int, traj, _dir=clip_dir) -> None:
                arrs = frames_to_arrays(traj, cs.conditioning_camera)
                save_video(arrs, _dir / f"step_{i:02d}.mp4", fps=real.fps)
                print(f"    step {i}: {len(arrs)} frames (saved)", flush=True)

            print(f"  [dream] frame {kf.index} ({kf.kind}) / {tag}: "
                  f"{cs.n_loop_steps} turns x {cs.n_actions} ...")
            dream = closed_loop_rollout(
                wm, seed, stepper_for(seed_index),
                n_steps=cs.n_loop_steps, conditioning_camera=cs.conditioning_camera,
                on_step=on_step,
            )

            arrs = frames_to_arrays(dream.trajectory, cs.conditioning_camera)
            save_video(arrs, clip_dir / "dream.mp4", fps=real.fps)

            verdict = None
            if reasoner is not None:
                verdict = reasoner.judge(arrs, cs.reasoner_question)
                print(f"    verdict: {verdict}")

            record = {
                "keyframe_index": kf.index, "kind": kf.kind, "seed_index": seed_index,
                "perturbation": instruction, "n_loop_steps": cs.n_loop_steps,
                "n_actions": cs.n_actions, "n_frames": len(arrs),
                "reasoner_question": cs.reasoner_question, "verdict": verdict,
            }
            (clip_dir / "verdict.json").write_text(json.dumps(record, indent=2))
            summary.append(record)
            (out / "summary.json").write_text(json.dumps(
                {"episode": episode, "policy": cs.policy_adapter, "clips": summary}, indent=2
            ))

    print(f"\n[dream] DONE: {len(summary)} clips -> {out}/")


if __name__ == "__main__":
    main()
