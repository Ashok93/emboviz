"""Policy-side dress rehearsal for the Cosmos closed loop — REAL policy, NO world model.

Runs the exact ``dream_cli`` path up to (but not including) Cosmos: load the
episode, connect the policy, build the seed concat + state tracker, and execute
ONE stepper turn. It confirms the policy's action chunk matches the bridge
(shape and layout) and that forward kinematics + the DROID encoder produce
``(n_actions, action_dim)`` conditioning — everything the GPU world-model run
depends on except the world model itself. If this passes, the only remaining
unknown is the Cosmos GPU server.

    uv run python tools/dream_smoke.py --config configs/droid_pi0.yaml --episode 312
"""

from __future__ import annotations

import argparse

import numpy as np

from emboviz.adapters import connect
from emboviz.config import _JOINT_ACTION_CONVENTIONS, load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import detect_keyframes


def _img(frame, role: str) -> np.ndarray:
    if role not in frame.observations.images:
        raise SystemExit(
            f"camera role {role!r} (from concat_cameras) is not in the episode "
            f"(available: {sorted(frame.observations.images)})."
        )
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    if cs is None or cs.policy_adapter is None:
        raise SystemExit("config needs analysis.cosmos_stress with a policy_adapter.")
    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[smoke] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    keyframes = detect_keyframes(real)
    print(f"[smoke] {len(real.frames)} frames, fps {real.fps:g}; {len(keyframes)} keyframes")
    if not keyframes:
        raise SystemExit("no keyframes detected — cannot pick a seed.")

    from emboviz_cosmos3.bridge import make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view
    from emboviz_cosmos3.dream_step import PolicyDreamStepper

    kinematics = None
    if cs.action_convention in _JOINT_ACTION_CONVENTIONS:
        from emboviz_robot import load_kinematics
        if cs.robot is not None:
            kinematics = load_kinematics(cs.robot)
        else:
            kinematics = load_kinematics(
                urdf=cs.robot_urdf, ee_frame=cs.robot_ee_frame, joint_names=cs.robot_joint_names
            )
        print(f"[smoke] forward kinematics: {cs.robot or cs.robot_urdf} "
              f"({kinematics.ee_frame}, {kinematics.n_joints} joints)")

    print(f"[smoke] connecting policy '{cs.policy_adapter}' {cs.policy_kwargs or '{}'} "
          "(builds the worker venv on first run — this can take a while) ...")
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)

    lead = int(round(cs.lead_s * real.fps))
    kf = keyframes[0]
    seed_index = max(0, kf.index - lead)
    frame = real.frames[seed_index]
    if frame.observations.state is None or frame.observations.gripper is None:
        raise SystemExit(f"seed frame {seed_index} lacks state/gripper.")
    print(f"[smoke] seed frame {seed_index} (keyframe {kf.index} {kf.kind}); "
          f"instruction={frame.instruction!r}")
    print(f"[smoke] seed state ({cfg.dataset.state.convention}, dim "
          f"{np.asarray(frame.observations.state.values).size}); "
          f"gripper {float(frame.observations.gripper.value):.3f}")

    seed_concat = build_concat_view(
        _img(frame, cs.concat_cameras["wrist"]),
        _img(frame, cs.concat_cameras["exterior_left"]),
        _img(frame, cs.concat_cameras["exterior_right"]),
    )
    print(f"[smoke] seed concat {seed_concat.shape}")

    tracker = make_state_tracker(
        np.asarray(frame.observations.state.values, dtype=np.float32),
        float(frame.observations.gripper.value),
        action_convention=cs.action_convention,
        state_convention=cs.state_convention,
        kinematics=kinematics,
    )
    stepper = PolicyDreamStepper(
        policy.client.predict,
        tracker=tracker,
        camera_map=cs.camera_map,
        instruction=frame.instruction,
        n_actions=cs.n_actions,
    )

    print("[smoke] running ONE stepper turn (real policy -> bridge -> FK -> Cosmos conditioning) ...")
    cosmos_actions = stepper(seed_concat)

    expected = (cs.n_actions, cs.action_dim)
    print(f"[smoke] cosmos conditioning shape {cosmos_actions.shape} (expect {expected})")
    print(f"[smoke]   action[0] = {np.round(cosmos_actions[0], 4)}")
    print(f"[smoke]   value range: min {cosmos_actions.min():.3f}  max {cosmos_actions.max():.3f}")
    if cosmos_actions.shape != expected:
        raise SystemExit(
            f"FAIL: conditioning shape {cosmos_actions.shape} != expected {expected}. "
            "The policy's chunk does not match the bridge — inspect the policy output."
        )
    print("\n[smoke] PASS — policy + bridge + forward kinematics + DROID encoder all agree. "
          "Only the Cosmos GPU server remains to wire in (set cosmos_stress.server_url).")


if __name__ == "__main__":
    main()
