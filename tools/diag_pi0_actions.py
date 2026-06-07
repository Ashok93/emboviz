"""Diagnostic: compare pi0's predicted action magnitude against the recorded demo.

Root-cause tool for the closed-loop dream collapse. For a few real frames it runs
pi0 on the REAL observation (no world model) and prints, side by side:

  * pi0's raw joint deltas (first N steps of its chunk) vs the recorded demo's
    joint deltas at the same frame,
  * the resulting end-effector motion per step (via forward kinematics), and
  * the Cosmos droid_lerobot conditioning each produces + its saturation fraction.

If pi0's deltas on real images are ~the recorded magnitude, the policy bridge is
fine and the closed-loop collapse comes from the dream degrading pi0's input.
If pi0's deltas are far larger, the problem is the policy/bridge itself (scale,
convention, or an off-distribution seed) — independent of the world model.

    uv run python tools/diag_pi0_actions.py --config configs/cosmos_droid_pi0_demo.yaml \
        --episode 312 --frames 0,30,60,90
"""

from __future__ import annotations

import argparse

import numpy as np

from emboviz.adapters import connect
from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source

np.set_printoptions(precision=4, suppress=True)


def _img(frame, role: str) -> np.ndarray:
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def _joints(frame) -> np.ndarray:
    return np.asarray(frame.observations.state.values, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--frames", default="0,30,60,90")
    p.add_argument("--n", type=int, default=4, help="steps of the chunk to inspect")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    episode = args.episode if args.episode is not None else int(str(cfg.analysis.episodes).split(",")[0])

    from emboviz_wire.observations import RGBImage
    from emboviz_wire.types import Observations, Scene
    from emboviz_cosmos3.bridge import integrate_joint_chunk, make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
    from emboviz_cosmos3.domains import encode_droid_states
    from emboviz_robot import load_kinematics

    print(f"[diag] loading episode {episode} ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    n_frames = len(real.frames)
    print(f"[diag] {n_frames} frames, fps {real.fps:g}")

    kin = load_kinematics(cs.robot)
    print("[diag] connecting pi0 ...")
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)

    def encode_from_joints(joint_seq: np.ndarray, grippers: np.ndarray) -> np.ndarray:
        """FK each joint config -> EE pose [xyz, euler] -> droid_lerobot conditioning."""
        rows = []
        for q in joint_seq:
            t, e = kin.fk(q).as_xyz_euler()
            rows.append(np.concatenate([t, e]))
        return encode_droid_states(np.stack(rows).astype(np.float32), grippers.astype(np.float32))

    N = args.n
    for fi in [int(x) for x in args.frames.split(",")]:
        if fi + N + 1 >= n_frames:
            print(f"\n=== frame {fi}: too close to end (need {N+1} ahead), skipping ===")
            continue
        frame = real.frames[fi]
        print(f"\n================= frame {fi}  (instruction: {frame.instruction!r}) =================")

        # --- pi0 on the REAL observation (same camera_map the loop uses) ---
        seed_concat = build_concat_view(
            _img(frame, cs.concat_cameras["wrist"]),
            _img(frame, cs.concat_cameras["exterior_left"]),
            _img(frame, cs.concat_cameras["exterior_right"]),
            wrist_size=cs.concat_resolution,
        )
        regions = split_concat_view(seed_concat)
        tracker = make_state_tracker(
            _joints(frame), float(frame.observations.gripper.value),
            action_convention=cs.action_convention, state_convention=cs.state_convention,
            kinematics=kin, control_hz=cs.control_hz,
        )
        scene = Scene(
            observations=Observations(
                images={role: RGBImage(data=regions[region], camera_id=role)
                        for role, region in cs.camera_map.items()},
                state=tracker.proprioception(), gripper=tracker.gripper_state(),
            ),
            instruction=frame.instruction,
        )
        chunk = np.asarray(policy.client.predict(scene).action_chunk, dtype=np.float32)  # (H, 8)
        pi0_jd = chunk[:N, :7]
        pi0_grip = chunk[:N, 7]

        # RAW vectors — is pi0's output an absolute joint target, a per-step delta,
        # or normalized? Compare row 0/1/3 directly to the real joint configs.
        sj = _joints(frame)
        print(f"  seed joints       : {sj}")
        print(f"  pi0 action row 0  : {chunk[0, :7]}")
        print(f"  pi0 action row 1  : {chunk[1, :7]}")
        print(f"  pi0 action row 3  : {chunk[3, :7]}")
        print(f"  REAL joints  i+1  : {_joints(real.frames[fi + 1])}")
        print(f"  REAL joints  i+3  : {_joints(real.frames[fi + 3])}")
        print(f"  REAL delta i->i+1 : {_joints(real.frames[fi + 1]) - sj}")
        print(f"  pi0 row0 - seed   : {chunk[0, :7] - sj}   (≈0 if absolute target near seed)")

        # --- recorded demo deltas at the same frame ---
        rec_joint = np.stack([_joints(real.frames[fi + k]) for k in range(N + 1)])  # (N+1, 7)
        rec_jd = np.diff(rec_joint, axis=0)                                          # (N, 7)
        rec_grip = np.array([float(real.frames[fi + k].observations.gripper.value) for k in range(N)])

        print(f"  pi0   |joint delta| per step: {np.abs(pi0_jd).max(axis=1)}   (max {np.abs(pi0_jd).max():.4f} rad)")
        print(f"  demo  |joint delta| per step: {np.abs(rec_jd).max(axis=1)}   (max {np.abs(rec_jd).max():.4f} rad)")
        ratio = np.abs(pi0_jd).max() / max(np.abs(rec_jd).max(), 1e-6)
        print(f"  >>> pi0 / demo joint-motion ratio: {ratio:.1f}x")

        # --- EE motion per step (FK) ---
        pi0_joint_seq, _, _ = integrate_joint_chunk(_joints(frame), chunk[:N], kin, dt=1.0 / cs.control_hz)
        pi0_ee = np.stack([kin.fk(q).translation for q in pi0_joint_seq])
        rec_ee = np.stack([kin.fk(q).translation for q in rec_joint])
        print(f"  pi0   EE move/step (mm): {np.linalg.norm(np.diff(pi0_ee, axis=0), axis=1)*1000}")
        print(f"  demo  EE move/step (mm): {np.linalg.norm(np.diff(rec_ee, axis=0), axis=1)*1000}")

        # --- conditioning + saturation ---
        pi0_cond = encode_from_joints(pi0_joint_seq, pi0_grip)
        rec_cond = encode_from_joints(rec_joint, rec_grip)
        print(f"  pi0   conditioning[0]: {pi0_cond[0]}  sat={np.mean(np.abs(pi0_cond)>=0.999):.2f}")
        print(f"  demo  conditioning[0]: {rec_cond[0]}  sat={np.mean(np.abs(rec_cond)>=0.999):.2f}")


if __name__ == "__main__":
    main()
