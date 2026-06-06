"""Diagnostic: does the policy track the expert demo on REAL observations?

The decisive "can this π0 actually do the task" test — no Cosmos, no dreaming.
For each frame in a window (skipping the initial settle), feed the policy the
REAL recorded observation (ground-truth images + joint state + gripper) and
compare what it commands against what the expert demo actually did:

  * EE direction agreement — cosine between the policy's committed end-effector
    displacement (FK of its integrated action chunk) and the demo's actual EE
    displacement over the same horizon. ~+1 = heading the same way; <=0 = wrong way.
  * EE magnitude ratio — is the motion the right scale.
  * Gripper trajectory — does the policy close the gripper around where the demo
    does (the grasp), or never commit to the grasp.

Run for both candidate checkpoints to compare language-following:

    uv run python tools/diag_pi0_vs_expert.py --config configs/droid_pi0.yaml \
        --episode 312 --start 30 --end 100 --stride 5 --config-name pi0_droid
    uv run python tools/diag_pi0_vs_expert.py --config configs/droid_pi0.yaml \
        --episode 312 --start 30 --end 100 --stride 5 --config-name pi0_fast_droid
"""

from __future__ import annotations

import argparse

import numpy as np

from emboviz.adapters import connect
from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source

np.set_printoptions(precision=4, suppress=True)


def _img(frame, role):
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def _joints(frame):
    return np.asarray(frame.observations.state.values, dtype=np.float32)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--start", type=int, default=30, help="first frame (skip the settle)")
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--horizon", type=int, default=8, help="EE displacement horizon (steps)")
    p.add_argument("--config-name", default=None, help="override policy config_name (e.g. pi0_fast_droid)")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    episode = args.episode if args.episode is not None else int(str(cfg.analysis.episodes).split(",")[0])

    from emboviz_wire.observations import RGBImage
    from emboviz_wire.types import Observations, Scene
    from emboviz_cosmos3.bridge import integrate_joint_chunk, make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
    from emboviz_robot import load_kinematics

    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    n_frames = len(real.frames)
    kin = load_kinematics(cs.robot)
    dt = 1.0 / cs.control_hz
    H = args.horizon

    policy_kwargs = dict(cs.policy_kwargs or {})
    if args.config_name:
        policy_kwargs["config_name"] = args.config_name
    cfg_name = policy_kwargs.get("config_name", "?")
    print(f"[vs-expert] episode {episode}, {n_frames} frames; policy config_name={cfg_name!r}")
    print(f"[vs-expert] window {args.start}..{args.end} stride {args.stride}, EE horizon {H} steps")
    policy = connect(cs.policy_adapter, actor_kwargs=policy_kwargs or None)

    def ee(q):
        return np.asarray(kin.fk(q).translation, dtype=np.float64)

    cosines, grip_rows = [], []
    print(f"\n{'frame':>5} {'cos(dir)':>9} {'|pi0|mm':>9} {'|demo|mm':>9} {'grip_pi0':>9} {'grip_demo':>10}  note")
    for fi in range(args.start, min(args.end, n_frames - H - 1), args.stride):
        frame = real.frames[fi]
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
        chunk = np.asarray(policy.client.predict(scene).action_chunk, dtype=np.float32)

        # Policy's committed EE displacement over H steps (FK of integrated chunk).
        pj, _, _ = integrate_joint_chunk(_joints(frame), chunk[:H], kin, dt=dt)
        pi0_disp = ee(pj[-1]) - ee(pj[0])
        # Demo's actual EE displacement over the same H real frames.
        demo_disp = ee(_joints(real.frames[fi + H])) - ee(_joints(real.frames[fi]))

        c = _cos(pi0_disp, demo_disp)
        cosines.append(c)
        gp = float(chunk[:H, 7].mean())                                   # mean commanded gripper
        gd = float(np.mean([real.frames[fi + k].observations.gripper.value for k in range(1, H + 1)]))
        grip_rows.append((fi, gp, gd))
        note = "WRONG WAY" if (c == c and c < 0.0) else ("off" if (c == c and c < 0.5) else "")
        print(f"{fi:>5} {c:>9.3f} {np.linalg.norm(pi0_disp)*1000:>9.1f} "
              f"{np.linalg.norm(demo_disp)*1000:>9.1f} {gp:>9.3f} {gd:>10.3f}  {note}")

    cset = [c for c in cosines if c == c]
    if cset:
        print(f"\n[vs-expert] mean EE-direction agreement: {np.mean(cset):+.3f}  "
              f"(frac frames heading right way, cos>0.5: {np.mean([c > 0.5 for c in cset]):.2f}; "
              f"frac wrong-way, cos<0: {np.mean([c < 0.0 for c in cset]):.2f})")
        print(f"[vs-expert] VERDICT: {cfg_name!r} "
              f"{'TRACKS the expert' if np.mean(cset) > 0.5 else 'does NOT track the expert (fails the task)'}")


if __name__ == "__main__":
    main()
