"""Diagnostic: dump the RAW output of a single Cosmos forward-dynamics call.

Isolates the world-model call from the closed loop and the policy. From the real
seed frame it makes ONE rollout call with (a) π0's step-0 conditioning and (b) the
recorded-action conditioning, and saves every returned frame as PNG plus the seed.
If Cosmos returns garbage from a good seed in a single call, the problem is the
world-model request itself (image / size / num_frames / prompt) — not the policy,
not the loop.

    uv run python tools/diag_cosmos_call.py --config configs/droid_pi0.yaml \
        --episode 312 --seed-index 0 --out outputs/cosmos_call_probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from emboviz.adapters import connect, connect_world_model
from emboviz.config import _JOINT_ACTION_CONVENTIONS, load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.viz import frames_to_arrays


def _img(frame, role):
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def _sat(a):
    return float(np.mean(np.abs(np.asarray(a)) >= 0.999))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--out", default="outputs/cosmos_call_probe")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    episode = args.episode if args.episode is not None else int(str(cfg.analysis.episodes).split(",")[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from emboviz_wire.observations import RGBImage
    from emboviz_wire.types import Observations, Scene
    from emboviz_cosmos3 import domains
    from emboviz_cosmos3.bridge import make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view
    from emboviz_cosmos3.dream_step import PolicyDreamStepper
    from emboviz_robot import load_kinematics

    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    si = args.seed_index
    frame = real.frames[si]
    print(f"[probe] episode {episode} seed {si}; instruction={frame.instruction!r}")

    wm = connect_world_model("cosmos3", world_model_kwargs={
        "server_url": cs.server_url, "domain_name": cs.domain,
        "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
    })

    seed = build_concat_view(
        _img(frame, cs.concat_cameras["wrist"]),
        _img(frame, cs.concat_cameras["exterior_left"]),
        _img(frame, cs.concat_cameras["exterior_right"]),
        wrist_size=cs.concat_resolution,
    )
    print(f"[probe] seed concat {seed.shape}")
    Image.fromarray(seed, "RGB").save(out / "seed.png")

    n = cs.n_actions
    scene = Scene(
        observations=Observations(
            images={cs.conditioning_camera: RGBImage(data=seed, camera_id=cs.conditioning_camera)}),
        instruction=frame.instruction,
    )

    # (a) π0 step-0 conditioning.
    kin = load_kinematics(cs.robot) if cs.action_convention in _JOINT_ACTION_CONVENTIONS else None
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)  # connect ONCE, reuse
    stepper = PolicyDreamStepper(
        policy.client.predict,
        tracker=make_state_tracker(
            np.asarray(frame.observations.state.values, np.float32),
            float(frame.observations.gripper.value),
            action_convention=cs.action_convention, state_convention=cs.state_convention,
            kinematics=kin, control_hz=cs.control_hz,
        ),
        camera_map=cs.camera_map, instruction=frame.instruction,
        n_actions=n, execute_steps=cs.execute_steps,
    )
    pi0_actions = stepper(seed)
    print(f"[probe] pi0 conditioning {pi0_actions.shape} sat={_sat(pi0_actions):.2f}")

    # (b) recorded-action conditioning at the same seed.
    rec_actions = domains.prepare_actions(cs.domain, real, frame_start=si, n_actions=n)
    print(f"[probe] recorded conditioning {rec_actions.shape} sat={_sat(rec_actions):.2f}")

    for tag, actions in (("pi0", pi0_actions), ("recorded", rec_actions)):
        print(f"[probe] rollout with {tag} actions ...", flush=True)
        traj = wm.rollout(scene, np.asarray(actions, np.float32))
        arrs = frames_to_arrays(traj, cs.conditioning_camera)
        for i, a in enumerate(arrs):
            Image.fromarray(np.asarray(a, np.uint8), "RGB").save(out / f"{tag}_frame_{i:02d}.png")
        print(f"[probe]   {tag}: {len(arrs)} frames saved (shape {arrs[0].shape})")

    # ---- multi-step closed loop: does re-conditioning push saturation up? ----
    print("[probe] --- closed loop (pi0, re-conditioning each step) ---", flush=True)
    loop_tracker = make_state_tracker(
        np.asarray(frame.observations.state.values, np.float32),
        float(frame.observations.gripper.value),
        action_convention=cs.action_convention, state_convention=cs.state_convention,
        kinematics=kin, control_hz=cs.control_hz,
    )
    loop_stepper = PolicyDreamStepper(
        policy.client.predict,
        tracker=loop_tracker, camera_map=cs.camera_map, instruction=frame.instruction,
        n_actions=n, execute_steps=cs.execute_steps,
    )
    img = seed
    for step in range(6):
        acts = loop_stepper(img)
        sc = Scene(observations=Observations(
            images={cs.conditioning_camera: RGBImage(data=img, camera_id=cs.conditioning_camera)}),
            instruction=frame.instruction)
        tr = wm.rollout(sc, np.asarray(acts, np.float32))
        fr = frames_to_arrays(tr, cs.conditioning_camera)
        commit = cs.execute_steps or n
        img = np.asarray(fr[commit - 1], np.uint8)
        Image.fromarray(img, "RGB").save(out / f"loop_step_{step:02d}.png")
        print(f"[probe]   loop step {step}: cond_sat={_sat(acts):.3f}  out {img.shape}", flush=True)

    print(f"[probe] DONE -> {out}/")


if __name__ == "__main__":
    main()
