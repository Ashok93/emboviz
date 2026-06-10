"""Resolution probe — render one closed-loop dream clip at a chosen resolution.

Cosmos generates at the conditioning image's pixel size (rounded to a multiple of
16). The default DROID concat is ~270 px tall, so the world model dreams at ~256
px and small objects (a pen in a cup) smear. This runs the exact closed-loop path
for the first keyframe but upscales the seed concat (and therefore the generated
frames, which feed the next turn) to a target height, so we can compare fidelity
at 256 vs 480 on the same scene without touching the main pipeline.

    uv run python tools/highres_probe.py --config configs/cosmos_droid_pi0_demo.yaml \
        --episode 312 --height 480 --out outputs/highres_probe_480
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from emboviz.adapters import connect, connect_world_model
from emboviz.config import _JOINT_ACTION_CONVENTIONS, load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import detect_keyframes
from emboviz.world_models.simulate import closed_loop_rollout
from emboviz.world_models.viz import frames_to_arrays, save_video


def _img(frame, role: str) -> np.ndarray:
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def _upscale(concat: np.ndarray, target_h: int) -> np.ndarray:
    """Upscale a concat to ``target_h`` (rounded to /16), preserving aspect."""
    h, w = concat.shape[:2]
    new_h = max(16, round(target_h / 16) * 16)
    new_w = max(16, round(w * new_h / h / 16) * 16)
    out = Image.fromarray(concat, "RGB").resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--height", type=int, default=480, help="target conditioning height (px)")
    p.add_argument("--out", default="outputs/highres_probe")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.stress
    episode = args.episode if args.episode is not None else int(
        str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[probe] loading episode {episode} ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    keyframes = detect_keyframes(real)

    wm = connect_world_model("cosmos3", world_model_kwargs={
        "server_url": cs.server_url, "domain_name": cs.domain,
        "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
    })

    from emboviz_wire.policy_bridge import make_state_tracker
    from emboviz_cosmos3.concat_view import build_concat_view
    from emboviz_cosmos3.dream_step import PolicyDreamStepper

    kinematics = None
    if cs.action_convention in _JOINT_ACTION_CONVENTIONS:
        from emboviz_robot import load_kinematics
        kinematics = load_kinematics(cs.robot)

    print(f"[probe] connecting policy '{cs.policy_adapter}' ...")
    policy = connect(cs.policy_adapter, actor_kwargs=cs.policy_kwargs or None)

    lead = int(round(cs.lead_s * real.fps))
    kf = keyframes[0]
    seed_index = max(0, kf.index - lead)
    frame = real.frames[seed_index]

    seed = build_concat_view(
        _img(frame, cs.concat_cameras["wrist"]),
        _img(frame, cs.concat_cameras["exterior_left"]),
        _img(frame, cs.concat_cameras["exterior_right"]),
    )
    seed_hi = _upscale(seed, args.height)
    print(f"[probe] seed {seed.shape} -> upscaled {seed_hi.shape}")

    tracker = make_state_tracker(
        np.asarray(frame.observations.state.values, dtype=np.float32),
        float(frame.observations.gripper.value),
        action_convention=cs.action_convention,
        state_convention=cs.state_convention,
        kinematics=kinematics,
    )
    stepper = PolicyDreamStepper(
        policy.client.predict, tracker=tracker, camera_map=cs.camera_map,
        instruction=frame.instruction, n_actions=cs.n_actions,
    )

    print(f"[probe] dreaming at height {args.height} ...")
    dream = closed_loop_rollout(
        wm, seed_hi, stepper, n_steps=cs.n_loop_steps,
        conditioning_camera=cs.conditioning_camera, instruction=frame.instruction,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(seed_hi, "RGB").save(out / "seed_hi.png")
    arrs = frames_to_arrays(dream.trajectory, cs.conditioning_camera)
    save_video(arrs, out / "dream.mp4", fps=real.fps)
    Image.fromarray(arrs[0]).save(out / "dream_first.png")
    Image.fromarray(arrs[-1]).save(out / "dream_last.png")
    print(f"[probe] DONE: {len(arrs)} frames at {arrs[0].shape} -> {out}/")


if __name__ == "__main__":
    main()
