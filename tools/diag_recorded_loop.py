"""Diagnostic: the closed loop driven by RECORDED ground-truth actions.

Isolates the re-feed loop mechanics from the policy. Runs the exact autoregressive
forward-dynamics loop the dream uses — condition on a frame, dream a chunk, re-feed
the last generated frame — but feeds Cosmos the *recorded* DROID action deltas for
each successive window instead of a policy's. This mirrors NVIDIA's robotics
autoregressive cookbook (run_fd_with_vllm.ipynb), which chains recorded 16-action
chunks and stays coherent.

If this stays coherent across chunks, the loop + re-feed + Cosmos are sound and the
dream's collapse is the policy (π0 dead-reckoning desync). If this *also* collapses,
the bug is in our re-feed mechanics, independent of the policy.

    uv run python tools/diag_recorded_loop.py --config configs/droid_pi0.yaml \
        --episode 312 --steps 6 --out outputs/recorded_loop_probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from emboviz.adapters import connect_world_model
from emboviz.config import load_run_config
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
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--n-actions", type=int, default=None, help="override chunk size (default cs.n_actions)")
    p.add_argument("--wrist-h", type=int, default=None, help="override wrist height (concat = 1.5*wrist_h); use 352 for a clean 528 concat")
    p.add_argument("--out", default="outputs/recorded_loop_probe")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    episode = args.episode if args.episode is not None else int(str(cfg.analysis.episodes).split(",")[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from emboviz_wire.observations import RGBImage
    from emboviz_wire.types import Observations, Scene
    from emboviz_cosmos3 import domains
    from emboviz_cosmos3.concat_view import build_concat_view

    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    frame0 = real.frames[0]
    print(f"[rec-loop] episode {episode}; instruction={frame0.instruction!r}; {len(real.frames)} frames")

    # Same world-model wiring as the dream path: DROID fps + guardrails off.
    wm = connect_world_model("cosmos3", world_model_kwargs={
        "server_url": cs.server_url, "domain_name": cs.domain,
        "action_dim": cs.action_dim, "conditioning_camera": cs.conditioning_camera,
        "fps": int(round(cs.control_hz)), "guardrails": False,
    })

    n = args.n_actions if args.n_actions is not None else cs.n_actions
    commit = n  # commit the whole chunk, re-feed last frame (NVIDIA autoregressive recipe)
    wrist_size = (args.wrist_h, cs.concat_resolution[1]) if args.wrist_h is not None else cs.concat_resolution
    seed = build_concat_view(
        _img(frame0, cs.concat_cameras["wrist"]),
        _img(frame0, cs.concat_cameras["exterior_left"]),
        _img(frame0, cs.concat_cameras["exterior_right"]),
        wrist_size=wrist_size,
    )
    Image.fromarray(seed, "RGB").save(out / "seed.png")
    print(f"[rec-loop] seed concat {seed.shape}; n_actions={n} commit={commit}")

    img = seed
    for step in range(args.steps):
        frame_start = step * commit
        if frame_start + n + 1 > len(real.frames):
            print(f"[rec-loop] step {step}: not enough recorded frames; stopping.")
            break
        # Recorded GT action deltas for this window — the exact encoder the dream uses.
        acts = domains.prepare_actions(cs.domain, real, frame_start=frame_start, n_actions=n)
        scene = Scene(
            observations=Observations(
                images={cs.conditioning_camera: RGBImage(data=img, camera_id=cs.conditioning_camera)}),
            instruction=frame0.instruction,
        )
        traj = wm.rollout(scene, np.asarray(acts, np.float32))
        frames = frames_to_arrays(traj, cs.conditioning_camera)
        img = np.asarray(frames[commit - 1], np.uint8)
        Image.fromarray(img, "RGB").save(out / f"rec_loop_step_{step:02d}.png")
        print(f"[rec-loop]   step {step} (frames {frame_start}..{frame_start+n}): "
              f"cond_sat={_sat(acts):.3f}  out {img.shape}", flush=True)

    print(f"[rec-loop] DONE -> {out}/")


if __name__ == "__main__":
    main()
