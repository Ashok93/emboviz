"""Cosmos 3 forward-dynamics smoke test — run on the pod beside a vLLM-Omni server.

Staged proof that the pipe works before paying for a full rollout:

  * ``--raw``  : one direct ``POST /v1/videos/sync`` (mirrors NVIDIA's reference
                 client). Proves the server + Cosmos, independent of emboviz.
  * (default)  : the same tiny rollout through ``Cosmos3WorldModel.rollout()``.
                 Proves the emboviz adapter end-to-end against the real server.

Keep it tiny: a handful of actions -> a handful of frames. The point is "frames
come back and decode", not generation quality.

Conditioning frame / actions:
  * ``--first-frame PATH`` — PNG/JPEG to condition on; a synthetic frame is
    generated when omitted (enough to prove the pipe).
  * ``--actions PATH`` — JSON ``{"action_chunks": [[...]], ...}`` (NVIDIA's
    asset shape) or a bare ``(T, action_dim)`` list; zeros are used when
    omitted (a no-motion rollout still exercises the full request/decode path).
  * ``--n-actions N`` — truncate to the first N actions (default 16, the trained
    ``action_chunk_size``; smaller chunks collapse to fewer frames because the
    video tokenizer is temporally compressed — a 2-action chunk yields 1 frame).

Examples::

    # Server-only smoke (no emboviz), synthetic frame + zero actions:
    python smoke_rollout.py --raw --domain agibotworld --action-dim 29

    # Adapter smoke through Cosmos3WorldModel, NVIDIA's AgiBot assets:
    python smoke_rollout.py --domain agibotworld --action-dim 29 \
        --first-frame example_action_fd_agibotworld_first_frame.png \
        --actions example_action_fd_agibotworld_action_chunks.json --n-actions 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _load_actions(path: str | None, action_dim: int, n: int) -> np.ndarray:
    """Return ``(n, action_dim)`` actions from a JSON file or zeros."""
    if path is None:
        return np.zeros((n, action_dim), dtype=np.float32)
    spec = json.loads(Path(path).read_text())
    if isinstance(spec, dict) and "action_chunks" in spec:
        flat = [row for chunk in spec["action_chunks"] for row in chunk]
    else:
        flat = spec  # a bare (T, action_dim) list
    actions = np.asarray(flat, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise SystemExit(
            f"actions from {path} have shape {actions.shape}; expected "
            f"(T, {action_dim})."
        )
    return actions[:n]


def _load_frame(path: str | None) -> np.ndarray:
    """Return an ``(H, W, 3)`` uint8 conditioning frame from a file or synthetic."""
    if path is None:
        h, w = 480, 480
        yy, xx = np.mgrid[0:h, 0:w]
        frame = np.stack(
            [(xx * 255 // w), (yy * 255 // h), np.full((h, w), 128)], axis=-1
        )
        return frame.astype(np.uint8)
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_frames(frames, out_dir: Path) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
            out_dir / f"frame_{i:03d}.png"
        )
    print(f"[smoke] saved {len(frames)} frame(s) to {out_dir}/")


def _raw(args, frame: np.ndarray, actions: np.ndarray) -> None:
    """Direct POST, mirroring the model card's reference client."""
    import io

    import imageio.v3 as iio
    import requests
    from PIL import Image

    h, w = frame.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="PNG")

    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": args.domain,
        "action_chunk_size": int(actions.shape[0]),
        "action": actions.tolist(),
        "image_size": args.image_size,
        "view_point": args.view_point,
        "guardrails": True,
    }
    data = {
        "prompt": args.prompt,
        "num_frames": str(int(actions.shape[0]) + 1),
        "fps": str(args.fps),
        "size": f"{w}x{h}",
        "num_inference_steps": str(args.steps),
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(extra_params),
    }
    url = f"{args.server_url.rstrip('/')}/v1/videos/sync"
    print(f"[smoke:raw] POST {url}  ({actions.shape[0]} actions, {args.steps} steps)")
    t0 = time.time()
    resp = requests.post(
        url, data=data,
        files={"input_reference": ("frame.png", buf.getvalue(), "image/png")},
        headers={"Accept": "video/mp4"}, timeout=600,
    )
    resp.raise_for_status()
    print(f"[smoke:raw] {len(resp.content)} bytes in {time.time() - t0:.1f}s")

    tmp = Path("/tmp/cosmos3_smoke_raw.mp4")
    tmp.write_bytes(resp.content)
    frames = np.asarray(iio.imread(tmp, plugin="pyav"))
    print(f"[smoke:raw] decoded {frames.shape[0]} frames of {frames.shape[1:]} ")
    _save_frames(frames[1:], Path(args.out))  # drop the conditioning frame


def _adapter(args, frame: np.ndarray, actions: np.ndarray) -> None:
    """The same rollout through Cosmos3WorldModel — the emboviz adapter path."""
    from emboviz_wire.observations import RGBImage
    from emboviz_wire.types import Observations, Scene

    from emboviz_cosmos3.model import Cosmos3WorldModel

    cam = "primary"
    scene = Scene(
        observations=Observations(images={cam: RGBImage(data=frame, camera_id=cam)}),
        instruction=args.prompt or None,
    )
    model = Cosmos3WorldModel(
        server_url=args.server_url,
        domain_name=args.domain,
        action_dim=args.action_dim,
        conditioning_camera=cam,
        action_chunk_size=max(1, int(actions.shape[0])),  # one request for the smoke
        num_inference_steps=args.steps,
        fps=args.fps,
        image_size=args.image_size,
        view_point=args.view_point,
        default_prompt=args.prompt,
    )
    print(f"[smoke:adapter] rollout: {actions.shape[0]} actions, {args.steps} steps")
    t0 = time.time()
    traj = model.rollout(scene, actions)
    print(f"[smoke:adapter] {len(traj.frames)} generated frames in {time.time() - t0:.1f}s")
    print(f"[smoke:adapter] metadata: {traj.metadata}")
    _save_frames(
        [s.observations.images["primary"].data for s in traj.frames], Path(args.out)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument("--domain", required=True, help="Cosmos domain_name, e.g. agibotworld")
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--first-frame", default=None)
    p.add_argument("--actions", default=None)
    p.add_argument("--n-actions", type=int, default=16)
    p.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image-size", type=int, default=480)
    p.add_argument("--view-point", default="concat_view")
    p.add_argument("--prompt", default="robot manipulation")
    p.add_argument("--out", default="/tmp/cosmos3_smoke_frames")
    p.add_argument("--raw", action="store_true", help="direct POST, bypass the adapter")
    args = p.parse_args()

    frame = _load_frame(args.first_frame)
    actions = _load_actions(args.actions, args.action_dim, args.n_actions)
    print(f"[smoke] frame {frame.shape}, actions {actions.shape}, domain={args.domain}")
    (_raw if args.raw else _adapter)(args, frame, actions)
    print("[smoke] OK")


if __name__ == "__main__":
    main()
