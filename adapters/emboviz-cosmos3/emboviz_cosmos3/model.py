"""Cosmos 3 world-model adapter (worker side).

Drives NVIDIA Cosmos3-Nano's **action-conditioned forward dynamics** — given a
conditioning frame and a sequence of robot actions, generate the future video
those actions would produce — and presents it behind emboviz's
:class:`emboviz_wire.world_model_protocol.WorldModel` contract.

Backend
-------
Action conditioning is exposed **only** by NVIDIA's vLLM-Omni server (the
``vllm/vllm-omni:cosmos3`` image / ``vllm serve nvidia/Cosmos3-Nano --omni``).
The diffusers ``Cosmos3OmniPipeline`` has no action parameters, and vLLM-Omni
serves **BF16 only** (FP8 / NVFP4 are not supported for the action path). This
adapter therefore carries no torch and holds no GPU: it is a thin HTTP client
that POSTs to a running vLLM-Omni server and decodes the returned MP4. The
heavy model lives in the server process (its own GPU box / container); the
adapter owns only the translation between emboviz types and the HTTP API.

Forward-dynamics HTTP contract (``POST /v1/videos/sync``), per the
Cosmos3-Nano model card's reference client:

  * multipart ``input_reference`` — the conditioning image (one frame).
  * form ``extra_params`` (JSON) — ``action_mode="forward_dynamics"``,
    ``domain_name``, ``action_chunk_size``, ``action`` (the chunk), plus
    optional ``image_size`` / ``view_point`` / ``guardrails``.
  * form ``num_frames = action_chunk_size + 1`` — the server returns the
    conditioning frame followed by the generated frames; the conditioning
    frame is dropped here so the rollout contains generated frames only.

Long rollouts are autoregressive by chunk: the final generated frame of one
chunk conditions the next request. This is the server's native multi-chunk
protocol and the structure emboviz's trust diagnostics build on.

Action normalization
---------------------
Cosmos conditions on actions in its own per-domain normalized space. This
adapter applies an explicitly supplied ``action_normalizer`` callable; with the
default (``None``) it passes actions through unchanged, i.e. the caller must
supply actions already in the domain's normalized convention. There is no
inferred or guessed normalization — an unknown mapping would silently corrupt
the conditioning, so the convention is a declared constructor contract.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Callable, Optional

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability


log = logging.getLogger("emboviz_cosmos3")

#: Action dimensionality per embodiment, from the Cosmos3-Nano model card's
#: supported-embodiment table. Reference data for the loud constructor check
#: below — NOT a default: the API ``domain_name`` strings (e.g.
#: ``"agibotworld"``, ``"bridge_orig_lerobot"``) are configured explicitly by
#: the caller together with the matching ``action_dim``.
DOCUMENTED_ACTION_DIMS = {
    "general_camera": 9,
    "av": 9,
    "egocentric": 57,
    "single_franka": 10,
    "dual_franka": 20,
    "agibot": 29,
    "ur": 10,
    "google_robot": 10,
    "widowx_250": 10,
    "umi": 9,
}


class Cosmos3WorldModel(WorldModel):
    """Forward-dynamics world model backed by a vLLM-Omni Cosmos 3 server.

    Parameters
    ----------
    server_url
        Base URL of the running vLLM-Omni server (e.g.
        ``"http://localhost:8000"``). The adapter POSTs to
        ``{server_url}/v1/videos/sync``.
    domain_name
        The Cosmos embodiment domain string passed in ``extra_params``
        (e.g. ``"agibotworld"``, ``"bridge_orig_lerobot"``). Required — the
        adapter never guesses an embodiment.
    action_dim
        Dimensionality of one action row, matching ``domain_name``. Required
        and validated against every rollout's actions. See
        :data:`DOCUMENTED_ACTION_DIMS` for the documented per-embodiment values.
    conditioning_camera
        Scene camera role whose image conditions generation. Default
        ``"primary"``.
    action_chunk_size
        Actions per server request. The rollout is split into chunks of this
        size and generated autoregressively. Default 16 (the card's value).
    num_inference_steps, guidance_scale, flow_shift, fps, seed
        Generation settings forwarded to the server, matching the reference
        client's defaults for forward dynamics.
    image_size, view_point
        Optional ``extra_params`` fields; omitted from the request when None.
    guardrails
        Whether the server runs its content guardrail. Default True.
    default_prompt
        Prompt used when a Scene carries no instruction. Forward dynamics
        still requires a text prompt; a recorded episode's instruction is
        used when present, else this.
    request_timeout
        Per-request HTTP timeout in seconds.
    action_normalizer
        Optional ``(T, action_dim) -> (T, action_dim)`` callable mapping native
        actions into the domain's normalized space. ``None`` passes actions
        through unchanged (caller supplies pre-normalized actions).
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        *,
        domain_name: str,
        action_dim: int,
        conditioning_camera: str = "primary",
        action_chunk_size: int = 16,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        flow_shift: float = 10.0,
        fps: int = 10,
        seed: int = 0,
        image_size: Optional[int] = 480,
        view_point: Optional[str] = "concat_view",
        guardrails: bool = True,
        default_prompt: str = "",
        request_timeout: float = 600.0,
        action_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if not server_url:
            raise ValueError("Cosmos3WorldModel: server_url is required.")
        if not domain_name:
            raise ValueError("Cosmos3WorldModel: domain_name is required.")
        if int(action_dim) < 1:
            raise ValueError(
                f"Cosmos3WorldModel: action_dim must be >= 1, got {action_dim}."
            )
        if int(action_chunk_size) < 1:
            raise ValueError(
                f"Cosmos3WorldModel: action_chunk_size must be >= 1, got "
                f"{action_chunk_size}."
            )

        self._server_url = server_url.rstrip("/")
        self._endpoint = f"{self._server_url}/v1/videos/sync"
        self._domain_name = str(domain_name)
        self._action_dim = int(action_dim)
        self._conditioning_camera = str(conditioning_camera)
        self._action_chunk_size = int(action_chunk_size)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._flow_shift = float(flow_shift)
        self._fps = int(fps)
        self._seed = int(seed)
        self._image_size = None if image_size is None else int(image_size)
        self._view_point = view_point
        self._guardrails = bool(guardrails)
        self._default_prompt = str(default_prompt)
        self._request_timeout = float(request_timeout)
        self._action_normalizer = action_normalizer

    # ----- WorldModel ABC: identification -----------------------------------

    @property
    def model_id(self) -> str:
        return "cosmos3-nano"

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability.FORWARD_DYNAMICS

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def supported_domains(self) -> frozenset[str]:
        return frozenset({self._domain_name})

    @property
    def conditioning_camera(self) -> str:
        return self._conditioning_camera

    # ----- WorldModel ABC: forward dynamics ---------------------------------

    def rollout(
        self,
        init: Scene,
        actions: np.ndarray,
        *,
        num_frames: Optional[int] = None,
    ) -> Trajectory:
        actions = np.asarray(actions, dtype=np.float32)
        reason = self.validate_rollout(init, actions)
        if reason is not None:
            raise ValueError(f"cosmos3 rollout request rejected: {reason}")

        if num_frames is not None:
            if int(num_frames) < 1:
                raise ValueError(
                    f"cosmos3 rollout: num_frames must be >= 1, got {num_frames}."
                )
            actions = actions[: int(num_frames)]

        actions = self._normalize_actions(actions)
        cond_image = self._conditioning_image(init)
        prompt = (init.instruction or self._default_prompt or "").strip()

        generated: list[np.ndarray] = []
        current = cond_image
        chunks = self._split_into_chunks(actions)
        for chunk_idx, chunk in enumerate(chunks):
            frames = self._request_chunk(current, chunk, prompt)
            if frames.shape[0] < 2:
                raise RuntimeError(
                    f"cosmos3 forward dynamics returned {frames.shape[0]} "
                    f"frame(s) for chunk {chunk_idx} (expected the "
                    f"conditioning frame + {len(chunk)} generated frames). "
                    "The server response is malformed."
                )
            # The first returned frame is the conditioning frame; drop it so
            # the rollout holds generated frames only.
            generated.extend(frames[1:])
            current = frames[-1]

        return self._build_trajectory(generated, prompt, n_chunks=len(chunks))

    # ----- episode → conditioning actions ----------------------------------

    def prepare_actions(
        self,
        episode: Trajectory,
        *,
        frame_start: int = 0,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Encode the episode into this domain's normalized action representation.

        Overrides the default (raw logged actions): Cosmos conditions each domain
        on a specific encoding (e.g. DROID's normalized pose deltas), implemented
        in :mod:`emboviz_cosmos3.domains`. The result feeds :meth:`rollout`.
        """
        from emboviz_cosmos3 import domains

        expected_dim = domains.ACTION_DIMS.get(self._domain_name)
        if expected_dim is not None and expected_dim != self._action_dim:
            raise ValueError(
                f"cosmos3 domain '{self._domain_name}' encodes {expected_dim}-D actions, "
                f"but the adapter was configured with action_dim={self._action_dim}. "
                "Set action_dim to match the domain."
            )

        if n_actions is None:
            # One relative delta per consecutive frame pair from frame_start.
            n_actions = len(episode.frames) - frame_start - 1
            if n_actions < 1:
                raise ValueError(
                    f"episode has too few frames ({len(episode.frames)}) to encode "
                    f"any action from frame_start={frame_start}"
                )
        return domains.prepare_actions(
            self._domain_name, episode, frame_start=frame_start, n_actions=int(n_actions)
        )

    # ----- request / response ----------------------------------------------

    def _request_chunk(
        self, cond_image: np.ndarray, chunk: np.ndarray, prompt: str,
    ) -> np.ndarray:
        """POST one action chunk and return the decoded frames ``(k+1, H, W, 3)``."""
        import requests

        height, width = int(cond_image.shape[0]), int(cond_image.shape[1])
        extra_params = {
            "action_mode": "forward_dynamics",
            "domain_name": self._domain_name,
            "action_chunk_size": int(chunk.shape[0]),
            "action": chunk.tolist(),
            "guardrails": self._guardrails,
        }
        if self._image_size is not None:
            extra_params["image_size"] = self._image_size
        if self._view_point is not None:
            extra_params["view_point"] = self._view_point

        data = {
            "prompt": prompt,
            "num_frames": str(int(chunk.shape[0]) + 1),
            "fps": str(self._fps),
            "size": f"{width}x{height}",
            "num_inference_steps": str(self._num_inference_steps),
            "guidance_scale": str(self._guidance_scale),
            "flow_shift": str(self._flow_shift),
            "seed": str(self._seed),
            "extra_params": json.dumps(extra_params),
        }
        files = {"input_reference": ("frame.png", self._encode_png(cond_image), "image/png")}

        response = requests.post(
            self._endpoint,
            data=data,
            files=files,
            headers={"Accept": "video/mp4"},
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        return self._decode_mp4(response.content)

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB").save(
            buf, format="PNG"
        )
        return buf.getvalue()

    @staticmethod
    def _decode_mp4(content: bytes) -> np.ndarray:
        """Decode MP4 bytes to a ``(T, H, W, 3)`` uint8 array.

        Written to a temp file because the video plugins need a seekable
        source — the same path the reference client uses.
        """
        import tempfile
        from pathlib import Path

        import imageio.v3 as iio

        if not content:
            raise RuntimeError("cosmos3: server returned an empty response body.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk.mp4"
            path.write_bytes(content)
            frames = np.asarray(iio.imread(path, plugin="pyav"))
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeError(
                f"cosmos3: decoded video has shape {frames.shape}, expected "
                "(T, H, W, 3)."
            )
        return frames.astype(np.uint8, copy=False)

    # ----- helpers ----------------------------------------------------------

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self._action_normalizer is None:
            return actions
        out = np.asarray(self._action_normalizer(actions), dtype=np.float32)
        if out.shape != actions.shape:
            raise ValueError(
                f"cosmos3 action_normalizer changed the action shape "
                f"{actions.shape} -> {out.shape}; it must be element-wise."
            )
        return out

    def _conditioning_image(self, init: Scene) -> np.ndarray:
        rgb = init.observations.images[self._conditioning_camera]
        arr = np.asarray(rgb.data)
        if arr.dtype != np.uint8:
            raise ValueError(
                f"cosmos3 conditioning image for camera "
                f"'{self._conditioning_camera}' has dtype {arr.dtype}; the "
                "conditioning frame must be uint8 RGB (the reader decodes to "
                "uint8)."
            )
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"cosmos3 conditioning image has shape {arr.shape}, expected "
                "(H, W, 3) RGB."
            )
        return arr

    def _split_into_chunks(self, actions: np.ndarray) -> list[np.ndarray]:
        n = actions.shape[0]
        size = self._action_chunk_size
        return [actions[i : i + size] for i in range(0, n, size)]

    def _build_trajectory(
        self, frames: list[np.ndarray], prompt: str, *, n_chunks: int,
    ) -> Trajectory:
        cam = self._conditioning_camera
        scenes = [
            Scene(
                observations=Observations(
                    images={cam: RGBImage(data=frame, camera_id=cam)}
                ),
                instruction=prompt or None,
                metadata={"source": "cosmos3-forward-dynamics"},
            )
            for frame in frames
        ]
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=float(self._fps),
            episode_id="cosmos3-rollout",
            source=f"cosmos3:{self._domain_name}",
            metadata={
                "world_model": "cosmos3-nano",
                "domain_name": self._domain_name,
                "action_dim": self._action_dim,
                "action_chunk_size": self._action_chunk_size,
                "n_chunks": n_chunks,
                "num_inference_steps": self._num_inference_steps,
                "guidance_scale": self._guidance_scale,
                "flow_shift": self._flow_shift,
                "seed": self._seed,
                "server_url": self._server_url,
            },
        )
