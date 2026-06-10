"""Ctrl-World world-model adapter (worker side).

Drives the Ctrl-World action-conditioned video world model (Guo et al., ICLR
2026, arXiv:2510.10125) — a Stable-Video-Diffusion fine-tune that predicts a
robot's camera views **jointly** and anchors each prediction to a
**pose-conditioned sparse history**, which is what keeps its closed-loop
rollouts coherent over tens of seconds where single-frame conditioning drifts
within one. Presented behind
:class:`emboviz_wire.world_model_protocol.WorldModel`.

Everything checkpoint-specific — which views are stacked, at which size and
rate, the history schedule, the action-normalization bounds, where the weights
live — comes from a :class:`emboviz_ctrlworld.profiles.CtrlWorldProfile`
(``"droid"`` ships; a fine-tune on another rig is a profile JSON, not a code
change). The conditioning *semantics* are fixed across profiles:

* **Frames** — one vertical stack of the profile's views, each VAE-encoded
  separately and stacked along the latent height (``extract_latent.py``;
  ``dataset_droid_exp33.py`` lines 183-186 in the vendored reference).
* **Actions** — absolute end-effector rows ``[x, y, z, roll, pitch, yaw,
  gripper]`` (extrinsic-XYZ euler, gripper in [0, 1]), min-max normalized to
  [-1, 1] with the profile's training-data percentile bounds
  (``dataset_droid_exp33.py`` lines 190-193). For the DROID profile this is
  the ``panda_link8`` flange pose — the convention the checkpoint trained on;
  the reference *rollout* script's TCP-frame FK deviates from it and is
  deliberately not reproduced.
* **Window** — one forward pass conditions on ``num_history`` sparse history
  latents + poses and predicts ``num_frames`` frames, of which frame 0
  re-renders the conditioning timestep; one chunk yields
  ``num_frames - 1`` future frames, and ``actions`` must come in multiples of
  that.
* **History selection** — the closed-loop driver passes the rollout's anchor
  frames (seed first, then each turn's committed conditioning frame); per
  chunk the adapter picks entries at the profile's ``history_idx`` relative to
  the end of that buffer, clamping out-of-range indices to the seed — exactly
  the reference buffer that is pre-filled with the seed
  (``rollout_interact_pi.py`` lines 336-341, 366-370).

The loop stays in **latent space**: every generated frame carries its latent
in ``Scene.metadata["ctrlworld_latent"]``, and conditioning prefers that
latent over re-encoding pixels, so the dream is never degraded by a
decode→re-encode round trip (the reference keeps ``his_cond`` as latents for
the same reason). A frame without the metadata key — the seed, or an edited
counterfactual seed — is encoded once from pixels, which is its defined entry
path, not a fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from emboviz_wire.observations import GripperState, Proprioception, RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability

from emboviz_ctrlworld.profiles import ACTION_DIM, CtrlWorldProfile, resolve_profile
from emboviz_ctrlworld.stack_view import split_stack_view

log = logging.getLogger("emboviz_ctrlworld")


@dataclass
class _WMArgs:
    """The argument namespace :class:`_ctrl_world.ctrl_world.CrtlWorld` reads."""

    svd_model_path: str
    clip_model_path: str
    action_dim: int
    num_history: int
    num_frames: int
    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False


def normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Min-max normalize to [-1, 1] — the reference ``normalize_bound``
    (``dataset_droid_exp33.py`` lines 105-115), reproduced exactly."""
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


def history_position(length: int, idx: int) -> int:
    """Map a ``history_idx`` entry to a position in a ``length``-long buffer.

    Reproduces the reference buffer that is pre-filled with the seed
    (``rollout_interact_pi.py`` lines 336-341): index 0 is always the seed;
    a negative index counts back from the end and clamps to the seed when it
    reaches past the rollout's start.
    """
    if length < 1:
        raise ValueError("history buffer is empty.")
    if idx == 0:
        return 0
    if idx > 0:
        raise ValueError(f"history_idx entries must be 0 or negative, got {idx}.")
    pos = length + idx
    return pos if pos >= 1 else 0


class CtrlWorldModel(WorldModel):
    """Forward-dynamics world model backed by a Ctrl-World checkpoint.

    Parameters
    ----------
    profile
        Checkpoint profile: a preconfigured name (``"droid"``) or the path to
        a profile JSON (:mod:`emboviz_ctrlworld.profiles`). Defines the
        checkpoint contract — views, sizes, rates, history schedule, bounds,
        and weight locations.
    conditioning_camera
        Scene camera role carrying the view stack. Default ``"primary"``.
    num_inference_steps
        Denoising steps per chunk. Default 50 (reference ``config.py``).
    guidance_scale
        Classifier-free guidance weight. Default 1.0 (reference value; 1.0
        disables CFG, halving UNet cost).
    decode_chunk_size
        Frames decoded per VAE call. Default 7 (reference value).
    dtype
        ``"bfloat16"`` (reference inference dtype) or ``"float32"``.
    device
        torch device string. The worker refuses to run without CUDA unless
        ``device`` explicitly names a non-CUDA device.
    seed
        Generation seed (VAE posterior sampling + diffusion noise). Fixed by
        default so a baseline-vs-counterfactual pair differs only in its
        conditioning.
    """

    def __init__(
        self,
        *,
        profile: str = "droid",
        conditioning_camera: str = "primary",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        decode_chunk_size: int = 7,
        dtype: str = "bfloat16",
        device: str = "cuda",
        seed: int = 0,
    ):
        import torch

        if int(num_inference_steps) < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}.")
        if dtype not in ("bfloat16", "float32"):
            raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {dtype!r}.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CtrlWorldModel: device='cuda' but CUDA is not available. The "
                "Ctrl-World worker runs the 1.5B SVD UNet locally and needs a GPU; "
                "pass an explicit non-CUDA device only for debugging."
            )

        self._p: CtrlWorldProfile = resolve_profile(profile)
        self._conditioning_camera = str(conditioning_camera)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._decode_chunk_size = int(decode_chunk_size)
        self._device = torch.device(device)
        self._dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self._seed = int(seed)

        self._state_p01 = np.asarray(self._p.state_p01, dtype=np.float64)[None, :]
        self._state_p99 = np.asarray(self._p.state_p99, dtype=np.float64)[None, :]

        self._model = self._load()

    def _load(self):
        import torch

        from emboviz_ctrlworld._ctrl_world.ctrl_world import CrtlWorld

        p = self._p
        svd_path = self._resolve_repo(p.svd_repo)
        clip_path = self._resolve_repo(p.clip_repo)
        ckpt_path = self._resolve_file(p.ckpt_repo, p.ckpt_file)

        log.info(
            "loading Ctrl-World profile '%s': svd=%s clip=%s ckpt=%s",
            p.name, svd_path, clip_path, ckpt_path,
        )
        model = CrtlWorld(_WMArgs(
            svd_model_path=svd_path,
            clip_model_path=clip_path,
            action_dim=ACTION_DIM,
            num_history=p.num_history,
            num_frames=p.num_frames,
        ))
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self._device).to(self._dtype)
        model.eval()
        log.info("Ctrl-World '%s' loaded on %s (%s)", p.name, self._device, self._dtype)
        return model

    @staticmethod
    def _resolve_repo(repo_or_path: str) -> str:
        if Path(repo_or_path).expanduser().is_dir():
            return str(Path(repo_or_path).expanduser())
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_or_path)

    @staticmethod
    def _resolve_file(repo_or_path: str, filename: str) -> str:
        local = Path(repo_or_path).expanduser()
        if local.is_dir():
            path = local / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"ckpt_repo {repo_or_path!r} is a local directory but "
                    f"{filename!r} is not in it."
                )
            return str(path)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_or_path, filename)

    # ----- WorldModel ABC: identification -----------------------------------

    @property
    def model_id(self) -> str:
        return f"ctrl-world-{self._p.name}"

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability.FORWARD_DYNAMICS

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def supported_domains(self) -> frozenset[str]:
        return frozenset({self._p.embodiment})

    @property
    def conditioning_camera(self) -> str:
        return self._conditioning_camera

    @property
    def conditions_on_history(self) -> bool:
        return True

    # ----- request validation -----------------------------------------------

    def validate_rollout(self, init: Scene, actions: np.ndarray) -> Optional[str]:
        base = super().validate_rollout(init, actions)
        if base is not None:
            return base
        actions = np.asarray(actions)
        p = self._p
        if actions.shape[0] % p.frames_per_chunk != 0:
            return (
                f"ctrl-world '{p.name}' generates {p.frames_per_chunk} future "
                f"frames per chunk (num_frames={p.num_frames} with frame 0 "
                "re-rendering the conditioning timestep), so actions must come "
                f"in multiples of {p.frames_per_chunk}; got {actions.shape[0]} rows."
            )
        reason = self._scene_problem(init, "init")
        if reason is not None:
            return reason
        if not (init.instruction or "").strip():
            return (
                "ctrl-world is text-conditioned (CLIP instruction embedding added "
                "to every action token); init.instruction must carry the task text."
            )
        return None

    def _scene_problem(self, scene: Scene, label: str) -> Optional[str]:
        cam = self._conditioning_camera
        if cam not in scene.observations.images:
            return f"{label} is missing camera '{cam}' (the view stack)."
        arr = np.asarray(scene.observations.images[cam].data)
        expected = (*self._p.stack_hw, 3)
        if arr.dtype != np.uint8 or arr.shape != expected:
            return (
                f"{label} camera '{cam}' must be the {expected} uint8 view stack "
                f"of profile '{self._p.name}' (build it with "
                "emboviz_ctrlworld.stack_view.build_stack_view); "
                f"got dtype={arr.dtype} shape={arr.shape}."
            )
        if scene.observations.state is None or len(np.asarray(scene.observations.state.values)) < 6:
            return (
                f"{label} needs a >=6-D end-effector pose [xyz, euler_xyz] in "
                "observations.state — ctrl-world anchors every frame to its pose."
            )
        convention = getattr(scene.observations.state, "convention", None)
        if convention != "ee_pose":
            return (
                f"{label}.observations.state.convention is {convention!r}; ctrl-world "
                "conditions on an absolute end-effector pose ('ee_pose', "
                "[xyz, euler_xyz]). A joint-space state must be forward-"
                "kinematicized by the driver — it is never reinterpreted here."
            )
        if scene.observations.gripper is None:
            return f"{label} needs observations.gripper (the pose's 7th dimension)."
        return None

    # ----- forward dynamics ---------------------------------------------------

    def rollout(
        self,
        init: Scene,
        actions: np.ndarray,
        *,
        history: Optional[Trajectory] = None,
        num_frames: Optional[int] = None,
    ) -> Trajectory:
        import torch

        p = self._p
        actions = np.asarray(actions, dtype=np.float64)
        if num_frames is not None:
            if int(num_frames) < 1:
                raise ValueError(f"ctrl-world rollout: num_frames must be >= 1, got {num_frames}.")
            actions = actions[: int(num_frames)]
        reason = self.validate_rollout(init, actions)
        if reason is not None:
            raise ValueError(f"ctrl-world rollout request rejected: {reason}")

        instruction = (init.instruction or "").strip()
        generator = torch.Generator(device=self._device).manual_seed(self._seed)

        # History buffer: (latent, pose (7,)) pairs, oldest first, entry 0 the
        # seed. ``history=None`` is the cold start — the buffer is just the
        # conditioning frame, matching the reference pre-fill of the whole
        # buffer with the first frame.
        current_latent, current_pose = self._scene_entry(init, "init", generator)
        if history is None:
            buffer = [(current_latent, current_pose)]
        else:
            buffer = [
                self._scene_entry(scene, f"history[{i}]", generator)
                for i, scene in enumerate(history.frames)
            ]
            if not buffer:
                raise ValueError(
                    "ctrl-world rollout: history was passed but holds no frames; "
                    "pass None for a cold start."
                )

        n_chunks = actions.shape[0] // p.frames_per_chunk
        out_frames: list[Scene] = []
        for chunk_idx in range(n_chunks):
            chunk = actions[chunk_idx * p.frames_per_chunk : (chunk_idx + 1) * p.frames_per_chunk]
            frames, last_latent = self._generate_chunk(
                current_latent, current_pose, chunk, buffer, instruction, generator
            )
            out_frames.extend(frames)
            current_latent, current_pose = last_latent, chunk[-1]
            buffer.append((last_latent, np.asarray(chunk[-1], dtype=np.float64)))

        return Trajectory(
            frames=out_frames,
            frame_indices=list(range(len(out_frames))),
            fps=p.native_fps,
            episode_id="ctrlworld-rollout",
            source=f"ctrlworld:{p.embodiment}",
            metadata={
                "world_model": self.model_id,
                "profile": p.name,
                "checkpoint": f"{p.ckpt_repo}/{p.ckpt_file}",
                "action_dim": ACTION_DIM,
                "frames_per_chunk": p.frames_per_chunk,
                "n_chunks": n_chunks,
                "num_inference_steps": self._num_inference_steps,
                "guidance_scale": self._guidance_scale,
                "history_idx": list(p.history_idx),
                "history_len": len(buffer) - n_chunks,
                "seed": self._seed,
            },
        )

    # ----- one chunk: history + current -> frames_per_chunk future frames ----

    def _generate_chunk(
        self,
        current_latent,
        current_pose: np.ndarray,
        chunk: np.ndarray,
        buffer: list,
        instruction: str,
        generator,
    ):
        import torch

        from emboviz_ctrlworld._ctrl_world.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        model = self._model
        p = self._p
        selected = [self._buffer_entry(buffer, idx) for idx in p.history_idx]

        # Action conditioning: one pose row per latent frame — the history
        # rows, the current pose, then the future poses (reference
        # rollout_interact_pi.py lines 366-369), normalized to [-1, 1].
        action_cond = np.stack(
            [pose for _, pose in selected] + [np.asarray(current_pose, dtype=np.float64)]
            + [chunk[i] for i in range(p.frames_per_chunk)]
        )
        action_cond = normalize_bound(action_cond, self._state_p01, self._state_p99)
        action_t = torch.tensor(action_cond).unsqueeze(0).to(self._device).to(self._dtype)

        his_latent = torch.cat([lat for lat, _ in selected], dim=0).unsqueeze(0)

        with torch.no_grad():
            text_token = model.action_encoder(
                action_t, instruction, model.tokenizer, model.text_encoder
            )
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                model.pipeline,
                image=current_latent,
                text=text_token,
                width=p.view_hw[1],
                height=p.stack_hw[0],
                num_frames=p.num_frames,
                history=his_latent,
                num_inference_steps=self._num_inference_steps,
                decode_chunk_size=self._decode_chunk_size,
                max_guidance_scale=self._guidance_scale,
                fps=p.svd_fps,
                motion_bucket_id=p.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
                generator=generator,
            )

        # latents: (1, num_frames, C, H_latent, W_latent). Frame 0 re-renders
        # the conditioning timestep; only the future frames are returned.
        future = latents[:, 1:]
        pixels = self._decode(future)
        frames = [
            self._frame_scene(pixels[i], future[0, i], chunk[i], instruction)
            for i in range(p.frames_per_chunk)
        ]
        return frames, latents[:, -1]

    def _buffer_entry(self, buffer: list, idx: int):
        return buffer[history_position(len(buffer), idx)]

    # ----- latent <-> pixel ----------------------------------------------------

    def _scene_entry(self, scene: Scene, label: str, generator):
        """A scene's (latent, pose) conditioning pair.

        The latent rides ``metadata["ctrlworld_latent"]`` on every frame this
        adapter generated; a frame without it (the seed, an edited seed) is
        encoded from pixels — per view, never the stacked image in one VAE
        pass."""
        import torch

        reason = self._scene_problem(scene, label)
        if reason is not None:
            raise ValueError(f"ctrl-world rollout request rejected: {reason}")

        p = self._p
        pose = np.concatenate(
            [
                np.asarray(scene.observations.state.values, dtype=np.float64)[:6],
                [float(scene.observations.gripper.value)],
            ]
        )

        cached = scene.metadata.get("ctrlworld_latent")
        if cached is not None:
            arr = np.asarray(cached, dtype=np.float32)
            if arr.shape != p.latent_shape:
                raise ValueError(
                    f"{label}.metadata['ctrlworld_latent'] has shape {arr.shape}, "
                    f"expected {p.latent_shape} (profile '{p.name}')."
                )
            latent = torch.from_numpy(arr).unsqueeze(0).to(self._device).to(self._dtype)
            return latent, pose

        stack = np.asarray(scene.observations.images[self._conditioning_camera].data)
        views = split_stack_view(stack, views=p.views)
        vae = self._model.pipeline.vae
        per_view = []
        with torch.no_grad():
            for name in p.views:
                x = torch.from_numpy(np.ascontiguousarray(views[name])).to(self._device)
                x = x.permute(2, 0, 1).unsqueeze(0).to(self._dtype) / 255.0 * 2 - 1
                lat = vae.encode(x).latent_dist.sample(generator).mul_(vae.config.scaling_factor)
                per_view.append(lat)
        latent = torch.cat(per_view, dim=2)  # stack along latent height
        return latent, pose

    def _decode(self, latents) -> np.ndarray:
        """Decode ``(1, T, C, H_latent, W_latent)`` latents to ``(T, stack_H,
        stack_W, 3)`` uint8 stacks.

        Mirrors the reference decode (``rollout_interact_pi.py`` lines 207-219):
        split per view, decode in ``decode_chunk_size`` batches, map [-1, 1] to
        uint8, then re-stack the views vertically."""
        import einops
        import torch

        p = self._p
        vae = self._model.pipeline.vae
        n_views = len(p.views)
        per_view = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=n_views, n=1
        )  # (n_views, T, C, H_view_latent, W_latent)
        n_frames = per_view.shape[1]
        flat = per_view.flatten(0, 1)
        decoded = []
        with torch.no_grad():
            for i in range(0, flat.shape[0], self._decode_chunk_size):
                batch = flat[i : i + self._decode_chunk_size] / vae.config.scaling_factor
                decoded.append(vae.decode(batch, num_frames=batch.shape[0]).sample)
        video = torch.cat(decoded, dim=0).reshape(n_views, n_frames, 3, *p.view_hw)
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
        video = video.detach().to(torch.float32).cpu().numpy().astype(np.uint8)
        # (views, T, 3, H, W) -> (T, views*H, W, 3): vertical stack per frame.
        video = video.transpose(1, 0, 3, 4, 2)
        return video.reshape(n_frames, *p.stack_hw, 3)

    def _frame_scene(self, stack: np.ndarray, latent, pose: np.ndarray, instruction: str) -> Scene:
        cam = self._conditioning_camera
        import torch

        latent_np = latent.detach().to(torch.float32).cpu().numpy().astype(np.float16)
        pose = np.asarray(pose, dtype=np.float32)
        return Scene(
            observations=Observations(
                images={cam: RGBImage(data=stack, camera_id=cam)},
                state=Proprioception(values=pose[:6].copy(), convention="ee_pose"),
                gripper=GripperState(value=float(pose[6])),
            ),
            instruction=instruction,
            metadata={
                "source": "ctrlworld-forward-dynamics",
                "ctrlworld_latent": latent_np,
            },
        )

    # ----- episode -> conditioning actions -------------------------------------

    def prepare_actions(
        self,
        episode: Trajectory,
        *,
        frame_start: int = 0,
        n_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Encode recorded-episode conditioning: absolute ``[pose, gripper]``
        rows at the profile's native rate.

        The episode must carry an absolute end-effector state (``convention
        'ee_pose'``) — a joint-space episode raises; forward-kinematics belongs
        to the driver, which owns the robot model. ``n_actions`` counts
        native-rate rows and must be a multiple of the profile's chunk quantum.
        """
        p = self._p
        if episode.fps <= 0:
            raise ValueError("ctrl-world prepare_actions: episode.fps is unset.")
        stride = episode.fps / p.native_fps
        if abs(stride - round(stride)) > 1e-6 or round(stride) < 1:
            raise ValueError(
                f"ctrl-world '{p.name}' runs at {p.native_fps:g} Hz; the episode's "
                f"{episode.fps:g} fps is not an integer multiple of it, so frames "
                "cannot be aligned without resampling."
            )
        stride = int(round(stride))

        if n_actions is None:
            available = (len(episode.frames) - 1 - frame_start) // stride
            n_actions = available - available % p.frames_per_chunk
        n_actions = int(n_actions)
        if n_actions < p.frames_per_chunk or n_actions % p.frames_per_chunk != 0:
            raise ValueError(
                f"ctrl-world prepare_actions: n_actions must be a positive multiple "
                f"of {p.frames_per_chunk}; got {n_actions}."
            )

        rows = []
        for k in range(1, n_actions + 1):
            idx = frame_start + k * stride
            if idx >= len(episode.frames):
                raise IndexError(
                    f"ctrl-world prepare_actions: needs episode frame {idx} "
                    f"({n_actions} rows at stride {stride} from {frame_start}) but "
                    f"the episode has {len(episode.frames)} frames."
                )
            scene = episode.frames[idx]
            state = scene.observations.state
            if state is None or getattr(state, "convention", None) != "ee_pose":
                raise ValueError(
                    f"ctrl-world prepare_actions: frame {idx} lacks an 'ee_pose' "
                    "state. Map the dataset's absolute end-effector pose "
                    "(convention: ee_pose) in the config; joint states are not "
                    "reinterpreted."
                )
            if scene.observations.gripper is None:
                raise ValueError(
                    f"ctrl-world prepare_actions: frame {idx} lacks a gripper value."
                )
            values = np.asarray(state.values, dtype=np.float64)
            rows.append(np.concatenate([values[:6], [float(scene.observations.gripper.value)]]))
        return np.stack(rows).astype(np.float32)


__all__ = [
    "ACTION_DIM",
    "CtrlWorldModel",
    "history_position",
    "normalize_bound",
]
