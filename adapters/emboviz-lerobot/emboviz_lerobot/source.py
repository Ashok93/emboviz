"""LeRobot episode source — runs inside the isolated reader venv.

Wraps ``lerobot.datasets.LeRobotDataset`` (the canonical, official
reader — we do not reimplement any decoding) and emits the framework's
universal :class:`Scene` / :class:`Trajectory` types. This package's
venv tracks the LATEST lerobot, whose on-disk format is **v3.0** — the
current official LeRobot standard. emboviz core never imports this module
and never installs lerobot.

emboviz accepts LeRobot **v3.0** datasets only. v3.0 is not backward-
compatible with v2.x (lerobot itself refuses), so a v2.x dataset is
rejected with a clear pointer to lerobot's own ``convert_dataset_v21_to_v30``
— we don't ship an old reader to humour old data.

``build_lerobot_source`` turns a run config's ``dataset`` section into a
configured source: it reads the dataset's own ``meta/info.json`` (a
single JSON file — a metadata peek, not format parsing) for the feature
shapes, builds the :class:`RobotProfile`, and constructs the reader. The
profile/gripper construction is the shared helper from ``emboviz_wire``,
so it matches core's in-process HDF5/RLDS readers exactly.

Only ``emboviz_wire`` (and lerobot/torch, imported lazily) is imported
here — never ``emboviz`` core.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from emboviz_wire.dataset_build import (
    build_profile,
    make_gripper_extractor,
    parse_lerobot_names,
)
from emboviz_wire.observations import GripperState, Proprioception, RGBImage
from emboviz_wire.profile import RobotProfile
from emboviz_wire.reader_protocol import EpisodeSource
from emboviz_wire.types import Observations, Scene, Trajectory

# ``torch`` and ``lerobot`` are intentionally NOT imported at module
# level — they're imported inside the methods that need them so that
# ``import emboviz_lerobot.source`` is cheap (the host imports the spec
# module for entry-point discovery and must not pay a torch import).


# Function that, given a raw state ndarray from the dataset, returns
# (proprioception_values, gripper_value_or_None).
GripperExtractor = Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]


def _identity_state(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """Default: proprio is the whole state, no gripper extraction."""
    return state, None


class LeRobotEpisodeSource(EpisodeSource):
    """Episode source backed by a ``LeRobotDataset`` (HF Hub or local).

    Fields:
      • ``repo_id``    — HF dataset repo id, or a local dataset directory.
      • ``profile``    — RobotProfile for this robot/dataset combination.
      • ``image_keys`` — {camera role → dataset image key}; must include
                         an explicit ``"primary"`` role.
    Optional:
      • ``state_key`` / ``action_key`` — dataset keys for proprio / action.
      • ``gripper_extractor`` — splits raw state into (proprio, gripper).
      • ``n_episodes`` — total episode count (used by ``list_episodes``).
    """

    def __init__(
        self,
        repo_id: str,
        profile: RobotProfile,
        image_keys: dict[str, str],
        *,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        gripper_extractor: GripperExtractor = _identity_state,
        n_episodes: int = 1_000_000,
    ):
        if not image_keys:
            raise ValueError("image_keys must have at least one entry")
        self.repo_id = repo_id
        self.profile = profile
        self.image_keys = dict(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.gripper_extractor = gripper_extractor
        self._n_episodes = n_episodes
        self.name = f"lerobot:{repo_id}"
        self._meta_dataset = None
        # Cache LeRobotDataset instances keyed by the frozen tuple of
        # episode indices. Each instantiation hits HF for ~50 tree-listing
        # API calls; the pool builder samples many episodes per call, so
        # the cache makes batched / repeated loads free.
        self._dataset_cache: dict[tuple[int, ...], object] = {}
        self._dataset_cache_max = 8

    # ----- EpisodeSource interface -----------------------------------

    def list_episodes(self) -> list[str]:
        return [str(i) for i in range(self._n_episodes)]

    def load_episode(self, episode_id: str) -> list[Scene]:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Batched load — single LeRobotDataset init for all indices.

        ``self.repo_id`` may be a HuggingFace repo id (``namespace/dataset``)
        or a local directory holding a lerobot-format dataset (``meta/``,
        ``data/``, ``videos/``). Local paths route via the ``root=`` kwarg
        so lerobot skips its hub lookup.
        """
        indices = sorted(set(int(i) for i in episode_indices))
        cache_key = tuple(indices)
        dataset = self._dataset_cache.get(cache_key)
        if dataset is None:
            dataset = self._open(indices)
            self._dataset_cache[cache_key] = dataset
            if len(self._dataset_cache) > self._dataset_cache_max:
                self._dataset_cache.pop(next(iter(self._dataset_cache)))

        out: dict[int, list[Scene]] = {i: [] for i in indices}
        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", indices[0]))
            if ep_i not in out:
                continue
            instruction = self._resolve_instruction(sample)
            scene = self._build_scene(sample, instruction, ep_i, len(out[ep_i]), dataset.fps)
            out[ep_i].append(scene)
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        fps = float(scenes[0].metadata.get("fps", 5.0)) if scenes else 5.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.repo_id},
        )

    def all_instructions(self) -> list[str]:
        if self._meta_dataset is None:
            self._meta_dataset = self._open([0])
        tasks = getattr(self._meta_dataset.meta, "tasks", None)
        if tasks is None:
            return []
        # LeRobot v3.0: ``meta.tasks`` is a pandas DataFrame indexed by the
        # task STRING (column = task_index). The strings are the index.
        if hasattr(tasks, "index") and not isinstance(tasks, (list, tuple, dict)):
            return [str(t) for t in tasks.index]
        # Defensive fallbacks for dict / list-shaped task tables.
        items = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
        out: list[str] = []
        for it in items:
            if isinstance(it, dict) and "task" in it:
                out.append(str(it["task"]))
            elif isinstance(it, str):
                out.append(it)
        return out

    # ----- internals -------------------------------------------------

    def _open(self, indices: list[int]):
        """Construct the underlying ``LeRobotDataset``.

        On a local failure we surface the REAL cause: lerobot falls back
        to an HF lookup when it can't read local metadata, and since we
        hand it the placeholder repo_id ``"local"`` that surfaces as a
        misleading ``404 ... repo 'local'`` — we re-raise with the true
        local context.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        is_local = os.path.isdir(self.repo_id) or self.repo_id.startswith("/")
        try:
            if is_local:
                return LeRobotDataset("local", root=self.repo_id, episodes=indices)
            return LeRobotDataset(self.repo_id, episodes=indices)
        except Exception as e:
            if not is_local:
                raise
            meta = Path(self.repo_id) / "meta"
            present = sorted(p.name for p in meta.iterdir()) if meta.is_dir() else []
            raise RuntimeError(
                f"Failed to load local LeRobot dataset at {self.repo_id!r}: "
                f"{type(e).__name__}: {e}\n"
                f"meta/ contains {present}. NOTE: lerobot retries on the HF Hub "
                f"when it cannot read a dataset's local metadata, so an "
                f"underlying \"404 ... datasets/local\" is the masked fallback — "
                f"the real failure is the local one above."
            ) from e

    def _build_scene(
        self, sample: dict, instruction: str,
        episode_idx: int, frame_offset: int, fps: float,
    ) -> Scene:
        images: dict[str, RGBImage] = {}
        for cam_name, key in self.image_keys.items():
            if key not in sample:
                continue
            pil = self._tensor_to_pil(sample[key])
            images[cam_name] = RGBImage(data=pil, camera_id=cam_name)

        # Strict: the binding MUST name a "primary" camera. We never
        # silently alias the first declared camera — that routinely puts
        # the wrong viewpoint into single-cam diagnostics.
        if "primary" not in images and images:
            raise KeyError(
                f"Dataset adapter for repo_id={self.repo_id!r} loaded "
                f"cameras {sorted(images)} but none are named 'primary'. "
                "Add an explicit \"primary\" entry to dataset.cameras so "
                "the framework knows which view is the main exterior camera."
            )

        import torch

        proprio: Optional[Proprioception] = None
        gripper: Optional[GripperState] = None
        raw_state = None
        if self.state_key and self.state_key in sample:
            raw_state = sample[self.state_key].to(torch.float32).reshape(-1).numpy()
            proprio_vals, gripper_val = self.gripper_extractor(raw_state)
            state_convention = (
                self.profile.state.convention if self.profile.state is not None
                else "joint_angles"
            )
            proprio = Proprioception(values=proprio_vals.copy(), convention=state_convention)
            if gripper_val is not None and self.profile.gripper is not None:
                gripper = GripperState(
                    value=float(gripper_val),
                    kind=self.profile.gripper.kind,
                    units=self.profile.gripper.units,
                )

        obs = Observations(images=images, state=proprio, gripper=gripper)

        metadata: dict = {
            "fps": float(fps),
            "frame_index": int(sample.get("frame_index", frame_offset)),
            "episode_index": episode_idx,
            "dataset": self.repo_id,
        }
        if raw_state is not None:
            metadata["raw_state"] = raw_state.tolist()
        if self.action_key and self.action_key in sample:
            metadata["expert_action"] = (
                sample[self.action_key].to(torch.float32).reshape(-1).tolist()
            )

        return Scene(
            observations=obs,
            instruction=instruction,
            profile=self.profile,
            metadata=metadata,
            scene_id=f"{self.name}:{episode_idx}:{frame_offset}",
        )

    def _tensor_to_pil(self, t) -> Image.Image:
        """Convert a lerobot image tensor → PIL.Image.

        Strict dtype handling: floating tensors are assumed [0, 1] and
        rescaled to [0, 255]; integer tensors are assumed already [0, 255]
        and only clipped. No "if max ≤ 1.5 multiply" heuristic — a
        genuinely-dark uint8 frame can have max < 2 and the heuristic
        would overflow it silently.
        """
        if hasattr(t, "detach"):
            raw = t.detach().cpu().numpy()
        else:
            raw = np.asarray(t)
        if raw.ndim == 3 and raw.shape[0] in (1, 3):
            raw = raw.transpose(1, 2, 0)
        if np.issubdtype(raw.dtype, np.floating):
            a = (raw * 255.0).astype(np.float32)
        elif np.issubdtype(raw.dtype, np.integer):
            a = raw.astype(np.float32)
        else:
            raise TypeError(
                f"LeRobotEpisodeSource._tensor_to_pil: unsupported dtype "
                f"{raw.dtype}. Expected floating ([0,1]) or integer ([0,255]) "
                "image tensor. No silent conversion."
            )
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _resolve_instruction(self, sample: dict) -> str:
        """Instruction string for this frame.

        LeRobot v3.0 carries the task STRING directly in each sample
        (``sample['task']``) — no task_index → table lookup needed.
        """
        task = sample.get("task", "")
        return task if isinstance(task, str) else ""


# ─────────────────────────────────────────────────────────────────────
# Config → source construction (runs in the reader worker)
# ─────────────────────────────────────────────────────────────────────


def _read_lerobot_info(path: str) -> dict:
    """Read ``meta/info.json`` for a LeRobot dataset — local dir or HF repo.

    A single-file metadata peek (``hf_hub_download`` of one file) — this
    is reading the dataset's declared schema, NOT parsing the format.
    """
    if os.path.isdir(path):
        info_path = Path(path) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"{info_path} not found in local dataset")
        return json.loads(info_path.read_text())
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=path, filename="meta/info.json", repo_type="dataset")
    return json.loads(Path(p).read_text())


def _assert_readable(path: str, info: dict) -> None:
    """Fail loudly unless the dataset is the LeRobot format this reader's
    lerobot reads (v3.0). lerobot's v3.0 is a hard major break — it cannot
    read v2.x — so we refuse v2.x up front with the exact one-time fix
    rather than emit a masked HF error or risk wrong frames."""
    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    except Exception:  # pragma: no cover - lerobot layout drift
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
    ds_version = str(info.get("codebase_version", "")).lstrip("v")
    supported = str(CODEBASE_VERSION).lstrip("v")
    if not ds_version:
        return
    try:
        ds_major = int(ds_version.split(".", 1)[0])
        sup_major = int(supported.split(".", 1)[0])
    except ValueError:  # pragma: no cover - unexpected version string
        return
    if ds_major == sup_major:
        return
    if ds_major < sup_major:
        raise RuntimeError(
            f"Dataset {path!r} is LeRobot format v{ds_version}; emboviz uses the "
            f"latest LeRobot (v{supported}), and v3.0 is NOT backward-compatible "
            f"with v2.x. Convert it once with lerobot's own tool, then re-run:\n"
            f"    python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 "
            f"--repo-id={path}\n"
            f"(If it's v2.0, run convert_dataset_v20_to_v21 first.) We refuse to "
            f"read v2.x with a v3.0 reader rather than risk wrong frames."
        )
    raise RuntimeError(
        f"Dataset {path!r} is LeRobot format v{ds_version}, NEWER than this "
        f"reader's lerobot (v{supported}). Bump emboviz-lerobot's lerobot pin to "
        f"a release that reads v{ds_major}.x."
    )


def build_lerobot_source(
    *,
    path: str,
    cameras: dict[str, str],
    state: Optional[dict] = None,
    action: Optional[dict] = None,
    gripper: Optional[dict] = None,
    instruction: Optional[dict] = None,
    n_episodes: Optional[int] = None,
) -> LeRobotEpisodeSource:
    """Build a configured :class:`LeRobotEpisodeSource` from a run config's
    ``dataset`` section. Runs in the reader worker (has lerobot)."""
    if "primary" not in (cameras or {}):
        raise KeyError(
            "dataset.cameras must include a 'primary' role (the main "
            f"exterior camera). Got roles {sorted(cameras or {})}. We never "
            "auto-pick a primary camera."
        )

    info = _read_lerobot_info(path)
    _assert_readable(path, info)
    features = info.get("features", {})

    state_key = state["key"] if state else None
    action_key = action["key"] if action else "action"

    state_dim = state_names = None
    if state_key is not None:
        feat = features.get(state_key)
        if feat is None:
            raise KeyError(
                f"dataset.state.key={state_key!r} is not a feature in "
                f"{path}'s info.json. Available: {sorted(features)}."
            )
        state_dim = feat["shape"][0]
        state_names = parse_lerobot_names(feat.get("names"))

    action_dim = action_names = None
    if action_key in features:
        action_dim = features[action_key]["shape"][0]
        action_names = parse_lerobot_names(features[action_key].get("names"))

    profile = build_profile(
        name=info.get("robot_type") or path,
        cameras=cameras,
        state_dim=state_dim, state_names=state_names,
        convention=(state or {}).get("convention"),
        action_dim=action_dim, action_names=action_names,
        gripper=gripper,
    )
    return LeRobotEpisodeSource(
        repo_id=path,
        profile=profile,
        image_keys=dict(cameras),
        state_key=state_key,
        action_key=action_key,
        gripper_extractor=make_gripper_extractor(gripper, state_names),
        n_episodes=int(n_episodes or info.get("total_episodes", 1_000_000)),
    )
