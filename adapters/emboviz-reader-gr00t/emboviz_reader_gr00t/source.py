"""GR00T-format dataset source — runs inside the isolated reader-gr00t venv.

A GR00T dataset is a standard LeRobot **v2.1** dataset (parquet + mp4 +
``meta/{info,episodes,tasks}.jsonl``) PLUS one extra file,
``meta/modality.json``, which declares how the packed
``observation.state`` / ``action`` vectors split into named fields
(NVIDIA Isaac-GR00T's schema). See NVIDIA's data_preparation.md.

This reader wraps the **canonical** ``lerobot.datasets.LeRobotDataset``
(the v2.1-era reader, lerobot 0.3.x) for ALL decoding — parquet, mp4
video, task lookup — and reimplements none of it. Its only GR00T-specific
work is reading ``modality.json`` to locate the gripper inside the packed
state vector. It emits the framework's universal :class:`Scene` /
:class:`Trajectory` types, so every diagnostic consumes a GR00T dataset
exactly like any other.

Why a SEPARATE reader (not the v3.0 ``emboviz-lerobot`` one): lerobot
>=0.4 reads only the v3.0 on-disk format and hard-refuses v2.x
(``BackwardCompatibilityError``). GR00T datasets are v2.1, so this
reader's venv pins the last v2.1-capable lerobot (``>=0.3.3,<0.4``).
emboviz core never imports this module or lerobot.

Episode loading uses lerobot's ``episode_data_index`` (per-episode
"from"/"to" frame ranges) over the FULL dataset, NOT the
``episodes=[...]`` constructor filter: in lerobot 0.3.x that filter
shortens ``episode_data_index`` and then indexes it by the absolute
episode number, raising IndexError for any non-trivial selection (GitHub
lerobot #816 / PR #1062, unmerged). ``episode_data_index`` is lerobot's
own canonical episode-boundary API and is unaffected.

Only ``emboviz_wire`` (and lerobot/torch, imported lazily) is imported
here — never ``emboviz`` core.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from emboviz_wire.dataset_build import build_profile, parse_lerobot_names
from emboviz_wire.observations import GripperState, Proprioception, RGBImage
from emboviz_wire.profile import RobotProfile
from emboviz_wire.reader_protocol import EpisodeSource
from emboviz_wire.types import Observations, Scene, Trajectory

# ``torch`` and ``lerobot`` are imported lazily inside the methods that
# need them — importing this module for entry-point discovery must not
# pay a torch import.


class Gr00tDatasetSource(EpisodeSource):
    """Episode source backed by a GR00T-format (LeRobot v2.1 + modality.json)
    dataset, read through the canonical v2.1 ``LeRobotDataset``.

    Fields:
      • ``repo_id``    — HF dataset repo id, or a local dataset directory.
      • ``profile``    — RobotProfile built from info.json + the config.
      • ``image_keys`` — {camera role → dataset image key}; must include
                         an explicit ``"primary"`` role.
    Optional:
      • ``state_key`` / ``action_key`` — dataset keys for proprio / action.
      • ``gripper_index`` — index of the gripper scalar inside the packed
                            state vector, derived from ``modality.json``.
                            ``None`` leaves the gripper inside the state.
    """

    def __init__(
        self,
        repo_id: str,
        profile: RobotProfile,
        image_keys: dict[str, str],
        *,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        gripper_index: Optional[int] = None,
        n_episodes: int = 0,
        fps: float = 0.0,
    ):
        if not image_keys:
            raise ValueError("image_keys must have at least one entry")
        self.repo_id = repo_id
        self.profile = profile
        self.image_keys = dict(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.gripper_index = gripper_index
        self._n_episodes = n_episodes
        # fps comes from info.json (the dataset's own schema) rather than a
        # lerobot ``LeRobotDataset.fps`` attribute, so we don't depend on a
        # 0.3.x-specific API surface.
        self.fps = float(fps)
        self.name = f"gr00t:{repo_id}"
        # One FULL LeRobotDataset handle, opened lazily and reused for every
        # episode (sliced via episode_data_index). Opening the full dataset
        # — episodes=None — keeps episode_data_index correct for every
        # episode and sidesteps the lerobot 0.3.x episodes=[...] index bug.
        self._dataset = None

    # ----- EpisodeSource interface -----------------------------------

    def list_episodes(self) -> list[str]:
        return [str(i) for i in range(self._n_episodes)]

    def load_episode(self, episode_id: str) -> list[Scene]:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Load several episodes, keyed by index.

        Slices each requested episode's frame range out of the single full
        dataset handle via ``episode_data_index``. No per-episode dataset
        construction, and no ``episodes=[...]`` filter — see module docstring.
        """
        dataset = self._open()
        from_idx, to_idx = self._episode_bounds(dataset)
        n_eps = len(from_idx)

        out: dict[int, list[Scene]] = {}
        for ep in sorted(set(int(i) for i in episode_indices)):
            if ep < 0 or ep >= n_eps:
                raise IndexError(
                    f"episode {ep} is out of range for {self.repo_id!r} "
                    f"(has {n_eps} episodes)."
                )
            start, end = int(from_idx[ep]), int(to_idx[ep])
            scenes: list[Scene] = []
            for offset, gi in enumerate(range(start, end)):
                sample = dataset[gi]
                instruction = self._resolve_instruction(sample)
                scenes.append(
                    self._build_scene(sample, instruction, ep, offset, self.fps)
                )
            out[ep] = scenes
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
        """Every unique task string the dataset declares.

        Read straight from the standard v2.1 ``meta/tasks.jsonl`` (one
        ``{"task_index", "task"}`` record per line) rather than through a
        lerobot internal whose shape varies across 0.3.x — the file is part
        of the documented format, so this is a metadata peek, not decoding.
        """
        seen: dict[str, None] = {}
        for record in self._read_tasks_jsonl():
            task = record.get("task")
            if isinstance(task, str) and task:
                seen.setdefault(task, None)
        return list(seen)

    # ----- internals -------------------------------------------------

    def _open(self):
        """Construct (once) the underlying FULL ``LeRobotDataset``.

        ``episodes=None`` so episode_data_index covers every episode with
        correct absolute frame bounds. On a local failure we surface the
        REAL cause: lerobot retries on the HF Hub when it can't read local
        metadata, and the placeholder repo_id ``"local"`` then surfaces as
        a misleading ``404 ... repo 'local'``.
        """
        if self._dataset is not None:
            return self._dataset

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        is_local = os.path.isdir(self.repo_id) or self.repo_id.startswith("/")
        try:
            if is_local:
                self._dataset = LeRobotDataset("local", root=self.repo_id)
            else:
                self._dataset = LeRobotDataset(self.repo_id)
        except Exception as e:
            if not is_local:
                raise
            meta = Path(self.repo_id) / "meta"
            present = sorted(p.name for p in meta.iterdir()) if meta.is_dir() else []
            raise RuntimeError(
                f"Failed to load local GR00T (LeRobot v2.1) dataset at "
                f"{self.repo_id!r}: {type(e).__name__}: {e}\n"
                f"meta/ contains {present}. NOTE: lerobot retries on the HF "
                f"Hub when it cannot read a dataset's local metadata, so an "
                f"underlying \"404 ... datasets/local\" is the masked "
                f"fallback — the real failure is the local one above."
            ) from e
        return self._dataset

    @staticmethod
    def _episode_bounds(dataset):
        """Return (from, to) per-episode frame-index tensors.

        ``episode_data_index`` is a dict of two 1-D tensors indexed by
        absolute episode number; ``from[ep]:to[ep]`` is episode ``ep``'s
        global frame range. We fail loudly if it is absent rather than
        guess episode boundaries.
        """
        edi = getattr(dataset, "episode_data_index", None)
        if not edi or "from" not in edi or "to" not in edi:
            raise RuntimeError(
                "LeRobotDataset did not expose episode_data_index "
                "{'from','to'}; cannot resolve episode frame ranges. This "
                "reader targets lerobot 0.3.x (LeRobot v2.1)."
            )
        return edi["from"], edi["to"]

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
                f"GR00T reader for repo_id={self.repo_id!r} loaded cameras "
                f"{sorted(images)} but none are named 'primary'. Add an "
                "explicit \"primary\" entry to dataset.cameras so the "
                "framework knows which view is the main exterior camera."
            )

        import torch

        proprio: Optional[Proprioception] = None
        gripper: Optional[GripperState] = None
        raw_state = None
        if self.state_key and self.state_key in sample:
            raw_state = sample[self.state_key].to(torch.float32).reshape(-1).numpy()
            state_convention = (
                self.profile.state.convention if self.profile.state is not None
                else "joint_angles"
            )
            # Proprio is the FULL state vector (the model consumes the whole
            # state it was trained on); the gripper scalar is ALSO read from
            # its declared index — same semantics as the lerobot reader.
            proprio = Proprioception(values=raw_state.copy(), convention=state_convention)
            if self.gripper_index is not None and self.profile.gripper is not None:
                if self.gripper_index >= raw_state.size:
                    raise ValueError(
                        f"gripper index {self.gripper_index} (from "
                        f"modality.json) is out of range for a "
                        f"{raw_state.size}-dim state vector."
                    )
                gripper = GripperState(
                    value=float(raw_state[self.gripper_index]),
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
                f"Gr00tDatasetSource._tensor_to_pil: unsupported dtype "
                f"{raw.dtype}. Expected floating ([0,1]) or integer ([0,255]) "
                "image tensor. No silent conversion."
            )
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _resolve_instruction(self, sample: dict) -> str:
        """Instruction string for this frame.

        lerobot 0.3.x resolves ``task_index`` → the task string and adds it
        to each frame as ``sample['task']`` — no manual table lookup.
        """
        task = sample.get("task", "")
        return task if isinstance(task, str) else ""

    def _read_tasks_jsonl(self) -> list[dict]:
        """Read the dataset's ``meta/tasks.jsonl`` (local dir or HF repo)."""
        if os.path.isdir(self.repo_id):
            tasks_path = Path(self.repo_id) / "meta" / "tasks.jsonl"
            if not tasks_path.is_file():
                return []
            text = tasks_path.read_text()
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(
                repo_id=self.repo_id, filename="meta/tasks.jsonl",
                repo_type="dataset",
            )
            text = Path(p).read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ─────────────────────────────────────────────────────────────────────
# Config → source construction (runs in the reader worker)
# ─────────────────────────────────────────────────────────────────────


def _read_json_member(path: str, member: str) -> dict:
    """Read a single ``meta/<member>`` JSON file — local dir or HF repo.

    A single-file metadata peek (``hf_hub_download`` of one file), i.e.
    reading the dataset's declared schema, NOT parsing the format.
    """
    if os.path.isdir(path):
        p = Path(path) / "meta" / member
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found in local dataset")
        return json.loads(p.read_text())
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=path, filename=f"meta/{member}", repo_type="dataset",
    )
    return json.loads(Path(local).read_text())


def _read_modality(path: str) -> dict:
    """Read ``meta/modality.json`` — the file that makes a LeRobot dataset a
    GR00T dataset. Its absence is a loud, actionable error: a GR00T dataset
    IS "LeRobot v2.1 + modality.json", so without it we have an ordinary
    LeRobot dataset and refuse to invent the state/action layout.
    """
    try:
        return _read_json_member(path, "modality.json")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{path!r} has no meta/modality.json — it is not a GR00T dataset "
            "yet. A GR00T dataset is a LeRobot v2.1 dataset PLUS "
            "meta/modality.json (NVIDIA Isaac-GR00T's state/action/video "
            "layout). Add it the way Isaac-GR00T does, e.g.\n"
            "    cp Isaac-GR00T/examples/LIBERO/modality.json "
            f"{path}/meta/\n"
            "then re-run. We refuse to guess the packed-state layout."
        ) from e
    except Exception as e:
        # HF repo without the file surfaces as a 404 EntryNotFound — give the
        # same actionable message rather than a raw hub error.
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            raise FileNotFoundError(
                f"HF dataset {path!r} does not carry meta/modality.json, so "
                "it is not a GR00T dataset. Download it locally, add "
                "modality.json to its meta/ (per Isaac-GR00T), and point "
                "dataset.path at the local copy."
            ) from e
        raise


def _gripper_index_from_modality(
    modality: dict, state_dim: int,
) -> Optional[int]:
    """Locate the gripper scalar inside the packed state via modality.json.

    modality.json's ``state`` maps field name → ``{start, end}`` (half-open
    slice into the packed ``observation.state`` vector). We return the
    ``start`` of the field named ``gripper`` — the first gripper dim — as
    the scalar open/close signal the framework's GripperState + gripper
    perturber operate on. A GR00T gripper field may span >1 dim (e.g. two
    finger positions); the diagnostic gripper is one scalar, so we surface
    the first dim and do not average/fabricate. ``None`` if the dataset
    declares no gripper field.
    """
    state_fields = modality.get("state")
    if not isinstance(state_fields, dict):
        raise ValueError(
            "modality.json has no 'state' object mapping field names to "
            "{start,end} ranges. It does not match the GR00T schema."
        )
    spec = state_fields.get("gripper")
    if spec is None:
        return None
    if not isinstance(spec, dict) or "start" not in spec:
        raise ValueError(
            f"modality.json state.gripper={spec!r} is malformed; expected "
            "{'start': int, 'end': int}."
        )
    start = int(spec["start"])
    end = int(spec.get("end", start + 1))
    if not (0 <= start < end <= state_dim):
        raise ValueError(
            f"modality.json state.gripper range [{start}:{end}] is out of "
            f"bounds for the {state_dim}-dim observation.state declared in "
            "info.json. modality.json and the dataset disagree."
        )
    return start


def build_gr00t_source(
    *,
    path: str,
    cameras: dict[str, str],
    state: Optional[dict] = None,
    action: Optional[dict] = None,
    gripper: Optional[dict] = None,
    instruction: Optional[dict] = None,
    n_episodes: Optional[int] = None,
) -> Gr00tDatasetSource:
    """Build a configured :class:`Gr00tDatasetSource` from a run config's
    ``dataset`` section. Runs in the reader worker (has lerobot 0.3.x).

    The state→proprio/gripper split is driven by the dataset's own
    ``meta/modality.json`` (the gripper index), not a hand-typed config
    field — the dataset declares its layout, we read it. ``state.convention``
    still comes from the config: no format (modality.json included) encodes
    joint-angles vs ee-pose, so the user states it.
    """
    if "primary" not in (cameras or {}):
        raise KeyError(
            "dataset.cameras must include a 'primary' role (the main "
            f"exterior camera). Got roles {sorted(cameras or {})}. We never "
            "auto-pick a primary camera."
        )

    # Expand ``~`` so a local path like ``~/data/...`` is recognised as a
    # local directory (os.path.isdir does not expand ``~``) and routed to
    # ``root=`` rather than mistaken for an HF repo id.
    path = os.path.expanduser(path)

    info = _read_json_member(path, "info.json")
    _assert_v2(path, info)
    modality = _read_modality(path)
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

    # Validate the declared camera keys exist as features (loud, not silent).
    for role, key in cameras.items():
        if key not in features:
            raise KeyError(
                f"dataset.cameras[{role!r}]={key!r} is not a feature in "
                f"{path}'s info.json. Available: {sorted(features)}."
            )

    # Gripper index from modality.json (the dataset's self-description). We
    # extract a gripper only when the config opts in with a gripper block
    # (kind/units/range — physical facts the user supplies) AND modality.json
    # declares where it is.
    gripper_index: Optional[int] = None
    if gripper is not None:
        if gripper.get("source") is not None:
            raise ValueError(
                "dataset.gripper.source is set for a 'gr00t' dataset, but the "
                "GR00T reader derives the gripper index from the dataset's "
                "own meta/modality.json (its single source of truth). Remove "
                "gripper.source from the config to avoid two conflicting "
                "definitions."
            )
        if state_dim is None:
            raise ValueError(
                "dataset.gripper is set but dataset.state is absent — the "
                "gripper is extracted from the state vector, so state.key is "
                "required."
            )
        gripper_index = _gripper_index_from_modality(modality, int(state_dim))
        if gripper_index is None:
            raise ValueError(
                f"dataset.gripper is set but {path}'s modality.json declares "
                "no 'gripper' field under 'state'. Either remove dataset."
                "gripper or fix modality.json. We do not guess the index."
            )

    profile = build_profile(
        name=info.get("robot_type") or path,
        cameras=cameras,
        state_dim=state_dim, state_names=state_names,
        convention=(state or {}).get("convention"),
        action_dim=action_dim, action_names=action_names,
        gripper=gripper,
    )
    return Gr00tDatasetSource(
        repo_id=path,
        profile=profile,
        image_keys=dict(cameras),
        state_key=state_key,
        action_key=action_key,
        gripper_index=gripper_index,
        n_episodes=int(n_episodes or info.get("total_episodes", 0)),
        fps=float(info.get("fps", 0.0)),
    )


def _assert_v2(path: str, info: dict) -> None:
    """Fail loudly unless the dataset is LeRobot v2.x — the format this
    reader's lerobot (0.3.x) reads. A v3.0 dataset belongs to the
    ``emboviz-lerobot`` reader; a v1.x one must be converted first."""
    ds_version = str(info.get("codebase_version", "")).lstrip("v")
    if not ds_version:
        return
    try:
        major = int(ds_version.split(".", 1)[0])
    except ValueError:  # pragma: no cover - unexpected version string
        return
    if major == 2:
        return
    if major >= 3:
        raise RuntimeError(
            f"Dataset {path!r} is LeRobot format v{ds_version}. The GR00T "
            "reader reads v2.x (GR00T datasets are LeRobot v2.1 + "
            "modality.json). A v3.0 dataset is read by the 'lerobot' reader "
            "(dataset.format: lerobot) — but note GR00T's modality.json "
            "layout is a v2.x convention."
        )
    raise RuntimeError(
        f"Dataset {path!r} is LeRobot format v{ds_version}, older than v2.x. "
        "Convert it up to v2.1 with lerobot's own converter before adding "
        "modality.json."
    )
