"""Checkpoint profiles — everything about a Ctrl-World checkpoint that is data,
not code.

A Ctrl-World checkpoint is defined by more than its weights: the camera views
it stacks (names, order, per-view size), the rate it dreams at, its history
schedule, the action-normalization bounds of its training data, and where the
weights live. This module carries that contract as a :class:`CtrlWorldProfile`,
so supporting a new checkpoint (a fine-tune on another embodiment, a different
camera rig) is a profile entry or a JSON file — never an adapter code change.

Resolution follows the ``emboviz-robot`` catalog pattern: ``resolve_profile``
accepts either a preconfigured name (``"droid"``, the released DROID
checkpoint) or a path to a profile JSON with the same fields. The driver and
the worker resolve the same profile from the same installed package, so the
two sides cannot disagree about the checkpoint's contract.

One thing is deliberately NOT a profile field: the action semantics. Every
profile conditions on absolute end-effector rows ``[x, y, z, roll, pitch, yaw,
gripper]`` (extrinsic-XYZ euler, gripper in [0, 1]) — joint-space embodiments
reach this format through forward kinematics (``emboviz-robot``), which is
what lets a checkpoint fine-tuned from the DROID weights keep a familiar
action space. A model trained on different action semantics is a different
adapter, not a profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

#: Action dimensionality — the fixed semantic contract described above.
ACTION_DIM = 7

#: SVD's VAE downsamples height/width by this factor; view sizes must divide it.
_VAE_FACTOR = 8

_DROID_STAT_PATH = Path(__file__).parent / "_ctrl_world" / "droid_stat.json"


@dataclass(frozen=True)
class CtrlWorldProfile:
    """The contract of one Ctrl-World checkpoint.

    Fields describe what the checkpoint was *trained with*; they are not
    tunables. Generation-time knobs (denoise steps, guidance, dtype, seed)
    stay constructor arguments of :class:`emboviz_ctrlworld.model.
    CtrlWorldModel`.
    """

    name: str                          # profile id, e.g. "droid"
    description: str                   # one line for listings and errors
    embodiment: str                    # reported as the WorldModel's supported domain
    # ----- weights -----------------------------------------------------------
    ckpt_repo: str                     # HF repo id or local directory
    ckpt_file: str                     # checkpoint filename within ckpt_repo
    svd_repo: str                      # frozen SVD base (HF id or local dir)
    clip_repo: str                     # CLIP text encoder (HF id or local dir)
    # ----- conditioning geometry --------------------------------------------
    views: tuple[str, ...]             # stack order, top to bottom
    view_hw: tuple[int, int]           # per-view (H, W) the checkpoint trained on
    # ----- conditioning dynamics --------------------------------------------
    native_fps: float                  # rate of one dreamed frame
    num_frames: int                    # frames per forward pass (frame 0 re-renders current)
    num_history: int                   # sparse history frames per forward pass
    history_idx: tuple[int, ...]       # history-buffer schedule (0 = seed, negatives from the end)
    svd_fps: int                       # SVD fps micro-conditioning the checkpoint trained with
    motion_bucket_id: int              # SVD motion micro-conditioning, same
    # ----- action normalization ---------------------------------------------
    state_p01: tuple[float, ...]       # per-dim lower bound (1st percentile of training data)
    state_p99: tuple[float, ...]       # per-dim upper bound (99th percentile)
    # ----- driver defaults ---------------------------------------------------
    # region (view name) -> episode camera role, used when the run config
    # leaves stress.concat_cameras unset.
    default_region_cameras: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.embodiment:
            raise ValueError("profile: name and embodiment must be non-empty.")
        if not self.views:
            raise ValueError(f"profile '{self.name}': views must be non-empty.")
        if len(set(self.views)) != len(self.views):
            raise ValueError(f"profile '{self.name}': views contain duplicates: {self.views}.")
        h, w = self.view_hw
        if h < _VAE_FACTOR or w < _VAE_FACTOR or h % _VAE_FACTOR or w % _VAE_FACTOR:
            raise ValueError(
                f"profile '{self.name}': view_hw {self.view_hw} must be positive "
                f"multiples of {_VAE_FACTOR} (the SVD VAE's downsampling factor)."
            )
        if self.num_frames < 2:
            raise ValueError(
                f"profile '{self.name}': num_frames must be >= 2 (frame 0 re-renders "
                f"the conditioning timestep); got {self.num_frames}."
            )
        if self.num_history < 1:
            raise ValueError(f"profile '{self.name}': num_history must be >= 1.")
        if len(self.history_idx) != self.num_history:
            raise ValueError(
                f"profile '{self.name}': history_idx has {len(self.history_idx)} "
                f"entries but num_history is {self.num_history}."
            )
        if any(i > 0 for i in self.history_idx):
            raise ValueError(
                f"profile '{self.name}': history_idx entries must be 0 (the seed) "
                f"or negative (from the buffer end); got {self.history_idx}."
            )
        if self.native_fps <= 0 or self.svd_fps < 1:
            raise ValueError(f"profile '{self.name}': native_fps/svd_fps out of range.")
        if len(self.state_p01) != ACTION_DIM or len(self.state_p99) != ACTION_DIM:
            raise ValueError(
                f"profile '{self.name}': state_p01/state_p99 must be {ACTION_DIM}-D "
                "[xyz(3), euler_xyz(3), gripper(1)] bounds."
            )
        if any(hi <= lo for lo, hi in zip(self.state_p01, self.state_p99)):
            raise ValueError(
                f"profile '{self.name}': every state_p99 entry must exceed its "
                "state_p01 entry (degenerate bounds cannot normalize)."
            )
        if self.default_region_cameras and set(self.default_region_cameras) != set(self.views):
            raise ValueError(
                f"profile '{self.name}': default_region_cameras keys "
                f"{sorted(self.default_region_cameras)} must equal the views "
                f"{sorted(self.views)}."
            )

    # ----- derived ------------------------------------------------------------

    @property
    def frames_per_chunk(self) -> int:
        """Future frames generated per forward pass."""
        return self.num_frames - 1

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """Per-frame latent ``(C, H_latent, W_latent)`` of the view stack."""
        return (
            4,
            self.view_hw[0] // _VAE_FACTOR * len(self.views),
            self.view_hw[1] // _VAE_FACTOR,
        )

    @property
    def stack_hw(self) -> tuple[int, int]:
        """Pixel size of the stitched view stack."""
        return (self.view_hw[0] * len(self.views), self.view_hw[1])


def _droid_profile() -> CtrlWorldProfile:
    """The released DROID checkpoint (Ctrl-World reference ``config.py``).

    Bounds come verbatim from the vendored ``droid_stat.json`` (the training
    pipeline's 1st/99th-percentile DROID ``[cartesian_position,
    gripper_position]`` stats; see ``_ctrl_world/README.md`` for provenance).
    """
    stat = json.loads(_DROID_STAT_PATH.read_text())
    return CtrlWorldProfile(
        name="droid",
        description="Ctrl-World DROID checkpoint (yjguo/Ctrl-World, ICLR 2026)",
        embodiment="droid",
        ckpt_repo="yjguo/Ctrl-World",
        ckpt_file="checkpoint-10000.pt",
        svd_repo="stabilityai/stable-video-diffusion-img2vid",
        clip_repo="openai/clip-vit-base-patch32",
        views=("exterior_1", "exterior_2", "wrist"),
        view_hw=(192, 320),
        native_fps=5.0,
        num_frames=5,
        num_history=6,
        history_idx=(0, 0, -12, -9, -6, -3),
        svd_fps=7,
        motion_bucket_id=127,
        state_p01=tuple(stat["state_01"]),
        state_p99=tuple(stat["state_99"]),
        default_region_cameras={
            "exterior_1": "primary",
            "exterior_2": "exterior_2",
            "wrist": "wrist",
        },
    )


#: Preconfigured profiles, by name. Lazily constructed (the droid entry reads
#: its bounds file once) and memoized.
_FACTORIES = {
    "droid": _droid_profile,
}
_CACHE: dict[str, CtrlWorldProfile] = {}


def get_profile(name: str) -> CtrlWorldProfile:
    """Return a preconfigured profile by name; raises with the catalog if unknown."""
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown ctrl-world profile {name!r}. Preconfigured: "
            f"{sorted(_FACTORIES)}. For a custom checkpoint, pass the path to a "
            "profile JSON instead (see emboviz_ctrlworld.profiles.load_profile)."
        )
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name]()
    return _CACHE[name]


def load_profile(path: str | Path) -> CtrlWorldProfile:
    """Load a custom profile from a JSON file.

    The JSON carries exactly the :class:`CtrlWorldProfile` fields (lists for
    the tuple fields). Unknown keys are rejected — a typo'd field must fail
    loud, not silently fall back to a default.
    """
    p = Path(path).expanduser()
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"profile file {p} must hold a JSON object.")
    allowed = {f.name for f in fields(CtrlWorldProfile)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"profile file {p} has unknown field(s) {unknown}; "
            f"allowed: {sorted(allowed)}."
        )
    missing = sorted(
        allowed - set(raw) - {"default_region_cameras"}   # the only defaulted field
    )
    if missing:
        raise ValueError(f"profile file {p} is missing required field(s) {missing}.")
    for key in ("views", "view_hw", "history_idx", "state_p01", "state_p99"):
        raw[key] = tuple(raw[key])
    return CtrlWorldProfile(**raw)


def resolve_profile(name_or_path: str) -> CtrlWorldProfile:
    """Resolve a profile from a preconfigured name or a JSON file path.

    Anything that exists on disk (or ends in ``.json``) is treated as a file;
    everything else is a catalog name.
    """
    p = Path(str(name_or_path)).expanduser()
    if p.suffix == ".json" or p.is_file():
        return load_profile(p)
    return get_profile(str(name_or_path))


def check_stress_compat(
    profile: CtrlWorldProfile,
    *,
    camera_map: dict[str, str],
    concat_cameras: Optional[dict[str, str]],
    n_actions: int,
    control_hz: float,
) -> dict[str, str]:
    """Validate a stress-run configuration against a profile's contract.

    Called by the dream driver before any worker spawns, so a mismatch fails
    as loudly and as early as the host-side config checks do for the Cosmos
    backend — the region vocabulary just lives with the checkpoint profile
    instead of in core. Returns the resolved region -> episode-camera mapping
    (the profile default when ``concat_cameras`` is None).
    """
    regions = set(profile.views)
    bad = {r for r in camera_map.values() if r not in regions}
    if bad:
        raise ValueError(
            f"stress.camera_map regions {sorted(bad)} are not views of ctrl-world "
            f"profile '{profile.name}'; valid views: {sorted(regions)}."
        )
    if concat_cameras is None:
        if not profile.default_region_cameras:
            raise ValueError(
                f"stress.concat_cameras is unset and ctrl-world profile "
                f"'{profile.name}' declares no default_region_cameras; map each "
                f"view ({sorted(regions)}) to an episode camera role explicitly."
            )
        concat_cameras = dict(profile.default_region_cameras)
    if set(concat_cameras) != regions:
        raise ValueError(
            f"stress.concat_cameras must map exactly the profile's views "
            f"{sorted(regions)} to episode camera roles; got keys "
            f"{sorted(concat_cameras)}."
        )
    if n_actions < 1 or n_actions % profile.frames_per_chunk != 0:
        raise ValueError(
            f"stress.n_actions={n_actions}: ctrl-world profile '{profile.name}' "
            f"dreams in chunks of {profile.frames_per_chunk} future frames; "
            "n_actions must be a positive multiple of it."
        )
    stride = control_hz / profile.native_fps
    if abs(stride - round(stride)) > 1e-9 or round(stride) < 1:
        raise ValueError(
            f"stress.control_hz ({control_hz:g}) must be a positive integer "
            f"multiple of profile '{profile.name}''s {profile.native_fps:g} Hz "
            "native rate."
        )
    return concat_cameras


__all__ = [
    "ACTION_DIM",
    "CtrlWorldProfile",
    "check_stress_compat",
    "get_profile",
    "load_profile",
    "resolve_profile",
]
