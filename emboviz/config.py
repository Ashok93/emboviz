"""The single emboviz run config — one file drives one ``emboviz analyze``.

A run config declares EVERYTHING for a run: the model (adapter + the
user's checkpoint kwargs), the dataset mapping (format + path + the
camera-role / state-convention / gripper bindings the format can't
encode), and the analysis parameters (episodes, memorization target,
diagnostics, output). There is no CLI flag soup — `emboviz analyze
--config run.yaml` reads it all from here.

The schema is identical for every dataset ``format`` (lerobot / gr00t /
hdf5 / rlds — the self-describing "saved episode" formats): only the
*reader* behind each ``key`` changes, not what the user writes. Things
the formats never encode — the state convention, the camera-role→source-key
binding, the gripper spec — are always declared here, the same way
regardless of format. (Rerun/MCAP are recording / debugging-viz formats,
not dataset inputs.)

Shipped templates live under ``configs/`` (one per supported model/task);
users copy and edit them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Valid values mirror emboviz_wire's StateConvention / GripperKind /
# GripperUnits literals. We re-declare them here (rather than import the
# wire types) so the host can validate a config without the wire package
# being importable — and we fail loud on a typo'd convention.
_STATE_CONVENTIONS = {
    "joint_angles", "joint_velocities", "joint_torques",
    "ee_pose", "ee_delta", "ee_velocity",
}
_GRIPPER_KINDS = {"parallel_jaw", "suction", "binary", "magnetic", "multi_finger"}
_GRIPPER_UNITS = {"unit", "m", "mm", "rad", "binary"}
_DATASET_FORMATS = {"lerobot", "gr00t", "hdf5", "rlds"}
# Memorization mask-fill ensemble (mirrors emboviz.diagnostics.memorization's
# fill-mode names; re-declared here so the host can validate a config
# without importing the diagnostic). 'lama_inpaint' is the on-manifold fill
# and pulls in the emboviz-lama worker.
_FILL_MODES = {"channel_mean", "gaussian_blur", "lama_inpaint"}


class _Strict(BaseModel):
    """Reject unknown keys so a typo'd field fails loud, not silently."""
    model_config = ConfigDict(extra="forbid")


class ModelCfg(_Strict):
    adapter: str                                  # installed emboviz adapter name (openvla/oft/pi0/gr00t)
    kwargs: dict[str, Any] = Field(default_factory=dict)   # constructor overrides → the user's checkpoint


class StateCfg(_Strict):
    key: str                                      # source key for the proprioception vector
    convention: str                               # joint_angles | ee_pose | ... — the format never encodes this

    @field_validator("convention")
    @classmethod
    def _check_convention(cls, v: str) -> str:
        if v not in _STATE_CONVENTIONS:
            raise ValueError(
                f"state.convention={v!r} is not one of {sorted(_STATE_CONVENTIONS)}. "
                "We refuse to guess joint-angles vs ee-pose — state it explicitly."
            )
        return v


class ActionCfg(_Strict):
    key: str                                      # source key for the action vector


class GripperCfg(_Strict):
    # Where the gripper scalar comes from. Provide exactly one of:
    #   • ``source`` — index (or per-dim name) of the gripper WITHIN the state
    #     vector (datasets that pack the gripper into observation.state).
    #   • ``key``    — a SEPARATE dataset feature key carrying the gripper on its
    #     own (e.g. DROID's ``observation.state.gripper_position``), used when
    #     the state vector declared by ``state.key`` does not contain it.
    # Both omitted is valid only for the ``gr00t`` reader, which derives the
    # gripper index from the dataset's own meta/modality.json. Every other
    # reader requires one and raises clearly if neither is given.
    source: Optional[Union[int, str]] = None
    key: Optional[str] = None
    kind: str = "parallel_jaw"
    units: str = "unit"
    range: tuple[float, float] = (0.0, 1.0)

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in _GRIPPER_KINDS:
            raise ValueError(f"gripper.kind={v!r} not in {sorted(_GRIPPER_KINDS)}")
        return v

    @field_validator("units")
    @classmethod
    def _check_units(cls, v: str) -> str:
        if v not in _GRIPPER_UNITS:
            raise ValueError(f"gripper.units={v!r} not in {sorted(_GRIPPER_UNITS)}")
        return v

    @model_validator(mode="after")
    def _check_source_xor_key(self) -> "GripperCfg":
        if self.source is not None and self.key is not None:
            raise ValueError(
                "dataset.gripper: set EITHER `source` (the gripper's index within "
                "the state vector) OR `key` (a separate gripper feature key), not "
                "both — they are two different ways to locate the same scalar."
            )
        return self


class InstructionCfg(_Strict):
    # One of:
    #   from: tasks          → read the dataset's task table (lerobot)
    #   key: <source key>    → per-step instruction field (rlds, e.g.
    #                          "language_instruction")
    #   text: "<literal>"    → a fixed instruction string (hdf5, which
    #                          carries no task metadata)
    from_: Optional[str] = Field(default=None, alias="from")
    key: Optional[str] = None
    text: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetCfg(_Strict):
    format: str                                   # lerobot | gr00t | hdf5 | rlds
    path: str                                     # the dataset's identity: LeRobot HF repo id / local dir,
                                                  # HDF5 file path, or RLDS TFDS builder name
    # model logical camera role -> this dataset's actual image/source key.
    cameras: dict[str, str] = Field(default_factory=dict)
    state: Optional[StateCfg] = None
    action: Optional[ActionCfg] = None
    gripper: Optional[GripperCfg] = None
    instruction: Optional[InstructionCfg] = None
    # Format-specific reader knobs the common fields don't cover
    # (HDF5 demo_group; RLDS data_dir + split).
    extra: dict[str, Any] = Field(default_factory=dict)
    n_episodes: Optional[int] = None

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in _DATASET_FORMATS:
            raise ValueError(f"dataset.format={v!r} not in {sorted(_DATASET_FORMATS)}")
        return v


class AnalysisCfg(_Strict):
    episodes: str = "0"                           # "7" / "0,3,7" / "0-5" / "all"
    frame_start: int = 0
    n_frames: int = -1                            # -1 = all frames from frame_start
    frame_stride: int = 1
    mask_query: str = ""                          # memorization target phrase (single — one episode, one mask)
    target_annotations: Optional[str] = None      # per-frame bbox/mask file — replaces text detection when set
    detector: str = "sam3"                        # sam3 | gd-sam
    # Target-detection thresholds (SAM 3 / GD-SAM). Default to SAM 3's
    # recommended 0.5 / 0.5 (the value used throughout the transformers SAM 3
    # docs). If a target is faint / small / partially-occluded and gets missed
    # on a camera, lower detector_score_threshold rather than guessing — a
    # missed detection on a REQUIRED camera (see memorization_require_cameras)
    # drops that frame. Phrase the mask_query neutrally (color words can throw
    # the detector off) before reaching for a lower threshold.
    detector_score_threshold: float = 0.5         # min detection confidence to keep a detection
    detector_mask_threshold: float = 0.5          # per-pixel mask-logit cutoff (SAM 3)
    # Views that must carry a detection for a memorization frame to be scored.
    # "primary" (default) gates on the main scene view — a wrist camera often
    # cannot see scene objects; "all" requires every camera; a list names roles.
    memorization_require_cameras: Union[str, list[str]] = "primary"
    # memorization mask-fill ensemble. Default = the two OOD-leaning pure
    # fills (no worker). Add 'lama_inpaint' for the on-manifold fill (needs
    # the emboviz-lama worker) so the agreement gate spans the on-manifold/
    # OOD axis the literature prescribes (LITERATURE.md §1).
    fills: list[str] = Field(default_factory=lambda: ["channel_mean", "gaussian_blur"])
    diagnostics: Union[str, list[str]] = "all"
    sensitivity_grid_side: int = 4
    modality_pool_size: int = 20
    modality_k_samples: int = 10
    modality_pool_seed: int = 0
    modality_pool_cache_dir: Optional[str] = None # optional on-disk cache for the SHAP-marginal pool
    show_imitation: bool = False

    @field_validator("detector")
    @classmethod
    def _check_detector(cls, v: str) -> str:
        if v not in {"sam3", "gd-sam"}:
            raise ValueError(
                f"analysis.detector={v!r} not in {{'sam3', 'gd-sam'}}"
            )
        return v

    @field_validator("detector_score_threshold", "detector_mask_threshold")
    @classmethod
    def _check_detector_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"detector threshold must be in [0, 1]; got {v}."
            )
        return v

    @field_validator("fills")
    @classmethod
    def _check_fills(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "analysis.fills must list at least one fill mode "
                f"(from {sorted(_FILL_MODES)})."
            )
        bad = [f for f in v if f not in _FILL_MODES]
        if bad:
            raise ValueError(
                f"analysis.fills has unknown mode(s) {bad}; "
                f"supported: {sorted(_FILL_MODES)}."
            )
        return v

    @field_validator("modality_k_samples")
    @classmethod
    def _check_modality_k(cls, v: int) -> int:
        # The SHAP-marginal modality-dropout estimate has Monte-Carlo error
        # ~1/sqrt(K). Below K=10 the per-modality response is noise dressed
        # up as a number — a verdict we will not stand behind — so we refuse
        # it loudly rather than silently emit it (this is a forbidden
        # "de-risk" shortcut, per the honesty rule).
        if v < 10:
            raise ValueError(
                f"analysis.modality_k_samples={v} is below the statistical "
                "floor of 10. Modality dropout averages K real substitutions "
                "per modality; K<10 gives a Monte-Carlo estimate too noisy to "
                "trust. Use modality_k_samples >= 10 (default 10)."
            )
        return v


class RunConfig(_Strict):
    model: ModelCfg
    dataset: DatasetCfg
    analysis: AnalysisCfg = Field(default_factory=AnalysisCfg)
    output: str

    def diagnostics_str(self) -> str:
        """Normalize ``analysis.diagnostics`` to the comma string the CLI
        resolver expects (it accepts ``"all"`` / ``"a,b,c"`` / ``"all,-x"``)."""
        d = self.analysis.diagnostics
        return d if isinstance(d, str) else ",".join(d)

    def dataset_build_kwargs(self) -> dict[str, Any]:
        """The kwargs for :func:`emboviz.datasets.manifest.build_source` —
        the ``dataset`` section serialized to plain JSON-able dicts. The
        keys match ``build_source``'s signature exactly (format, path,
        cameras, state, action, gripper, instruction, extra, n_episodes)."""
        return self.dataset.model_dump(by_alias=True)


# ── shipped-template resolution ──────────────────────────────────────

def _configs_dir() -> Path:
    """Repo-root ``configs/`` holding the shipped templates."""
    # emboviz/config.py → repo root is two parents up (emboviz/ then root).
    return Path(__file__).resolve().parent.parent / "configs"


def load_run_config(name_or_path: str) -> RunConfig:
    """Load + validate a RunConfig from a YAML file path OR a shipped
    template name (e.g. ``"pi0"`` resolves to
    ``configs/pi0.yaml``).

    Raises a clear error on unknown keys, bad enum values, or a missing
    file — never silently coerces.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Reading a run config needs PyYAML (an emboviz core dep). "
            f"Underlying error: {e}"
        ) from e

    p = Path(name_or_path)
    if not p.exists():
        # treat as a shipped-template name
        candidate = _configs_dir() / f"{name_or_path}.yaml"
        if candidate.exists():
            p = candidate
        else:
            shipped = sorted(
                f.stem for f in _configs_dir().glob("*.yaml")
            ) if _configs_dir().exists() else []
            raise FileNotFoundError(
                f"run config {name_or_path!r} is neither an existing file "
                f"nor a shipped template. Shipped templates: {shipped}. "
                f"Pass a path to your own .yaml or one of those names."
            )

    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"run config {p} did not parse to a mapping")
    return RunConfig.model_validate(raw)
