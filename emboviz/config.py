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


# Cartesian conventions track an end-effector pose; joint conventions track a
# joint vector and forward-kinematics it (so they require a robot). Kept in sync
# with emboviz_wire.policy_bridge — core does not import the adapter packages.
_CARTESIAN_ACTION_CONVENTIONS = {"absolute_xyz_euler", "delta_xyz_euler_base"}
_JOINT_ACTION_CONVENTIONS = {"droid_joint_velocity"}
_ACTION_CONVENTIONS = _CARTESIAN_ACTION_CONVENTIONS | _JOINT_ACTION_CONVENTIONS

_WORLD_MODELS = {"cosmos3", "ctrlworld"}
# Cosmos's regions of the stitched conditioning frame are fixed by its DROID
# ``concat_view`` geometry (wrist on top, exteriors below) — mirrors
# emboviz_cosmos3.concat_view, re-declared here (like _STATE_CONVENTIONS) so a
# config validates without the adapter installed. Ctrl-World's regions depend
# on the selected checkpoint *profile* (stress.profile), so its region /
# chunk-quantum / rate checks live with the profile and run in the dream
# driver (emboviz_ctrlworld.profiles.check_stress_compat) — still before any
# worker spawns.
_COSMOS_REGIONS = {"wrist", "exterior_left", "exterior_right"}
_COSMOS_DEFAULT_REGION_CAMERAS = {
    "wrist": "wrist", "exterior_left": "primary", "exterior_right": "exterior_2",
}


class SceneSwapCfg(_Strict):
    """Masked counterfactual object swap for the closed-loop dream.

    SAM 3 localizes ``mask_query`` independently in every concat camera; the
    masked region is then either replaced with ``replace_query`` (Stable
    Diffusion text-guided inpainting, the ``sd-inpaint`` adapter) or — when
    ``replace_query`` is empty — removed (LaMa fills it with plausible
    background). The dream is run from both the original seed and the edited seed
    and shown side by side, so the policy's behaviour under the counterfactual is
    judged against reality.

    A camera with no confident detection keeps its ORIGINAL image: the policy
    needs all concat cameras to drive the dream, so dropping a view is not an
    option. This is not a silent fallback — the per-camera status (which views
    were edited, which kept their original, and why) is recorded on the clip, so
    a partial swap (e.g. wrist only) is never mistaken for a full one. When no
    camera detects the target at all, the swap column is left empty and the clip
    marks it as "not run".
    """

    mask_query: str                               # SAM 3 phrase: the object to locate (e.g. "the marker")
    replace_query: str = ""                        # object to paint in its place (SD inpaint); empty -> remove (LaMa)
    # Grow the detected mask by this many pixels before editing. A tight mask
    # around a thin object leaves the inserter no room to paint a different shape
    # (a spoon into a marker's silhouette); a margin gives it space. 0 = off.
    mask_dilation: int = 0
    # SD-inpaint insertion controls (used only when replace_query is set). SD
    # inpainting defaults to harmonizing the masked region with its surroundings
    # (erasing the object); a high guidance plus a negative prompt forbidding the
    # empty background forces it to actually paint replace_query instead.
    edit_guidance_scale: float = 7.5              # higher = obey replace_query harder
    edit_negative_prompt: str = ""                # e.g. "empty cup, nothing" to prevent erasure
    # Detection thresholds (SAM 3). Defaults are SAM 3's recommended 0.5/0.5. A
    # target seen only from the close wrist view but missed by the distant
    # exteriors is expected; lower these (or phrase mask_query more neutrally)
    # rather than masking a region the detector is not confident about.
    detector_score_threshold: float = 0.5         # min detection confidence to keep a detection
    detector_mask_threshold: float = 0.5          # per-pixel mask-logit cutoff

    @field_validator("mask_query")
    @classmethod
    def _check_mask_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "stress.scene_swap.mask_query must be a non-empty phrase "
                "naming the object to locate (e.g. \"the marker\"). The swap "
                "refuses to guess the target."
            )
        return v.strip()

    @field_validator("detector_score_threshold", "detector_mask_threshold")
    @classmethod
    def _check_thresholds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"stress.scene_swap detector thresholds must be in [0, 1], got {v}."
            )
        return v

    @field_validator("mask_dilation")
    @classmethod
    def _check_dilation(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"stress.scene_swap.mask_dilation must be >= 0, got {v}.")
        return v


class WorldStressCfg(_Strict):
    """Critical-moment world-model stress test (the closed-loop simulator).

    Find the episode's decisive instants, optionally perturb each seed frame,
    then run the user's policy inside the world model step by step and judge
    the outcome. ``world_model`` selects the simulator backend:

      * ``cosmos3`` — NVIDIA Cosmos3-Nano forward dynamics via a vLLM-Omni
        server (``server_url`` required). Single-frame conditioning; faithful
        for roughly one or two re-conditioning cycles.
      * ``ctrlworld`` — Ctrl-World (ICLR 2026), run locally on the GPU.
        Multi-view joint prediction + pose-anchored sparse history; coherent
        over tens of seconds. The embodiment, camera views, rates, and chunk
        quantum come from the checkpoint ``profile`` ("droid" ships).

    Two editing modes are mutually exclusive:

      * ``perturbations`` — whole-frame instruction edits (one separate clip per
        instruction). Cosmos-only: the edit is rendered by the Cosmos server.
      * ``scene_swap`` — masked counterfactual object edit (SAM 3 + LaMa
        removal, or SD-inpaint replacement on cosmos3), rendered as a
        baseline-vs-swap side-by-side in ONE clip.
    """

    world_model: str = "cosmos3"                  # cosmos3 | ctrlworld
    server_url: Optional[str] = None              # vLLM-Omni Cosmos server; required for cosmos3
    domain: str = "droid_lerobot"                 # cosmos3 only (ctrlworld embodiment comes from its profile)
    action_dim: int = 10                          # cosmos3 only (ctrlworld is 7-fixed)
    # Ctrl-World checkpoint profile: a preconfigured name ("droid") or the path
    # to a profile JSON (see emboviz_ctrlworld.profiles). ctrlworld only.
    profile: str = "droid"
    # Policy under test. None -> recorded-action faithfulness baseline (no policy).
    policy_adapter: Optional[str] = None
    policy_kwargs: dict[str, Any] = Field(default_factory=dict)  # adapter constructor kwargs (e.g. {config_name: pi0_droid})
    action_convention: Optional[str] = None       # required when policy_adapter is set
    # Robot for joint-space action conventions (forward kinematics: joints -> EE
    # pose). Give EITHER a preconfigured catalog name (``robot: franka_panda``) OR
    # a custom URDF triple (``robot_urdf`` + ``robot_ee_frame`` + ``robot_joint_names``).
    # Required for joint conventions, forbidden for cartesian ones.
    robot: Optional[str] = None
    robot_urdf: Optional[str] = None
    robot_ee_frame: Optional[str] = None
    robot_joint_names: Optional[list[str]] = None
    # policy camera role -> stitched-frame region (e.g. {"primary": "exterior_left",
    # "wrist_left": "wrist"}). Valid regions depend on world_model — see
    # _WORLD_MODEL_REGIONS.
    camera_map: dict[str, str] = Field(default_factory=dict)
    # stitched-frame region -> the episode's camera role used to build the seed.
    # None resolves to the world model's default mapping (_DEFAULT_REGION_CAMERAS).
    concat_cameras: Optional[dict[str, str]] = None
    # Wrist-panel size (H, W) the seed concat is built at — sets the world model's
    # conditioning resolution. The Cosmos DROID domain trained on 640x360 (W x H)
    # per camera (a 360 px wrist -> 540x640 concat); feeding less puts the model
    # off-distribution and the dream blurs. None keeps the cameras' native size.
    # cosmos3 only: ctrlworld's view size is fixed by its checkpoint (320x192).
    concat_resolution: Optional[tuple[int, int]] = None
    perturbations: list[str] = Field(default_factory=list)  # whole-frame edit instructions; empty -> none
    # Masked counterfactual object swap (baseline-vs-swap side-by-side). Mutually
    # exclusive with ``perturbations``.
    scene_swap: Optional[SceneSwapCfg] = None
    n_loop_steps: int = 2                         # closed-loop turns (small — drifts after the first turn or two)
    n_actions: int = 16                           # prediction horizon: frames dreamed per turn (one Cosmos chunk)
    # Execution horizon: dreamed frames committed before the policy re-plans
    # (receding horizon). None -> commit the whole chunk. 1 is most reactive: the
    # policy re-decides on the next dreamed frame, so the policy (not the dream)
    # drives the rollout. Must satisfy 1 <= execute_steps <= n_actions.
    execute_steps: Optional[int] = None
    lead_s: float = 0.5                           # seconds before each keyframe to seed
    # Policy control rate (Hz) for joint-velocity conventions: joint configs advance
    # by velocity * (1/control_hz) per step. π0-DROID runs at 15 Hz (openpi
    # DROID_CONTROL_FREQUENCY). Unused by cartesian conventions.
    control_hz: float = 15.0
    conditioning_camera: str = "primary"
    state_convention: str = "ee_pose"
    reasoner_url: Optional[str] = None            # reasoner server; None -> no verdict
    reasoner_question: str = (
        "Did the robot successfully grasp and lift the target object? "
        "Answer in one sentence, and if it failed, say exactly how."
    )

    @field_validator("world_model")
    @classmethod
    def _check_world_model(cls, v: str) -> str:
        if v not in _WORLD_MODELS:
            raise ValueError(
                f"stress.world_model={v!r} not in {sorted(_WORLD_MODELS)}."
            )
        return v

    @field_validator("action_convention")
    @classmethod
    def _check_convention(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ACTION_CONVENTIONS:
            raise ValueError(
                f"stress.action_convention={v!r} not in {sorted(_ACTION_CONVENTIONS)}."
            )
        return v

    @field_validator("n_loop_steps", "n_actions")
    @classmethod
    def _check_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"stress: n_loop_steps / n_actions must be >= 1, got {v}.")
        return v

    @field_validator("concat_resolution")
    @classmethod
    def _check_concat_resolution(cls, v: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        if v is not None and (len(v) != 2 or v[0] < 2 or v[1] < 2):
            raise ValueError(
                f"stress.concat_resolution must be (H, W) with each >= 2, got {v}."
            )
        return v

    @field_validator("control_hz")
    @classmethod
    def _check_control_hz(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"stress.control_hz must be > 0, got {v}.")
        return v

    @model_validator(mode="after")
    def _check_backend(self) -> "WorldStressCfg":
        """Backend-conditional checks: required/forbidden fields and the region
        vocabulary depend on which world model drives the loop.

        cosmos3's region vocabulary is fixed by its concat geometry and is
        checked here. ctrlworld's depends on the selected checkpoint profile,
        so its region / chunk-quantum / control-rate checks run in the dream
        driver via ``emboviz_ctrlworld.profiles.check_stress_compat`` —
        equally loud, still before any worker spawns."""
        if self.world_model == "cosmos3":
            if not self.server_url:
                raise ValueError(
                    "stress.world_model='cosmos3' needs stress.server_url (the "
                    "running vLLM-Omni Cosmos server). The ctrlworld backend runs "
                    "locally and needs no server."
                )
            if "profile" in self.model_fields_set:
                raise ValueError(
                    "stress.profile selects a ctrlworld checkpoint profile and "
                    "does not apply to the cosmos3 backend; remove it (cosmos's "
                    "embodiment is stress.domain)."
                )
            if self.concat_cameras is None:
                self.concat_cameras = dict(_COSMOS_DEFAULT_REGION_CAMERAS)
            if set(self.concat_cameras) != _COSMOS_REGIONS:
                raise ValueError(
                    f"stress.concat_cameras must map exactly {sorted(_COSMOS_REGIONS)} "
                    f"(the cosmos3 regions) to episode camera roles; got keys "
                    f"{sorted(self.concat_cameras)}."
                )
            bad = {r for r in self.camera_map.values() if r not in _COSMOS_REGIONS}
            if bad:
                raise ValueError(
                    f"stress.camera_map regions {sorted(bad)} invalid for "
                    f"world_model='cosmos3'; valid regions: {sorted(_COSMOS_REGIONS)}."
                )
        else:  # ctrlworld
            forbidden = {"server_url", "domain", "action_dim", "concat_resolution"}
            set_anyway = [
                f for f in sorted(forbidden & self.model_fields_set)
                if getattr(self, f) is not None
            ]
            if set_anyway:
                raise ValueError(
                    f"stress.{set_anyway[0]} applies only to the cosmos3 backend "
                    "(ctrlworld runs locally; its embodiment, action encoding, and "
                    "view sizes come from stress.profile); remove it."
                )
            if self.perturbations:
                raise ValueError(
                    "stress.perturbations (whole-frame instruction edits) are "
                    "rendered by the Cosmos server and are not available on the "
                    "ctrlworld backend; use scene_swap with an empty replace_query "
                    "(SAM 3 + LaMa removal) instead."
                )
            if self.scene_swap is not None and self.scene_swap.replace_query:
                raise ValueError(
                    "stress.scene_swap.replace_query (SD-inpaint object insertion) "
                    "is not wired for the ctrlworld backend; leave it empty to "
                    "remove the object (SAM 3 + LaMa)."
                )
        return self

    @model_validator(mode="after")
    def _check_execute_steps(self) -> "WorldStressCfg":
        if self.execute_steps is not None and not 1 <= self.execute_steps <= self.n_actions:
            raise ValueError(
                "stress.execute_steps must satisfy 1 <= execute_steps <= "
                f"n_actions ({self.n_actions}); got {self.execute_steps}."
            )
        return self

    @model_validator(mode="after")
    def _check_edit_modes_exclusive(self) -> "WorldStressCfg":
        if self.scene_swap is not None and self.perturbations:
            raise ValueError(
                "stress: set EITHER `perturbations` (whole-frame instruction "
                "edits, one clip each) OR `scene_swap` (masked baseline-vs-swap "
                "comparison), not both — they are different editing modes and "
                "combining them is ambiguous."
            )
        return self

    @model_validator(mode="after")
    def _check_policy_requirements(self) -> "WorldStressCfg":
        if self.policy_adapter is not None:
            if self.action_convention is None:
                raise ValueError(
                    "stress.policy_adapter is set, so action_convention is required "
                    f"(one of {sorted(_ACTION_CONVENTIONS)})."
                )
            if not self.camera_map:
                raise ValueError(
                    "stress.policy_adapter is set, so camera_map is required "
                    "(policy camera role -> stitched-frame region)."
                )
        self._check_robot()
        return self

    def _check_robot(self) -> None:
        has_custom = any(
            x is not None for x in (self.robot_urdf, self.robot_ee_frame, self.robot_joint_names)
        )
        is_joint = self.action_convention in _JOINT_ACTION_CONVENTIONS

        if is_joint:
            if self.robot is None and not has_custom:
                raise ValueError(
                    f"stress.action_convention={self.action_convention!r} is "
                    "joint-space, so a robot is required for forward kinematics. Set "
                    "`robot` (a preconfigured name, e.g. franka_panda) or the custom "
                    "triple robot_urdf + robot_ee_frame + robot_joint_names."
                )
            if self.robot is not None and has_custom:
                raise ValueError(
                    "stress: set EITHER `robot` (preconfigured) OR the custom "
                    "robot_urdf/robot_ee_frame/robot_joint_names triple, not both."
                )
            if has_custom and not (
                self.robot_urdf and self.robot_ee_frame and self.robot_joint_names
            ):
                raise ValueError(
                    "stress: a custom robot needs all of robot_urdf, "
                    "robot_ee_frame, and robot_joint_names."
                )
        else:
            if self.robot is not None or has_custom:
                raise ValueError(
                    f"stress.action_convention={self.action_convention!r} is "
                    "cartesian and tracks the end-effector pose directly; remove the "
                    "robot / robot_urdf settings (they apply only to joint conventions)."
                )


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
    # Critical-moment world-model stress test (the closed-loop simulator).
    # Optional; only consumed by the stress driver, not the standard diagnostics.
    stress: Optional[WorldStressCfg] = None

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
