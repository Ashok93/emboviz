"""Core data types — the lingua franca every other module speaks.

These types intentionally avoid hard dependencies on heavy libraries at the
type level (we use ``Any``/``ndarray`` rather than torch tensors so this file
is import-safe before torch is installed). Adapters convert their native
representations into these types at the protocol boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np

from emboviz.core.observations import (
    ActionHistory,
    DepthMap,
    ForceTorque,
    GripperState,
    Proprioception,
    RGBImage,
    TactileReading,
)

if TYPE_CHECKING:
    from emboviz.core.profile import RobotProfile

# Sentinel for "PIL image" without importing PIL here — we accept anything
# that has a `.size` and is convertible via numpy.asarray, but adapters do
# the actual loading.
ImageLike = Any


@dataclass(frozen=True)
class Observations:
    """The full sensor payload at one timestep.

    `images` is a dict from day one: single-camera setups populate
    `{"primary": ...}`, multi-camera setups add more keys (e.g.
    `"wrist_left"`, `"head"`). The `"primary"` key is the convention that
    single-cam-aware diagnostics use.

    All other fields are optional. A model that doesn't consume state can
    receive a Scene whose `state` is None; the runtime validator (see
    `VLAModel.required_inputs`) checks that what the model declares it
    needs is actually present.

    Experimental sensors that haven't earned a first-class slot yet live
    in `extras` — promote to a typed field once ≥2 adapters use them.
    """

    images: dict[str, RGBImage]
    state: Optional[Proprioception] = None
    gripper: Optional[GripperState] = None
    action_history: Optional[ActionHistory] = None
    depth: Optional[dict[str, DepthMap]] = None
    force_torque: Optional[ForceTorque] = None
    tactile: Optional[TactileReading] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_image(self) -> RGBImage:
        """The default `"primary"` camera. Raises KeyError if not present."""
        return self.images["primary"]


@dataclass(frozen=True)
class Scene:
    """One observation point — everything fed to the policy at one timestep.

    Built around a typed `Observations` bag so multi-camera, proprio,
    gripper, and action history are first-class. Single-camera + text-only
    callers should use `Scene.from_image(image, instruction)` for the
    common case.

    A trajectory is a list of Scenes.
    """

    observations: Observations
    instruction: Optional[str] = None
    profile: Optional["RobotProfile"] = None
    metadata: dict = field(default_factory=dict)
    scene_id: str = ""

    @property
    def primary_image_data(self) -> Any:
        """PIL image of the primary camera — the most common access path."""
        return self.observations.primary_image.data

    @classmethod
    def from_image(
        cls,
        image: ImageLike,
        instruction: Optional[str] = None,
        scene_id: str = "",
        metadata: Optional[dict] = None,
        profile: Optional["RobotProfile"] = None,
    ) -> "Scene":
        """Convenience constructor for the single-cam, text-only case."""
        obs = Observations(images={"primary": RGBImage(data=image)})
        return cls(
            observations=obs,
            instruction=instruction,
            profile=profile,
            scene_id=scene_id,
            metadata=metadata or {},
        )

    def with_image(self, new_image: ImageLike, camera: str = "primary") -> "Scene":
        """Return a new Scene with the named camera's image replaced.

        Other modalities (state, gripper, action_history, etc.) and other
        cameras are preserved. Used by diagnostics that need to swap one
        camera's content (occlusion, sensitivity map, memorization mask).
        """
        from dataclasses import replace
        new_images = dict(self.observations.images)
        new_images[camera] = RGBImage(data=new_image, camera_id=camera)
        new_obs = replace(self.observations, images=new_images)
        return replace(self, observations=new_obs)

    def with_instruction(self, new_instruction: str) -> "Scene":
        """Return a new Scene with the instruction replaced.

        All observations are preserved. Used by diagnostics that vary the
        text input while holding the visual/state context constant
        (cross-modal attention, ad-hoc instruction probes).
        """
        from dataclasses import replace
        return replace(self, instruction=new_instruction)


@dataclass
class ActionResult:
    """The output of one VLA inference call, in a model-agnostic shape.

    Adapters fill `action` with a continuous numpy vector regardless of the
    model's internal representation (discrete tokens, flow-matching, diffusion);
    they may also populate the optional fields for richer diagnostics.
    """

    action: np.ndarray                                  # (action_dim,) continuous
    action_dim: int = 0                                 # informational
    action_tokens: Optional[Any] = None                 # discrete tokens if any
    action_distribution: Optional[Any] = None           # logits if available
    confidence: Optional[float] = None                  # adapter-defined scalar
    metadata: dict = field(default_factory=dict)


@dataclass
class TokenSelector:
    """How to pick a query position when extracting attention or hidden states.

    Use exactly one of:
      • `position`  — absolute index into the LLM sequence
      • `relative` — "last", "first", "before_action"
      • `word`     — the substring whose first token position to use; the
        adapter resolves this via its tokenizer
    """

    position: Optional[int] = None
    relative: Optional[Literal["last", "first", "before_action"]] = None
    word: Optional[str] = None

    def __post_init__(self):
        provided = sum(x is not None for x in (self.position, self.relative, self.word))
        if provided != 1:
            raise ValueError(
                "TokenSelector requires exactly one of: position, relative, word"
            )


@dataclass
class AttentionMaps:
    """Attention from a query position to all key positions, per layer/head.

    All attention is in a single tensor of shape (n_layers, n_heads, n_keys)
    — already projected so the query is fixed. Image-token positions are
    indicated by `image_token_range` so callers can slice and reshape.
    """

    weights: np.ndarray                          # (n_layers, n_heads, n_keys)
    query_position: int
    n_keys: int
    image_token_range: tuple[int, int]           # (start, end) exclusive
    image_grid_side: int                         # n_image_tokens = side*side
    layer_indices: Optional[list[int]] = None    # which layers (None = all)
    metadata: dict = field(default_factory=dict)

    def image_weights(self) -> np.ndarray:
        """Slice and reshape the attention to (n_layers, n_heads, side, side)."""
        s, e = self.image_token_range
        img = self.weights[..., s:e]
        side = self.image_grid_side
        n_image = side * side
        if img.shape[-1] >= n_image:
            img = img[..., :n_image]
        elif img.shape[-1] < n_image:
            pad = np.zeros(img.shape[:-1] + (n_image - img.shape[-1],), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=-1)
        return img.reshape(*img.shape[:-1], side, side)


@dataclass
class HiddenStates:
    """Hidden-state vectors at a query position, sampled at requested layers."""

    states: np.ndarray                # (n_layers, hidden_dim)
    query_position: int
    layer_indices: list[int]
    hidden_dim: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class FFNActivations:
    """Pre-down_proj FFN activations at a query position, per layer.

    Keyed by layer index; each value is (intermediate_dim,) — the activations
    that get multiplied by `down_proj.weight` to form the FFN output.
    This is the surface mechanistic-interp papers operate on.
    """

    by_layer: dict[int, np.ndarray]
    query_position: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Perturbation result types
# ---------------------------------------------------------------------------


@dataclass
class PerturbedScene:
    """A scene produced by a Perturber, with provenance attached."""

    scene: Scene                            # the perturbed input
    perturber_name: str                     # what produced it
    axis: str                               # category, e.g. "noun_swap"
    variant_id: str                         # unique within a Perturber run
    parameters: dict = field(default_factory=dict)
    description: str = ""                   # human-readable label for reports


# ---------------------------------------------------------------------------
# Trajectory: a temporal sequence of Scenes
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A sequence of Scenes from one rollout / episode.

    `frames` is the canonical time-ordered list. `frame_indices` maps each
    entry to its index in the *original* dataset episode (useful when
    subsampling — frame_indices[i] is the dataset frame number for
    frames[i]).

    Trajectories are read-only after construction. Use `subsample()` or
    `slice()` to produce derivatives.
    """

    frames: list[Scene]
    frame_indices: list[int] = field(default_factory=list)
    fps: float = 0.0
    episode_id: str = ""
    source: str = ""                        # e.g. "bridge:0"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.frame_indices:
            self.frame_indices = list(range(len(self.frames)))
        if len(self.frame_indices) != len(self.frames):
            raise ValueError("frames and frame_indices must have the same length")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Trajectory(
                frames=self.frames[i],
                frame_indices=self.frame_indices[i],
                fps=self.fps,
                episode_id=self.episode_id,
                source=self.source,
                metadata=self.metadata,
            )
        return self.frames[i]

    def subsample(self, stride: int) -> "Trajectory":
        """Keep every `stride`-th frame, preserving frame_indices."""
        if stride <= 1:
            return self
        return Trajectory(
            frames=self.frames[::stride],
            frame_indices=self.frame_indices[::stride],
            fps=self.fps / stride,
            episode_id=self.episode_id,
            source=self.source,
            metadata=self.metadata,
        )
