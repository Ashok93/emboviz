"""Core data types — the lingua franca every other module speaks.

These types intentionally avoid hard dependencies on heavy libraries at the
type level (we use ``Any``/``ndarray`` rather than torch tensors so this file
is import-safe before torch is installed). Adapters convert their native
representations into these types at the protocol boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from emboviz_wire.observations import (
    ActionHistory,
    DepthMap,
    ForceTorque,
    GripperState,
    Proprioception,
    RGBImage,
    TactileReading,
)

if TYPE_CHECKING:
    from emboviz_wire.profile import RobotProfile

# Sentinel for "PIL image" without importing PIL here — we accept anything
# that has a `.size` and is convertible via numpy.asarray, but adapters do
# the actual loading. Re-exported from ``emboviz_wire`` so adapters can
# annotate image arguments with it.
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

        Raises KeyError if ``camera`` is not already present in the scene —
        we never invent a new camera silently. To add a new camera, use
        ``with_images({...})`` and pass the full image dict.
        """
        if camera not in self.observations.images:
            raise KeyError(
                f"Camera '{camera}' is not in scene.observations.images "
                f"(available: {sorted(self.observations.images)}). "
                "with_image only replaces existing cameras."
            )
        from dataclasses import replace
        new_images = dict(self.observations.images)
        new_images[camera] = RGBImage(data=new_image, camera_id=camera)
        new_obs = replace(self.observations, images=new_images)
        return replace(self, observations=new_obs)

    def with_images(self, new_images_by_camera: dict[str, ImageLike]) -> "Scene":
        """Return a new Scene with multiple cameras replaced in one step.

        Each key must already be present in the scene — we do not invent
        new cameras. All other modalities and any cameras not in the dict
        are preserved.
        """
        existing = set(self.observations.images.keys())
        requested = set(new_images_by_camera.keys())
        missing = requested - existing
        if missing:
            raise KeyError(
                f"with_images: cameras {sorted(missing)} are not in the scene "
                f"(available: {sorted(existing)}). To replace an existing "
                "camera use this method; to add a new camera, build a new Scene."
            )
        from dataclasses import replace
        new_images = dict(self.observations.images)
        for cam, img in new_images_by_camera.items():
            new_images[cam] = RGBImage(data=img, camera_id=cam)
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


def resolve_cameras(scene: "Scene", requested: Optional[list[str]]) -> list[str]:
    """Resolve a (possibly-None) camera selection against a scene's cameras.

    - ``requested=None`` → return every camera in the scene (alphabetical).
    - ``requested=["primary", "wrist_left"]`` → return those exact cameras.
      Raises ValueError if any requested camera is missing from the scene.

    This is the single source of truth for "which cameras does this
    diagnostic / perturber operate on?". Callers must use it instead of
    silently defaulting to ``"primary"`` — that pattern silently makes
    multi-camera scenes look single-camera and hides real model behaviour.
    """
    available = set(scene.observations.images.keys())
    if requested is None:
        return sorted(available)
    requested_set = set(requested)
    missing = requested_set - available
    if missing:
        raise ValueError(
            f"resolve_cameras: requested cameras {sorted(missing)} are "
            f"not in the scene (available: {sorted(available)}). Either "
            "remove them from the cameras list, load the missing cameras "
            "in the dataset adapter, or pass cameras=None to iterate "
            "every camera the scene actually provides."
        )
    return sorted(requested_set)


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
    # Multi-step action chunk if the model predicts one (π0, OFT, ACT, GR00T,
    # diffusion policies). Shape (chunk_len, action_dim). The first row is
    # the immediate action — same as ``self.action`` — and subsequent rows
    # are the model's predicted future actions. Adapters that predict a
    # single action leave this None. ChunkConsistencyDiagnostic needs this
    # populated to do the real chunk-coherence test.
    action_chunk: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


def average_action_results(results: list[ActionResult]) -> ActionResult:
    """Average several ActionResults (samples of the SAME scene) into one.

    Stochastic policies (π0 flow-matching, GR00T diffusion) return a
    different action each call; the mean of N independent samples has
    sampling noise ``σ/√N``. This is the single averaging implementation
    shared by the host (``emboviz.calibration.averaged_predict``) and the
    in-worker ``VLAModel.predict_batch`` n-sample expansion, so both reduce
    noise identically.

    ``action`` and ``action_chunk`` are averaged. ``action_tokens`` and
    ``action_distribution`` are deliberately NOT carried onto the averaged
    result: discrete tokens / logits do not average meaningfully across
    stochastic samples. (Only OpenVLA populates ``action_tokens`` and it is
    deterministic, so it is never averaged with n>1 in practice.) A model
    that emits chunks of inconsistent shape across samples of one scene has a
    real bug (truncated / diverged decoding) — we raise rather than silently
    truncate, so a chunk-based diagnostic never runs on garbage.

    A single-element list is returned unchanged (no copy, no metadata
    mutation) — the deterministic / ``n_samples == 1`` fast path.
    """
    if not results:
        raise ValueError("average_action_results: empty results list")
    if len(results) == 1:
        return results[0]

    mean_action = np.stack(
        [np.asarray(r.action, dtype=np.float32) for r in results], axis=0
    ).mean(axis=0).astype(np.float32)

    chunks = [
        np.asarray(r.action_chunk, dtype=np.float32)
        for r in results if r.action_chunk is not None
    ]
    mean_chunk: Optional[np.ndarray] = None
    if chunks:
        shapes = {c.shape for c in chunks}
        if len(shapes) != 1:
            raise ValueError(
                f"average_action_results: action chunks have inconsistent "
                f"shapes across {len(chunks)} samples: {sorted(shapes)}. The "
                "same input produced different-shaped chunks — fix the model "
                "adapter (likely truncated decoding under noise) before "
                "trusting any chunk-based diagnostic on this model."
            )
        mean_chunk = np.stack(chunks, axis=0).mean(axis=0).astype(np.float32)

    last = results[-1]
    return ActionResult(
        action=mean_action,
        action_dim=last.action_dim if last.action_dim else int(mean_action.size),
        action_chunk=mean_chunk,
        confidence=last.confidence,
        metadata={**last.metadata, "n_samples_averaged": len(results)},
    )


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
    indicated per-camera so callers can slice and reshape one camera at a
    time. Multi-camera adapters (OpenVLA-OFT primary+wrist, GR00T per
    embodiment cameras, π0 ALOHA's 4-cam stack) populate one entry per
    camera; single-camera adapters (OpenVLA) populate just ``"primary"``.

    **Multi-tile / temporal cameras.** A single user-facing camera can
    contribute MULTIPLE contiguous runs of image tokens in the key
    sequence — e.g. GR00T's Qwen3-VL receives each camera as T temporal
    tiles when ``video_horizon > 1``. We model this honestly: each
    camera's ``image_token_ranges`` entry is a *list* of (start, end)
    runs, one per tile. ``image_weights(camera)`` then sums the per-tile
    reshapes into the per-camera attention map.

    Strict:
      • Every camera in ``image_token_ranges`` must also be in
        ``image_grid_sides`` (and vice versa).
      • Each tile's slice size must equal ``side * side`` (the side
        applies *per tile*; total tokens per camera = N_tiles * side²).
      • No silent padding.
    """

    weights: np.ndarray                                                # (n_layers, n_heads, n_keys)
    query_position: int
    n_keys: int
    image_token_ranges: dict[str, list[tuple[int, int]]]              # camera_id → list of tile (start, end) pairs (exclusive end)
    # Per-tile grid shape. Two equivalent ways to declare it; each camera
    # in image_token_ranges must be covered by exactly one:
    #   • image_grid_sides[cam] = side  → square grid, tile tokens = side*side
    #   • image_grid_shapes[cam] = (h, w) → rectangular grid (e.g. a CNN
    #     feature map on a non-square image), tile tokens = h*w
    image_grid_sides: dict[str, int] = field(default_factory=dict)
    image_grid_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    layer_indices: Optional[list[int]] = None                         # which layers (None = all)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        cams = set(self.image_token_ranges)
        covered = set(self.image_grid_sides) | set(self.image_grid_shapes)
        if cams != covered:
            raise ValueError(
                "AttentionMaps: every camera in image_token_ranges must have "
                "a grid in image_grid_sides or image_grid_shapes; got "
                f"ranges={sorted(cams)} vs grids={sorted(covered)}."
            )
        dup = set(self.image_grid_sides) & set(self.image_grid_shapes)
        if dup:
            raise ValueError(
                "AttentionMaps: cameras must declare their grid in exactly "
                f"one of image_grid_sides / image_grid_shapes; both given for "
                f"{sorted(dup)}."
            )
        for cam in cams:
            ranges = self.image_token_ranges[cam]
            h, w = self.grid_shape(cam)
            if not isinstance(ranges, list) or not ranges:
                raise ValueError(
                    f"AttentionMaps camera '{cam}': image_token_ranges["
                    f"'{cam}'] must be a non-empty list of (start, end) "
                    f"tile pairs; got {ranges!r}."
                )
            for s, e in ranges:
                if (e - s) != h * w:
                    raise ValueError(
                        f"AttentionMaps camera '{cam}': tile range {(s, e)} "
                        f"yields {e - s} tokens but grid {(h, w)} requires "
                        f"{h * w}. Adapter has the wrong range or grid."
                    )

    def grid_shape(self, camera: str) -> tuple[int, int]:
        """Per-tile (height, width) for a camera's image grid."""
        if camera in self.image_grid_shapes:
            h, w = self.image_grid_shapes[camera]
            return int(h), int(w)
        side = int(self.image_grid_sides[camera])
        return side, side

    @property
    def cameras(self) -> list[str]:
        """Sorted list of camera ids with attention available."""
        return sorted(self.image_token_ranges)

    def image_weights(self, camera: Optional[str] = None) -> np.ndarray:
        """Raw per-camera attention tensor (n_layers, n_heads, side, side).

        For multi-tile cameras (the same camera duplicated across a
        temporal stack), per-tile maps are SUMMED.

        This returns the unfiltered model output. For the user-facing
        "where is the model looking?" question, use
        :meth:`image_weights_clean` which applies the literature-backed
        defaults (mid-layer head filter + sink masking) so the heatmap
        isn't dominated by softmax routing artifacts.

        Raises:
            KeyError: if the camera was not declared.
            ValueError: if multiple cameras are declared and camera is None.
        """
        if camera is None:
            if len(self.image_token_ranges) > 1:
                raise ValueError(
                    "AttentionMaps.image_weights: this AttentionMaps has "
                    f"{len(self.image_token_ranges)} cameras "
                    f"({sorted(self.image_token_ranges)}); pass "
                    "``camera=<id>`` to pick one. We never silently "
                    "default to the first."
                )
            camera = next(iter(self.image_token_ranges))
        if camera not in self.image_token_ranges:
            raise KeyError(
                f"AttentionMaps has no attention for camera '{camera}'. "
                f"Available cameras: {sorted(self.image_token_ranges)}."
            )
        ranges = self.image_token_ranges[camera]
        h, w = self.grid_shape(camera)
        tile_maps = []
        for s, e in ranges:
            img = self.weights[..., s:e]
            tile_maps.append(img.reshape(*img.shape[:-1], h, w))
        if len(tile_maps) == 1:
            return tile_maps[0]
        return np.sum(tile_maps, axis=0)

    def image_weights_clean(
        self,
        camera: Optional[str] = None,
        *,
        layer_range_fraction: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Layer-adaptive per-camera attention heatmap — the user-facing
        "where is the model looking?" map.

        Method (layer-adaptive last-token attention, arXiv:2602.04304;
        "How Multimodal LLMs Solve Image Tasks", arXiv:2508.20279):

          1. **Restrict to a candidate layer range.** Average over heads
             within a fraction of the model's transformer layers
             (typically the middle band). Early layers do token-grouping
             (attention sinks dominate), late layers do prediction
             summarization, and the middle holds the visual-grounding
             heads. The exact fraction is per-backbone (LLaMA vs Gemma vs
             Qwen3-VL each have a different stage structure) and comes
             from the adapter's ``attention_profile``.

          2. **Pick the single best layer from the data.** Within that
             candidate range, select the one layer whose head-mean
             attention is most concentrated on the image INTERIOR (where
             objects are) rather than the border ring (where the
             positional / RoPE sink sits). This replaces per-cell sink
             masking: the grounding signal lives in one mid-stack layer,
             so we choose it by interior concentration instead of masking
             a fixed top-fraction of cells.

        Only ``layer_range_fraction`` is parameterised; it defaults to
        ``metadata["attention_profile"]["recommended_layer_range_fraction"]``
        populated by the adapter. Callers override only to test
        variations — the adapter's declared value IS the literature-backed
        recommendation.

        Args:
            camera: which camera (same semantics as ``image_weights``).
            layer_range_fraction: ``(frac_start, frac_end)`` in [0, 1]
                — fraction of total layers to use. ``None`` reads from
                the adapter's attention_profile.

        Returns:
            ``(heatmap, debug)`` where ``heatmap`` is the (side, side)
            cleaned attention map and ``debug`` records the selected
            layer, candidate range, interior concentration, and the
            adapter's literature citation.

        Raises:
            ValueError: if no ``recommended_layer_range_fraction`` is in
                metadata's attention_profile AND no override is passed —
                refusing to fabricate a layer range that isn't grounded
                in this model's literature.
        """
        raw = self.image_weights(camera)   # (L, H, gh, gw)
        L, H, gh, gw = raw.shape

        # Source per-model defaults from adapter's literature.
        profile = self.metadata.get("attention_profile", {})

        if layer_range_fraction is None:
            layer_range_fraction = profile.get("recommended_layer_range_fraction")
            if layer_range_fraction is None:
                raise ValueError(
                    "AttentionMaps.image_weights_clean: no "
                    "``recommended_layer_range_fraction`` in metadata's "
                    "attention_profile, and no override passed. Every "
                    "adapter must declare its model's recommended layer "
                    "range based on the model's literature — see each "
                    "adapter's ATTENTION_PROFILE for the template. "
                    "Refusing to fabricate a default."
                )

        frac_start, frac_end = layer_range_fraction
        if not (0.0 <= frac_start < frac_end <= 1.0):
            raise ValueError(
                f"image_weights_clean: layer_range_fraction "
                f"{layer_range_fraction} invalid; expected "
                "0.0 <= start < end <= 1.0."
            )

        start = int(round(frac_start * L))
        end = int(round(frac_end * L))
        if start == end:
            end = start + 1   # never zero layers
        end = min(end, L)
        start = max(0, min(start, L - 1))

        # ── Layer-adaptive "last-token attention" map ──
        # The last text token's attention to the image, mean over heads, at the
        # ONE layer where visual grounding is clearest. Per the layer-adaptive
        # localization literature (arXiv:2602.04304; "How Multimodal LLMs Solve
        # Image Tasks" arXiv:2508.20279) the grounding signal lives in a single
        # mid-stack layer, while attention sinks (special / border tokens)
        # dominate the other layers. So within the candidate layer range we
        # SELECT the layer whose attention is most concentrated on the image
        # INTERIOR (where objects are) rather than the border ring (where the
        # positional/RoPE sink sits), then average over heads. One layer chosen
        # from the data — no magic layer index, no per-cell sink removal, no
        # head selection, no calibration.
        per_layer = raw[start:end].mean(axis=1).astype(np.float64)   # (Lc, gh, gw)
        border = np.zeros((gh, gw), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        interior = ~border
        interior_frac = np.array([
            float(m[interior].sum() / s) if (s := float(m.sum())) > 1e-12 else 0.0
            for m in per_layer
        ])
        best = int(np.argmax(interior_frac))
        clean = per_layer[best]
        debug = {
            "selected_layer":       int(start + best),
            "candidate_layer_range": (int(start), int(end)),
            "layer_range_fraction": (float(frac_start), float(frac_end)),
            "interior_fraction":    float(interior_frac[best]),
            "n_layers_total":       int(L),
            "n_heads":              int(H),
            "method":               "layer-adaptive last-token attention (arXiv:2602.04304): "
                                    "pick the layer with max interior concentration, mean over heads",
            "profile_source":       profile.get("literature_citation", "unspecified"),
        }
        return clean, debug


@dataclass
class AttentionTrace:
    """Action→image cross-attention with the denoise-step and head axes KEPT.

    This is the structure a VLA attention visualizer actually needs (cf. the
    pi0.5 attention visualizer + villekuosmanen/physical-AI-interpretability):

      • ``per_camera[cam]`` has shape ``(n_steps, n_heads, side, side)`` — the
        action queries' cross-attention onto that camera's image patches.
      • ``n_steps`` is the flow-matching / diffusion denoise steps. Attention
        **sharpens** from t=0 (pure-noise action, diffuse) to the last step
        (clean action, locked onto task-relevant objects) — so the default
        view is the LAST step, and the whole t=0..last progression is worth
        scrubbing. We never average across steps (that blurs the signal).
      • ``n_heads`` is the attention heads, which **specialize** — so we expose
        per-head maps and a head-mean, never a forced global average.

    Raw attention, reshaped row-major to the grid (no transpose/flip). All
    display normalization happens in the renderer, not here.
    """

    per_camera: dict[str, np.ndarray]          # cam -> (n_steps, n_heads, side, side)
    grid_sides: dict[str, int]                 # cam -> side (tokens = side*side)
    n_steps: int
    n_heads: int
    source: str                                # e.g. "pi0 action-expert cross-attention"
    query_desc: str                            # e.g. "action chunk tokens (mean)"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for cam, a in self.per_camera.items():
            if a.ndim != 4:
                raise ValueError(
                    f"AttentionTrace[{cam}]: expected (n_steps,n_heads,side,side), got {a.shape}"
                )
            s = self.grid_sides[cam]
            if a.shape[2] != s or a.shape[3] != s:
                raise ValueError(
                    f"AttentionTrace[{cam}]: grid {a.shape[2:]} != side {s}"
                )

    @property
    def cameras(self) -> list[str]:
        return sorted(self.per_camera)

    def step_head(self, camera: str, step: int, head: int) -> np.ndarray:
        """Single (side, side) map for one denoise step + one head."""
        return self.per_camera[camera][step, head]

    def step_mean(self, camera: str, step: int) -> np.ndarray:
        """Mean over heads at one denoise step → (side, side)."""
        return self.per_camera[camera][step].mean(axis=0)

    def final_mean(self, camera: str) -> np.ndarray:
        """The default map: last denoise step, mean over heads → (side, side)."""
        return self.per_camera[camera][-1].mean(axis=0)


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
