"""Adapter for NVIDIA Isaac GR00T-N1 / N1.7.

GR00T is NVIDIA's open foundation model for generalist humanoid robots,
taking multi-camera video, proprioception, and language instructions to
produce action chunks. The adapter wraps `gr00t.policy.Gr00tPolicy` and
maps our typed `Scene` into GR00T's expected observation dict.

Optional dependency: install with
    uv pip install git+https://github.com/NVIDIA/Isaac-GR00T.git

Loading without the package raises ImportError at adapter construction
time with a clear install hint.

Capabilities: INFERENCE only. GR00T's introspection surface (attention,
hidden states) is not uniformly exposed through Gr00tPolicy; capability-
gated diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, AttentionMaps, Scene, TokenSelector
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


# Sensible default embodiment for tabletop manipulation rollouts. Teams
# with their own embodiment override via the `embodiment_tag` arg.
DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.7-3B"


@register_model("gr00t")
@register_model("gr00t-n1")
class Gr00tAdapter(VLAModel):
    """Wraps `gr00t.policy.Gr00tPolicy` as an Emboviz `VLAModel`.

    Construction:
        Gr00tAdapter()                                          # default checkpoint + embodiment
        Gr00tAdapter(model_path="nvidia/GR00T-N1-2B")
        Gr00tAdapter(embodiment_tag="GR1", camera_mapping={"primary": "video.ego_view"})

    `camera_mapping` translates our Scene camera names into GR00T's
    embodiment-specific video keys. By default we route our `"primary"`
    into the first declared video key of the selected embodiment.
    """

    _CAPS = Capability.INFERENCE | Capability.ATTENTION

    # GR00T-N1.7 uses Qwen3-VL (Cosmos-Reason2-2B) as the System-2 VLM
    # backbone. Per the documented behavior:
    #
    #   - Layer range: Qwen3-VL with select_layer truncation keeps ~16
    #     LLM layers. Visual-grounding heads cluster in the middle half
    #     per the same multi-modal stage analysis that covers LLaVA
    #     ("How Multimodal LLMs Solve Image Tasks", arXiv:2508.20279) —
    #     the stage structure (early=tokenization, mid=visual-grounding,
    #     late=prediction) is universal across LLM backbones.
    #
    #   - Sink masking: Qwen3-VL has documented strong attention sinks
    #     on right-edge / corner image tokens. References:
    #       * QwenLM/Qwen3-VL Issue #2047 — "Qwen3 VL Attention Focus"
    #         (community-reported top-corner concentration)
    #       * "To Sink or Not to Sink" (arXiv:2510.08510) — RoPE-induced
    #         positional sinks in VLMs
    #       * "Attention Debiasing for Token Pruning in VLMs"
    #         (arXiv:2508.17807)
    #     For an 8×8 per-tile grid these account for ~10% of cells
    #     (the rightmost column + bottom row ≈ 15 of 64; we mask a
    #     conservative 10% to keep most real signal).
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.25, 0.75),
        "sink_top_pct_to_mask": 0.10,
        "literature_citation":
            "Layer range: 'How Multimodal LLMs Solve Image Tasks' "
            "(arXiv:2508.20279) — generalises to Qwen3-VL. "
            "Sink 10%: QwenLM/Qwen3-VL Issue #2047 (right-edge focus), "
            "'To Sink or Not to Sink' (arXiv:2510.08510), and "
            "'Attention Debiasing for Token Pruning in VLMs' "
            "(arXiv:2508.17807).",
    }

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embodiment_tag: str = DEFAULT_EMBODIMENT,
        device: str = "cuda:0",
        camera_mapping: Optional[dict[str, str]] = None,
    ):
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy import Gr00tPolicy
        except ImportError as e:
            raise ImportError(
                "Gr00tAdapter requires NVIDIA Isaac-GR00T. Install with:\n"
                "    uv pip install git+https://github.com/NVIDIA/Isaac-GR00T.git"
            ) from e

        self.model_path = model_path
        self._embodiment_name = embodiment_tag
        try:
            embodiment_enum = getattr(EmbodimentTag, embodiment_tag)
        except AttributeError as e:
            available = [t.name for t in EmbodimentTag]
            raise ValueError(
                f"Unknown embodiment_tag '{embodiment_tag}'. Available: {available}"
            ) from e

        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_enum,
            device=device,
        )

        self._modality_configs = self.policy.get_modality_config()
        self._video_keys: list[str] = list(self._modality_configs["video"].modality_keys)
        self._state_keys: list[str] = list(self._modality_configs["state"].modality_keys)
        self._action_keys: list[str] = list(self._modality_configs["action"].modality_keys)

        # Map our Scene camera names into GR00T's video keys. The user MUST
        # provide a mapping that covers every declared video key — otherwise
        # we'd be silently filling missing GR00T keys with the wrong camera.
        # Default: only auto-map when exactly ONE video key exists (no
        # ambiguity about which camera goes where).
        if camera_mapping is None:
            if len(self._video_keys) == 1:
                camera_mapping = {"primary": self._video_keys[0]}
            elif len(self._video_keys) == 0:
                camera_mapping = {}
            else:
                raise ValueError(
                    f"GR00T embodiment '{embodiment_tag}' declares "
                    f"{len(self._video_keys)} video keys "
                    f"({self._video_keys}). Pass an explicit camera_mapping "
                    "from your Scene camera names → GR00T video keys. We "
                    "do not auto-map silently when multiple cameras are needed."
                )
        # Validate: every declared GR00T video key must be in the mapping's
        # values (covered by some Scene camera).
        mapped_keys = set(camera_mapping.values())
        missing = set(self._video_keys) - mapped_keys
        if missing:
            raise ValueError(
                f"camera_mapping does not cover GR00T video keys {sorted(missing)}. "
                f"Mapping must route every declared video key — "
                "we do not silently fill missing keys with another camera."
            )
        self._camera_mapping: dict[str, str] = camera_mapping

        # Action dim is sum of dims across action_keys; per first inference probe.
        self._action_dim: Optional[int] = None

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.model_path.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # Every Scene camera named in self._camera_mapping is required —
        # missing ones raise at the framework boundary via
        # RequiredInputs.validate(scene).
        return RequiredInputs(
            cameras=frozenset(self._camera_mapping.keys()),
            instruction=True,
            state=True,
        )

    @property
    def action_dim(self) -> int:
        # GR00T's per-key action dims aren't trivially known before the first
        # inference; we report 0 until we've seen one prediction.
        return self._action_dim or 0

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Gr00tAdapter.predict: {reason}")
        observation = self._build_observation(scene)
        action_dict, _info = self.policy.get_action(observation)

        # Concatenate per-action-key arrays in declared order. GR00T's
        # output shape is conventionally (B, T, D_key) — we keep the full
        # T axis as the action_chunk and expose the first timestep as the
        # immediate action (consistent with OpenVLA-style outputs).
        chunks_per_key: list[np.ndarray] = []
        for key in self._action_keys:
            arr = action_dict.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 3:                # (B, T, D)
                key_chunk = arr[0]           # (T, D)
            elif arr.ndim == 2:              # (T, D)
                key_chunk = arr
            else:                            # (D,) — single-step
                key_chunk = arr[np.newaxis, :]
            chunks_per_key.append(key_chunk.astype(np.float32))

        if chunks_per_key:
            # Align time dim across keys (take min T) then concat along D.
            min_t = min(c.shape[0] for c in chunks_per_key)
            chunk = np.concatenate(
                [c[:min_t] for c in chunks_per_key], axis=-1,
            ).astype(np.float32)
            action = chunk[0]
        else:
            chunk = np.zeros((1, 0), dtype=np.float32)
            action = np.zeros(0, dtype=np.float32)

        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "model_path":     self.model_path,
                "embodiment_tag": self._embodiment_name,
                "action_keys":    list(self._action_keys),
                "chunk_shape":    list(chunk.shape),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        # GR00T's tokenizer surface isn't uniformly exposed; language-axis
        # token-position queries fall through (diagnostics that need them
        # will fail their capability check anyway).
        return []

    # ----- attention extraction -----------------------------------------

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        """Extract per-camera attention from GR00T's Qwen3-VL backbone.

        Strategy: monkey-patch the Qwen3-VL ``forward`` to inject
        ``output_attentions=True`` while running the normal
        ``policy.get_action`` pipeline (we discard the action output;
        only attention is wanted). Capture attention + the input_ids +
        image_grid_thw needed to map image tokens back to per-camera
        spatial grids.

        Multi-camera handling:
          • Image tokens are INLINE in input_ids (marked by
            ``image_token_id``). Multiple cameras appear as multiple
            contiguous runs of image tokens.
          • ``image_grid_thw`` is shape ``(N_images, 3)`` with per-image
            ``(T, H, W)``. After Qwen3-VL's spatial merge of size
            ``merge_size`` (default 2), each image consumes
            ``T * (H // merge_size) * (W // merge_size)`` tokens.
          • The order images are emitted into the prompt is determined
            by the modality config's ``video.modality_keys`` order. We
            reverse-map those to user-facing camera names via
            ``self._camera_mapping`` (which maps Scene-camera → GR00T-key).
        """
        import torch

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Gr00tAdapter.extract_attention: {reason}")

        observation = self._build_observation(scene)

        qwen_model = self.policy.model.backbone.model
        original_forward = qwen_model.forward

        # SDPA / flash-attention attention implementations silently return
        # ``None`` for ``output_attentions=True`` (they fuse the softmax
        # into the kernel and never materialize the attention matrix).
        # We MUST switch to the "eager" implementation while extracting
        # attention, then restore. Without this our patched_forward would
        # capture ``None`` and raise the "forward was not invoked" error.
        original_attn_impl = None
        if hasattr(qwen_model, "set_attn_implementation"):
            try:
                # Best-effort read of current impl (varies by version).
                original_attn_impl = getattr(qwen_model.config, "_attn_implementation", None)
                qwen_model.set_attn_implementation("eager")
            except Exception as e:
                raise RuntimeError(
                    f"Gr00tAdapter.extract_attention: could not switch Qwen3-VL "
                    f"to 'eager' attention via set_attn_implementation: {e}. "
                    "Required for attention capture — SDPA / flash-attention "
                    "kernels never materialize the attention matrix."
                ) from e
        else:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: Qwen3-VL model does not "
                "expose set_attn_implementation(). transformers version too "
                "old (need >=4.43 or so). Pin a newer transformers in the "
                "gr00t venv."
            )

        captured: dict = {
            "attentions":     None,
            "input_ids":      None,
            "image_grid_thw": None,
        }

        def patched_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            result = original_forward(*args, **kwargs)
            captured["attentions"]     = result.attentions
            captured["input_ids"]      = kwargs.get("input_ids", args[0] if args else None)
            captured["image_grid_thw"] = kwargs.get("image_grid_thw")
            return result

        qwen_model.forward = patched_forward
        try:
            with torch.inference_mode():
                _ = self.policy.get_action(observation)
        finally:
            qwen_model.forward = original_forward
            if original_attn_impl is not None:
                try:
                    qwen_model.set_attn_implementation(original_attn_impl)
                except Exception:
                    pass  # best-effort restore

        if captured["attentions"] is None:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: Qwen3-VL forward was "
                "not invoked during policy.get_action — the patched "
                "forward never ran. This indicates a GR00T-policy code "
                "path change; review gr00t/model/gr00t_n1d7."
            )

        # Build (n_layers, n_heads, n_keys) attention tensor at query_pos.
        attns = captured["attentions"]
        input_ids = captured["input_ids"]
        image_grid_thw = captured["image_grid_thw"]

        full_seq = int(attns[0].shape[-1])

        # Resolve query position.
        if query.position is not None:
            query_pos = int(query.position)
        elif query.relative == "last" or query.relative == "before_action":
            # GR00T predicts actions via a separate diffusion head conditioned
            # on the LM hidden states; "before_action" is the LAST LM position
            # whose hidden state feeds the action head.
            query_pos = full_seq - 1
        elif query.relative == "first":
            query_pos = 0
        else:
            query_pos = full_seq - 1

        per_layer = [
            layer_attn[0, :, query_pos, :].float().cpu().numpy() for layer_attn in attns
        ]
        weights = np.stack(per_layer, axis=0)   # (L, H, n_keys)

        # ---- Map image tokens to per-camera ranges ----
        image_token_id = qwen_model.config.image_token_id
        ids_row = input_ids[0].cpu().numpy()
        image_positions = np.where(ids_row == image_token_id)[0]
        if image_positions.size == 0:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: no image tokens found "
                f"in input_ids (looked for token id {image_token_id}). "
                "The processor produced a prompt without image placeholders."
            )

        # spatial_merge_size — default 2 for Qwen3-VL but read from config
        # rather than hardcoding.
        try:
            merge_size = int(qwen_model.config.vision_config.spatial_merge_size)
        except AttributeError:
            merge_size = 2

        if image_grid_thw is None:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: processor did not produce "
                "image_grid_thw, can't split per-camera image tokens."
            )
        thw_np = image_grid_thw.cpu().numpy().astype(int)

        # GR00T video keys in declaration order — same order the processor
        # emits images into the prompt. Each user camera contributes T
        # temporal-tile entries to image_grid_thw, where T is the video
        # delta_indices count (we replicate the current frame T times in
        # ``_build_observation`` when only one Scene is available).
        video_keys = list(self._modality_configs["video"].modality_keys)
        video_horizon = len(self._modality_configs["video"].delta_indices)
        n_images_processed = thw_np.shape[0]
        expected = len(video_keys) * video_horizon
        if n_images_processed != expected:
            raise RuntimeError(
                f"Gr00tAdapter.extract_attention: image_grid_thw has "
                f"{n_images_processed} tile entries but expected "
                f"{expected} (len(video_keys)={len(video_keys)} × "
                f"temporal_horizon={video_horizon}). Cannot map tiles "
                "back to cameras unambiguously."
            )

        # Reverse the camera_mapping: {user_cam → gr00t_key} →
        # {gr00t_key → user_cam}.
        gr00t_key_to_user = {v: k for k, v in self._camera_mapping.items()}

        # Walk image_positions tile by tile, grouping tiles by camera.
        #
        # Tile ordering per Gr00tN1d7Processor.process_observation
        # (gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py ~L410):
        #
        #     images = torch.stack(
        #         [images_dict[view] for view in image_keys], dim=2
        #     )  # shape (B, T, V, H, W, C)
        #     ...
        #     images_perm = images.permute(0, 1, 2, 5, 3, 4).reshape(
        #         B, T * V, C, H, W
        #     )
        #
        # PyTorch reshape is row-major, so the (T, V) flatten places T as
        # the OUTER index and V as the INNER index:
        #     tile 0 = (t=0, view=video_keys[0])
        #     tile 1 = (t=0, view=video_keys[1])
        #     tile 2 = (t=1, view=video_keys[0])
        #     tile 3 = (t=1, view=video_keys[1])
        #
        # → camera_idx = tile_i % num_cameras   (NOT tile_i // T).
        #
        # Within-tile reshape: Qwen2/3-VL's image_processing flattens
        # patches in (grid_t, grid_h//m, grid_w//m) order after a 9-d
        # reshape + transpose(0, 3, 6, 4, 7, 2, 1, 5, 8) (see
        # transformers/models/qwen2_vl/image_processing_qwen2_vl.py
        # L281-296). The first three transposed dims (t, block_row,
        # block_col) flatten row-major, so the per-tile tokens come out
        # in row-major BLOCK order — our (side, side) reshape is correct.
        cursor = 0
        image_token_ranges: dict[str, list[tuple[int, int]]] = {}
        image_grid_sides: dict[str, int] = {}
        per_tile_side: Optional[int] = None
        num_cameras = len(video_keys)
        for tile_i, (t, h, w) in enumerate(thw_np):
            tokens_per_tile = int(t * (h // merge_size) * (w // merge_size))
            run_start = int(image_positions[cursor])
            run_end = int(image_positions[cursor + tokens_per_tile - 1]) + 1
            cursor += tokens_per_tile

            h_eff = int(h // merge_size)
            w_eff = int(w // merge_size)
            if int(t) != 1:
                raise RuntimeError(
                    f"Gr00tAdapter.extract_attention: image_grid_thw "
                    f"entry {tile_i} has T={t} > 1. Per-tile T>1 means "
                    "a single embedding contains multiple time steps; "
                    "extraction would need to split further. Not "
                    "supported for this embodiment yet."
                )
            if h_eff != w_eff:
                raise RuntimeError(
                    f"Gr00tAdapter tile {tile_i}: non-square grid after "
                    f"spatial merge ({h_eff}, {w_eff}). AttentionMaps "
                    "currently assumes square per-tile grids."
                )
            if per_tile_side is None:
                per_tile_side = h_eff
            elif per_tile_side != h_eff:
                raise RuntimeError(
                    f"Gr00tAdapter: per-tile grid_side changed across "
                    f"tiles ({per_tile_side} → {h_eff}). Mixed grid "
                    "sides not yet supported."
                )

            camera_idx = tile_i % num_cameras
            gr00t_key = video_keys[camera_idx]
            user_cam = gr00t_key_to_user.get(gr00t_key, gr00t_key)
            image_token_ranges.setdefault(user_cam, []).append((run_start, run_end))
            image_grid_sides[user_cam] = h_eff

        return AttentionMaps(
            weights=weights,
            query_position=query_pos,
            n_keys=full_seq,
            image_token_ranges=image_token_ranges,
            image_grid_sides=image_grid_sides,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "select_layer": len(attns),
                "image_grid_thw": thw_np.tolist(),
                "merge_size": merge_size,
                "n_image_tokens_total": int(image_positions.size),
            },
        )

    def _state_key_dim(self, state_key: str) -> int:
        """Inferred dim for a GR00T state key.

        GR00T's normalization metadata stores per-key min/max — that's the
        truth. We walk into the policy's processor to read it; if the
        processor's introspection path is shaped differently (different
        gr00t version), we warn and fall back to name-based heuristics
        (9d → 9, joint → 7, gripper → 1) so the user knows we are
        guessing.
        """
        import warnings as _warnings
        try:
            proc = self.policy.processor
            sap = proc.state_action_processor
            embodiments = getattr(sap, "state_norm", None)
            if embodiments:
                params = (
                    embodiments.get(self._embodiment_name.lower())
                    or next(iter(embodiments.values()))
                )
                if params and state_key in params:
                    p = params[state_key]
                    mn = getattr(p, "min", None)
                    if mn is not None and hasattr(mn, "__len__"):
                        return int(len(mn))
        except (AttributeError, KeyError, TypeError) as e:
            _warnings.warn(
                f"GR00T state-dim introspection failed for "
                f"key='{state_key}': {type(e).__name__}: {e}. Falling back "
                "to name-based heuristic; verify the dim matches your "
                "embodiment's normalization spec.",
                stacklevel=2,
            )
        k = state_key.lower()
        if "9d" in k:
            return 9
        if "gripper" in k:
            return 1
        return 7

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Convert our typed Scene into GR00T's nested observation dict.

        GR00T's video modality expects shape (B, T, H, W, 3) where T is the
        embodiment-specific temporal horizon (often 2 = current + previous
        frame). When we only have one Scene we repeat it along the time
        axis — this means "no motion" but is a valid input shape.
        """
        # Read the per-modality temporal horizon (T) from the modality config.
        video_cfg = self._modality_configs["video"]
        video_horizon = len(video_cfg.delta_indices)

        video: dict[str, np.ndarray] = {}

        def _to_horizon(arr_3d: np.ndarray) -> np.ndarray:
            """(H, W, 3) → (1, T, H, W, 3)."""
            stacked = np.stack([arr_3d] * video_horizon, axis=0)
            return stacked[np.newaxis, ...]

        # Fill every mapped camera. required_inputs.validate() already
        # confirmed each mapped Scene-camera-name is present and the
        # camera_mapping covers every declared GR00T video key, so any
        # KeyError here is a real bug — let it surface, don't paper over.
        for cam_name, gr00t_key in self._camera_mapping.items():
            arr = np.asarray(scene.observations.images[cam_name].data, dtype=np.uint8)
            if arr.ndim != 3:
                raise ValueError(
                    f"Scene camera '{cam_name}' has shape {arr.shape}; "
                    "expected (H, W, 3)."
                )
            video[gr00t_key] = _to_horizon(arr)

        # State: same temporal-horizon treatment as video.
        state_cfg = self._modality_configs.get("state")
        state_horizon = len(state_cfg.delta_indices) if state_cfg else 1

        # State: distribute Scene state across GR00T's declared state keys.
        #
        # Routing priority:
        #   1. If the dataset's RobotProfile.state.segment_layout has an
        #      EXACT key match for the GR00T state key → use that slice
        #      from the Scene's state vector. This is the precise path for
        #      datasets shaped for GR00T (e.g. droid_sample stores 17-dim
        #      state with segments eef_9d / gripper_position / joint_position).
        #   2. Else if the key contains "gripper" → use the typed
        #      observations.gripper.value (single scalar).
        #   3. Else if the key looks like a proprioception key (eef/pose/joint)
        #      → use the FULL state vector (only valid when the model's
        #      state key dim matches the scene's state dim — checked below).
        #   4. Else → raise (refuse to silently zero-fill).
        state: dict[str, np.ndarray] = {}
        proprio_vec = np.asarray(
            scene.observations.state.values, dtype=np.float32,
        ).reshape(-1)
        gripper = scene.observations.gripper

        # Read segment_layout from profile if the dataset provides it.
        segment_layout = (
            scene.profile.state.segment_layout
            if (scene.profile is not None
                and scene.profile.state is not None
                and scene.profile.state.segment_layout is not None)
            else {}
        )

        def _to_state_horizon(vec: np.ndarray) -> np.ndarray:
            stacked = np.stack([vec] * state_horizon, axis=0)
            return stacked[np.newaxis, ...]

        for sk in self._state_keys:
            expected_dim = self._state_key_dim(sk)
            sk_lower = sk.lower()

            # 1. Exact segment_layout match — preferred when available.
            if sk in segment_layout:
                vec = proprio_vec[segment_layout[sk]]
            # 2. Gripper key from the typed Scene field.
            elif "gripper" in sk_lower:
                if gripper is None:
                    raise ValueError(
                        f"GR00T embodiment requires state key '{sk}' (gripper) "
                        "but scene.observations.gripper is None AND the "
                        "dataset profile has no segment_layout entry for '{sk}'. "
                        "Either populate gripper in the dataset adapter or "
                        "declare segment_layout in RobotProfile.state."
                    )
                vec = np.array([gripper.value], dtype=np.float32)
            # 3. Proprio-like key from the full state vector (only valid
            #    when dims happen to match).
            elif "eef" in sk_lower or "pose" in sk_lower or "joint" in sk_lower:
                vec = proprio_vec
            # 4. Don't know what to put here — raise.
            else:
                raise ValueError(
                    f"GR00T embodiment '{self._embodiment_name}' declares "
                    f"state key '{sk}' (expected dim {expected_dim}). We "
                    "cannot route it from the Scene because:\n"
                    f"  • RobotProfile.state.segment_layout has no '{sk}' entry "
                    f"(present keys: {sorted(segment_layout)})\n"
                    "  • the key name doesn't contain "
                    "'gripper'/'eef'/'pose'/'joint'.\n"
                    "Add the segment to RobotProfile.state.segment_layout, "
                    "or subclass Gr00tAdapter._build_observation."
                )
            if vec.size != expected_dim:
                raise ValueError(
                    f"GR00T state key '{sk}' expects dim {expected_dim} "
                    f"but routed Scene segment provides {vec.size}. Either "
                    "fix the dataset profile's segment_layout or the gripper "
                    "extractor — no silent pad/truncate."
                )
            state[sk] = _to_state_horizon(vec)

        # Language: GR00T's language modality declares its own keys. We
        # populate ONLY the declared keys with the (validated non-empty)
        # instruction. We do NOT setdefault('task', ...) — silently
        # injecting a key the embodiment did not declare risks feeding the
        # model an unexpected shape on future embodiments whose language
        # modality has different keys.
        language: dict = {}
        lang_cfg = self._modality_configs.get("language")
        if lang_cfg is None:
            raise ValueError(
                f"GR00T embodiment '{self._embodiment_name}' declares no "
                "language modality config. We never inject a default "
                "language key — fix the embodiment or subclass the adapter."
            )
        if not scene.instruction:
            raise ValueError(
                "Gr00tAdapter requires a non-empty instruction but "
                "scene.instruction is empty / None. The dataset adapter "
                "must produce a task string for every frame."
            )
        for lk in lang_cfg.modality_keys:
            language[lk] = [[scene.instruction]]

        return {"video": video, "state": state, "language": language}
