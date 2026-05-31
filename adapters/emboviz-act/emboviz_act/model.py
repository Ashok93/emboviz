"""ACTPolicy wrapped as a VLAModel.

ACT consumes per-camera images and a proprioceptive-state token and emits
an action chunk; it has no language input. Inference runs through
lerobot's own pre/post-processor pipeline (loaded from the checkpoint), so
normalization stats are the model's own and are never reconstructed here.

Attention is the decoder cross-attention: each learned action query
attends to the transformer-encoder memory, whose image tokens are the
flattened per-camera ResNet feature maps. The feature grid is
``H/stride x W/stride`` and is generally NOT square, so the map is
reported with an explicit ``(h, w)`` grid shape.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emboviz_wire import (
    ActionResult,
    AttentionMaps,
    Capability,
    RequiredInputs,
    Scene,
    TokenSelector,
    VLAModel,
)


def _read_rename_map(preprocessor) -> dict[str, str]:
    """Observation rename map (dataset key -> model feature key) from the
    checkpoint's preprocessor pipeline. Empty when the pipeline has no rename
    step (the model consumes the dataset keys directly)."""
    for step in getattr(preprocessor, "steps", []):
        if type(step).__name__ == "RenameObservationsProcessorStep":
            return dict(getattr(step, "rename_map", {}) or {})
    return {}


class ACTAdapter(VLAModel):
    """lerobot ACTPolicy behind the emboviz VLAModel interface.

    Args:
        checkpoint: HF repo id or local directory of a trained ACT
            checkpoint (loaded with ``ACTPolicy.from_pretrained``).
        camera_mapping: maps each emboviz logical camera role to the
            dataset key the checkpoint's preprocessor expects as input (the
            preprocessor renames + normalizes it), e.g.
            ``{"primary": "observation.images.top"}``. Must cover every
            image feature ACT consumes.
        device: ``"auto"`` (cuda if available, else cpu), or an explicit
            torch device string.
    """

    # DETR-style decoder cross-attention. ACT's decoder is shallow (the
    # lerobot default is a single layer), so the layer band spans all
    # layers and image_weights_clean picks within it by interior
    # concentration. Carion et al. 2020 (DETR) visualize exactly these
    # decoder-query cross-attention maps.
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.0, 1.0),
        # ACT has a single decoder layer with 8 heads that SPECIALISE (DETR,
        # Carion et al. 2020): some heads ground on the end-effectors/contact
        # while others are spatial sinks on the frame border. Averaging heads
        # blends the sink in, so we select the single most interior-concentrated
        # head rather than the head-mean (verified per-head on the reference
        # checkpoint: grounding heads sit at interior-fraction ~0.85-1.0, sink
        # heads at ~0.42).
        "head_reduction": "select_interior",
        "literature_citation": (
            "DETR decoder cross-attention visualization (Carion et al. "
            "2020, arXiv:2005.12872) on ACT (Zhao et al. 2023, "
            "arXiv:2304.13705): query = first action token; pick the (layer, "
            "head) most concentrated on the image interior — heads specialise "
            "in ACT's single shallow decoder layer, so the grounding head is "
            "selected instead of averaging the spatial-sink heads in. "
            "Action-pathway attention, not a language-anchored map."
        ),
    }

    def __init__(
        self,
        checkpoint: str,
        camera_mapping: dict[str, str],
        device: str = "auto",
        **kwargs: Any,
    ):
        if not checkpoint or not isinstance(checkpoint, str):
            raise ValueError(
                "ACTAdapter requires ``checkpoint`` (HF repo id or local "
                "directory of a trained ACT policy)."
            )
        if not camera_mapping or not isinstance(camera_mapping, dict):
            raise ValueError(
                "ACTAdapter requires ``camera_mapping`` mapping each emboviz "
                "logical camera role to the checkpoint's image-feature key, "
                "e.g. {\"primary\": \"observation.images.top\"}."
            )
        self.checkpoint = checkpoint
        self.camera_mapping = {str(k): str(v) for k, v in camera_mapping.items()}
        self._device_pref = device
        self._policy = None
        self._pre = None
        self._post = None
        self._device: Optional[str] = None
        self._load()

    # ----- lifecycle ------------------------------------------------------

    def _resolve_device(self) -> str:
        import torch
        if self._device_pref in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return str(self._device_pref)

    def _load(self) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from emboviz_act._checkpoint import load_processors

        device = self._resolve_device()
        policy = ACTPolicy.from_pretrained(self.checkpoint)
        policy.config.device = device
        policy.to(device).eval()

        cfg = policy.config
        if cfg.env_state_feature is not None:
            raise ValueError(
                "ACTAdapter: this checkpoint consumes an environment-state "
                "feature (observation.environment_state), which emboviz "
                "scenes do not provide. Use an image+state ACT checkpoint."
            )
        if not cfg.image_features:
            raise ValueError(
                "ACTAdapter: checkpoint declares no image features; emboviz "
                "expects a vision-based ACT policy."
            )

        self._policy = policy
        self._device = device
        self._pre, self._post = load_processors(
            cfg, self.checkpoint, make_pre_post_processors,
        )

        # camera_mapping maps each emboviz logical role to the dataset key the
        # checkpoint's preprocessor expects; the preprocessor renames +
        # normalizes. ACT's predict_action_chunk indexes EVERY image feature,
        # so the mapping must cover all of them (after the rename).
        rename = _read_rename_map(self._pre)        # input key -> model camera
        modelcam_to_role: dict[str, str] = {}
        for role, input_key in self.camera_mapping.items():
            model_cam = rename.get(input_key, input_key)
            if model_cam not in cfg.image_features:
                raise ValueError(
                    f"ACTAdapter: camera_mapping role '{role}' -> input key "
                    f"'{input_key}' resolves (after the checkpoint's rename) to "
                    f"'{model_cam}', which is not one of the model's image "
                    f"features {list(cfg.image_features)}."
                )
            if model_cam in modelcam_to_role:
                raise ValueError(
                    f"ACTAdapter: two camera_mapping entries resolve to the "
                    f"same model camera '{model_cam}'."
                )
            modelcam_to_role[model_cam] = role
        if set(modelcam_to_role) != set(cfg.image_features):
            missing = set(cfg.image_features) - set(modelcam_to_role)
            raise ValueError(
                "ACTAdapter: ACT consumes every image feature; camera_mapping "
                f"is missing {sorted(missing)} (after the checkpoint's rename)."
            )

        # Encoder token layout: [latent, (robot_state), image tokens...].
        # ACT processes all image features, in config order.
        self._present_model_cameras = [
            c for c in cfg.image_features if c in modelcam_to_role
        ]
        self._modelcam_to_role = modelcam_to_role
        self._uses_state = cfg.robot_state_feature is not None
        self._n_non_image_tokens = 1 + (1 if self._uses_state else 0)
        self._n_decoder_layers = int(cfg.n_decoder_layers)
        self._n_heads = int(cfg.n_heads)
        self._dim_model = int(cfg.dim_model)
        self._action_dim = int(cfg.action_feature.shape[-1])

    def close(self) -> None:
        self._policy = None
        self._pre = None
        self._post = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----- identification -------------------------------------------------

    @property
    def model_id(self) -> str:
        return "act"

    @property
    def capabilities(self) -> Capability:
        return Capability.INFERENCE | Capability.ATTENTION

    @property
    def required_inputs(self) -> RequiredInputs:
        return RequiredInputs(
            cameras=frozenset(self.camera_mapping),
            instruction=False,
            state=self._uses_state,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_layers(self) -> Optional[int]:
        return self._n_decoder_layers

    @property
    def num_heads(self) -> Optional[int]:
        return self._n_heads

    @property
    def hidden_dim(self) -> Optional[int]:
        return self._dim_model

    # ----- inference ------------------------------------------------------

    def _build_frame(self, scene: Scene) -> dict:
        """Build the un-batched lerobot frame: image tensors (C,H,W float
        in [0,1]) keyed by the policy's image-feature names, plus the
        state vector. The pre-processor adds the batch dim, moves to
        device, and normalizes."""
        import torch
        from lerobot.utils.constants import OBS_STATE

        frame: dict[str, Any] = {}
        for role, input_key in self.camera_mapping.items():
            img = scene.observations.images[role].data
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"ACTAdapter: camera '{role}' image must be HxWx3 RGB; "
                    f"got shape {arr.shape}."
                )
            t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
            frame[input_key] = t.permute(2, 0, 1).contiguous()

        if self._uses_state:
            state = scene.observations.state
            if state is None:
                raise ValueError(
                    "ACTAdapter: checkpoint requires proprioceptive state "
                    "but scene.observations.state is None."
                )
            vec = np.asarray(state.values, dtype=np.float32).reshape(-1)
            if vec.size != self._policy.config.robot_state_feature.shape[-1]:
                raise ValueError(
                    "ACTAdapter: state dim "
                    f"{vec.size} != checkpoint's "
                    f"{self._policy.config.robot_state_feature.shape[-1]}."
                )
            frame[OBS_STATE] = torch.from_numpy(vec)
        return frame

    def _action_chunk(self, scene: Scene) -> np.ndarray:
        import torch
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"ACTAdapter.predict: {reason}")
        frame = self._build_frame(scene)
        batch = self._pre(frame)
        with torch.no_grad():
            chunk = self._policy.predict_action_chunk(batch)
        chunk = self._post(chunk)
        arr = np.asarray(chunk.detach().to("cpu", torch.float32).numpy())
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise RuntimeError(
                f"ACTAdapter: expected a (1, chunk, action_dim) chunk; got "
                f"shape {arr.shape}."
            )
        return arr[0]   # (chunk_size, action_dim)

    def predict(self, scene: Scene) -> ActionResult:
        chunk = self._action_chunk(scene)
        action = chunk[0]
        return ActionResult(
            action=np.asarray(action, dtype=np.float32),
            action_dim=int(action.size),
            action_chunk=np.asarray(chunk, dtype=np.float32),
            metadata={"model": "act", "chunk_shape": list(chunk.shape)},
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        # ACT has no language tokenizer.
        return []

    # ----- attention ------------------------------------------------------

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        """Decoder cross-attention from the first action query to the
        encoder image tokens, per decoder layer and head.

        ``query`` is ignored: ACT has no token sequence to select from; the
        map is always the first action query (the immediate action).
        """
        import torch

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"ACTAdapter.extract_attention: {reason}")

        model = self._policy.model
        frame = self._build_frame(scene)
        batch = self._pre(frame)

        feat_hw: dict[str, tuple[int, int]] = {}
        captured: list = []

        def backbone_hook(_module, _inp, out):
            fm = out["feature_map"] if isinstance(out, dict) else out
            feat_hw["hw"] = (int(fm.shape[-2]), int(fm.shape[-1]))

        bb_handle = model.backbone.register_forward_hook(backbone_hook)
        patched: list = []
        for layer in model.decoder.layers:
            mha = layer.multihead_attn
            original = mha.forward

            def make_patched(orig):
                # ACT calls multihead_attn with keyword query=/key=/value=,
                # so forward *args/**kw verbatim and only override the
                # weight-averaging flags (default discards per-head weights).
                def patched_forward(*args, **kw):
                    kw.pop("need_weights", None)
                    kw.pop("average_attn_weights", None)
                    out, weights = orig(
                        *args, need_weights=True, average_attn_weights=False, **kw,
                    )
                    captured.append(weights.detach())
                    return out, weights
                return patched_forward

            mha.forward = make_patched(original)
            patched.append((mha, original))

        try:
            with torch.no_grad():
                self._policy.predict_action_chunk(batch)
        finally:
            bb_handle.remove()
            for mha, original in patched:
                mha.forward = original

        if "hw" not in feat_hw:
            raise RuntimeError(
                "ACTAdapter.extract_attention: the ResNet backbone hook did "
                "not fire; lerobot's ACT forward path may have changed."
            )
        if len(captured) != self._n_decoder_layers:
            raise RuntimeError(
                "ACTAdapter.extract_attention: captured "
                f"{len(captured)} cross-attention layers but the config "
                f"declares {self._n_decoder_layers}."
            )

        h, w = feat_hw["hw"]
        tokens_per_cam = h * w
        n_cams = len(self._present_model_cameras)
        encoder_len = int(captured[0].shape[-1])
        expected = self._n_non_image_tokens + n_cams * tokens_per_cam
        if encoder_len != expected:
            raise RuntimeError(
                "ACTAdapter.extract_attention: encoder length "
                f"{encoder_len} != expected {expected} "
                f"({self._n_non_image_tokens} non-image + {n_cams} cameras x "
                f"{tokens_per_cam} tokens)."
            )

        # weights (L, H, encoder_len) from the first action query's row.
        per_layer = []
        for layer_w in captured:
            a = layer_w[0]                          # (H, n_queries, encoder_len)
            per_layer.append(a[:, 0, :].to(torch.float32).cpu().numpy())
        weights = np.stack(per_layer, axis=0)       # (L, H, encoder_len)

        first_row = weights[0, 0, :]
        if np.allclose(first_row, first_row[0], atol=1e-9):
            raise RuntimeError(
                "ACTAdapter.extract_attention: cross-attention is uniform "
                "across all keys — degenerate output, refusing to return it."
            )

        image_token_ranges: dict[str, list[tuple[int, int]]] = {}
        image_grid_shapes: dict[str, tuple[int, int]] = {}
        cursor = self._n_non_image_tokens
        for model_cam in self._present_model_cameras:
            role = self._modelcam_to_role[model_cam]
            image_token_ranges[role] = [(cursor, cursor + tokens_per_cam)]
            image_grid_shapes[role] = (h, w)
            cursor += tokens_per_cam

        return AttentionMaps(
            weights=weights,
            query_position=0,
            n_keys=encoder_len,
            image_token_ranges=image_token_ranges,
            image_grid_shapes=image_grid_shapes,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "n_decoder_layers": self._n_decoder_layers,
                "n_heads": self._n_heads,
                "feature_grid": [h, w],
                "query_token": "first action query (decoder)",
                "attention_source": (
                    "ACT decoder cross-attention: action query -> encoder "
                    "image tokens (DETR-style)"
                ),
            },
        )
