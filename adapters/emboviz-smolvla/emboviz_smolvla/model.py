"""SmolVLAPolicy wrapped as a VLAModel.

SmolVLA consumes per-camera images, a language instruction, and the robot
state, and a flow-matching action expert produces an action chunk.
Inference runs through lerobot's own pre/post-processor pipeline (loaded
from the checkpoint), which tokenizes the instruction and applies the
model's normalization stats; none are reconstructed here. Inference is
stochastic (the action expert samples noise), so per-frame predictions are
averaged over samples by the calibration layer.

Attention is the SmolVLM2 prefix self-attention: the last instruction
token's attention over the image patches, read from the prefix forward
that fills the KV cache before denoising. This is the visual-grounding
signal used for the OpenVLA / pi0 maps; the action expert's
suffix->prefix attention is the action pathway and is not used here.
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


# Pinned RNG seed for prediction. SmolVLA's action expert samples noise
# (torch) each call, so an unseeded ``predict`` varies run-to-run. emboviz's
# comparative diagnostics measure how the action CHANGES under an intervention;
# against a stochastic predict that change is confounded with sampling jitter.
# Pinning the seed makes ``predict`` deterministic in its input, so the baseline
# and the intervention draw IDENTICAL noise and it cancels in their difference
# (common random numbers): one forward pass per arm, no averaging. Analysis-time
# only — the deployed policy is untouched.
_ANALYSIS_SEED = 0


class SmolVLAAdapter(VLAModel):
    """lerobot SmolVLAPolicy behind the emboviz VLAModel interface.

    Args:
        checkpoint: HF repo id or local directory of a SmolVLA checkpoint
            (default: the public ``lerobot/smolvla_base``).
        camera_mapping: maps each emboviz logical camera role to the
            dataset key the checkpoint's preprocessor expects as input (the
            key it renames to the model's internal slot), e.g.
            ``{"primary": "observation.images.image",
               "wrist": "observation.images.image2"}``.
        device: ``"auto"`` (cuda if available, else cpu), or an explicit
            torch device string.
    """

    # SmolVLM2 (SigLIP + SmolLM2) backbone: last-instruction-token
    # attention over the image patches at the mid-stack layer, per the
    # visual-grounding localization-head literature.
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.25, 0.85),
        "literature_citation": (
            "Last-instruction-token localization heads (arXiv:2503.06287) "
            "with layer-adaptive selection (arXiv:2602.04304) on the "
            "SmolVLM2 backbone of SmolVLA (Shukor et al. 2025): query = "
            "last instruction token of the prefix; pick the mid-stack layer "
            "most concentrated on the image interior; mean over heads. Raw "
            "attention, no gradient."
        ),
    }

    def __init__(
        self,
        checkpoint: str = "lerobot/smolvla_base",
        camera_mapping: Optional[dict[str, str]] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        if not checkpoint or not isinstance(checkpoint, str):
            raise ValueError(
                "SmolVLAAdapter requires ``checkpoint`` (HF repo id or local "
                "directory of a SmolVLA policy)."
            )
        if not camera_mapping or not isinstance(camera_mapping, dict):
            raise ValueError(
                "SmolVLAAdapter requires ``camera_mapping`` mapping each "
                "emboviz logical camera role to the checkpoint's image-"
                "feature key, e.g. {\"primary\": \"observation.images.top\"}."
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
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from emboviz_smolvla._checkpoint import load_processors

        device = self._resolve_device()
        policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
        policy.config.device = device
        policy.to(device).eval()

        cfg = policy.config
        if not cfg.image_features:
            raise ValueError(
                "SmolVLAAdapter: checkpoint declares no image features."
            )

        self._policy = policy
        self._device = device
        self._pre, self._post = load_processors(
            cfg, self.checkpoint, make_pre_post_processors,
        )

        # The checkpoint's preprocessor renames the training dataset's keys to
        # the model's internal feature slots (e.g. observation.images.image ->
        # observation.images.camera1) and normalizes with the model's own
        # stats. camera_mapping therefore maps each emboviz logical role to the
        # key the preprocessor EXPECTS as input; the preprocessor handles the
        # rename + normalization.
        rename = _read_rename_map(self._pre)        # input key -> model camera
        modelcam_to_role: dict[str, str] = {}
        for role, input_key in self.camera_mapping.items():
            model_cam = rename.get(input_key, input_key)
            if model_cam not in cfg.image_features:
                raise ValueError(
                    f"SmolVLAAdapter: camera_mapping role '{role}' -> input key "
                    f"'{input_key}' resolves (after the checkpoint's rename) to "
                    f"'{model_cam}', which is not one of the model's image "
                    f"features {list(cfg.image_features)}."
                )
            if model_cam in modelcam_to_role:
                raise ValueError(
                    f"SmolVLAAdapter: two camera_mapping entries resolve to the "
                    f"same model camera '{model_cam}'."
                )
            modelcam_to_role[model_cam] = role

        # Cameras the model actually processes, in config order (SmolVLA uses
        # the present cameras; padded/unused slots are simply not fed).
        self._present_model_cameras = [
            c for c in cfg.image_features if c in modelcam_to_role
        ]
        self._modelcam_to_role = modelcam_to_role
        self._uses_state = cfg.robot_state_feature is not None
        self._action_dim = int(cfg.action_feature.shape[-1])
        mwe = policy.model.vlm_with_expert
        self._num_vlm_layers = int(mwe.num_vlm_layers)
        self._n_heads = int(mwe.num_attention_heads)
        hidden = getattr(mwe.vlm.config.text_config, "hidden_size", None)
        self._dim_model = int(hidden) if hidden is not None else None

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
        return "smolvla"

    @property
    def capabilities(self) -> Capability:
        return Capability.INFERENCE | Capability.ATTENTION

    @property
    def required_inputs(self) -> RequiredInputs:
        return RequiredInputs(
            cameras=frozenset(self.camera_mapping),
            instruction=True,
            state=self._uses_state,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_layers(self) -> Optional[int]:
        return self._num_vlm_layers

    @property
    def num_heads(self) -> Optional[int]:
        return self._n_heads

    @property
    def hidden_dim(self) -> Optional[int]:
        return self._dim_model

    # ----- inference ------------------------------------------------------

    def _build_frame(self, scene: Scene) -> dict:
        """Build the un-batched frame the checkpoint's preprocessor expects:
        image tensors (C,H,W float in [0,1]) keyed by the dataset keys the
        preprocessor renames, the state vector, and the instruction string.
        The preprocessor renames + normalizes, batches, tokenizes, and moves
        to device."""
        import torch
        from lerobot.utils.constants import OBS_STATE

        frame: dict[str, Any] = {}
        for role, input_key in self.camera_mapping.items():
            img = scene.observations.images[role].data
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"SmolVLAAdapter: camera '{role}' image must be HxWx3 RGB; "
                    f"got shape {arr.shape}."
                )
            t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
            frame[input_key] = t.permute(2, 0, 1).contiguous()

        if self._uses_state:
            state = scene.observations.state
            if state is None:
                raise ValueError(
                    "SmolVLAAdapter: scene.observations.state is None."
                )
            vec = np.asarray(state.values, dtype=np.float32).reshape(-1)
            frame[OBS_STATE] = torch.from_numpy(vec)
        frame["task"] = scene.instruction or ""
        return frame

    def _action_chunk(self, scene: Scene) -> np.ndarray:
        import torch
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"SmolVLAAdapter.predict: {reason}")
        frame = self._build_frame(scene)
        batch = self._pre(frame)
        with torch.no_grad():
            chunk = self._policy.predict_action_chunk(batch)
        chunk = self._post(chunk)
        arr = np.asarray(chunk.detach().to("cpu", torch.float32).numpy())
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise RuntimeError(
                f"SmolVLAAdapter: expected a (1, chunk, action_dim) chunk; "
                f"got shape {arr.shape}."
            )
        return arr[0]   # (chunk_size, action_dim)

    def predict(self, scene: Scene) -> ActionResult:
        import torch
        torch.manual_seed(_ANALYSIS_SEED)   # deterministic sampling — see _ANALYSIS_SEED
        chunk = self._action_chunk(scene)
        action = chunk[0]
        return ActionResult(
            action=np.asarray(action, dtype=np.float32),
            action_dim=int(action.size),
            action_chunk=np.asarray(chunk, dtype=np.float32),
            metadata={"model": "smolvla", "chunk_shape": list(chunk.shape)},
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        # The attention query (last instruction token) is resolved from the
        # language attention mask inside extract_attention, not via explicit
        # token positions.
        return []

    # ----- attention ------------------------------------------------------

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        """Last-instruction-token attention over the image patches, from
        the SmolVLM2 prefix self-attention (the KV-cache-fill forward).

        ``query`` is ignored: the query is always the last valid
        instruction token of the prefix.
        """
        import torch
        import torch.nn.functional as F
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"SmolVLAAdapter.extract_attention: {reason}")

        policy = self._policy
        model = policy.model
        mwe = model.vlm_with_expert
        frame = self._build_frame(scene)
        batch = self._pre(frame)

        images, img_masks = policy.prepare_images(batch)
        state_t = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        img_emb_n: dict[str, int] = {}
        original_embed_image = mwe.embed_image

        def patched_embed_image(image):
            out = original_embed_image(image)
            img_emb_n.setdefault("n", int(out.shape[1]))
            return out

        num_att = mwe.num_attention_heads
        num_kv = mwe.num_key_value_heads
        groups = num_att // num_kv
        captured: list = []
        original_eager = mwe.eager_attention_forward

        def capturing_eager(attention_mask, batch_size, head_dim,
                            query_states, key_states, value_states):
            # Verbatim SmolVLMWithExpertModel.eager_attention_forward
            # (lerobot 0.5.1), capturing the softmax probs as a side effect
            # so the captured weights are exactly the ones used downstream.
            seq_len = key_states.shape[1]
            k = key_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv, groups, head_dim,
            ).reshape(batch_size, seq_len, num_kv * groups, head_dim)
            v = value_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv, groups, head_dim,
            ).reshape(batch_size, seq_len, num_kv * groups, head_dim)
            q = query_states.to(torch.float32).transpose(1, 2)
            k = k.to(torch.float32).transpose(1, 2)
            att_weights = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
            att_weights = att_weights.to(torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = F.softmax(masked, dim=-1)
            captured.append(probs.detach())
            p = probs.to(dtype=v.dtype)
            out = torch.matmul(p, v.permute(0, 2, 1, 3))
            out = out.permute(0, 2, 1, 3).reshape(
                batch_size, -1, num_kv * groups * head_dim,
            )
            return out

        mwe.embed_image = patched_embed_image
        mwe.eager_attention_forward = capturing_eager
        try:
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state_t,
            )
            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            with torch.no_grad():
                mwe.forward(
                    attention_mask=att_2d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=policy.config.use_cache,
                    fill_kv_cache=True,
                )
        finally:
            mwe.embed_image = original_embed_image
            mwe.eager_attention_forward = original_eager

        if "n" not in img_emb_n:
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: embed_image was not "
                "invoked; lerobot's SmolVLA prefix path may have changed."
            )
        if len(captured) != self._num_vlm_layers:
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: captured "
                f"{len(captured)} attention layers but the VLM has "
                f"{self._num_vlm_layers}."
            )

        num_img = img_emb_n["n"]
        side = int(round(num_img ** 0.5))
        if side * side != num_img:
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: image tokens per camera "
                f"({num_img}) are not a square grid; cannot reshape to a "
                "spatial map."
            )

        prefix_len = int(captured[0].shape[-1])
        add_special = bool(getattr(model, "add_image_special_tokens", False))

        image_token_ranges: dict[str, list[tuple[int, int]]] = {}
        image_grid_sides: dict[str, int] = {}
        cursor = 0
        for model_cam in self._present_model_cameras:
            role = self._modelcam_to_role[model_cam]
            if add_special:
                cursor += 1
            image_token_ranges[role] = [(cursor, cursor + num_img)]
            image_grid_sides[role] = side
            cursor += num_img
            if add_special:
                cursor += 1

        lang_start = cursor
        num_lang = int(lang_tokens.shape[1])
        lang_end = lang_start + num_lang
        if lang_end > prefix_len:
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: language block "
                f"[{lang_start}, {lang_end}) exceeds prefix length "
                f"{prefix_len}; token-layout assumption is wrong."
            )
        lang_valid = np.asarray(lang_masks[0].detach().cpu().numpy()).astype(bool)
        valid_idx = np.where(lang_valid)[0]
        if valid_idx.size == 0:
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: language attention mask "
                "is all-False — no instruction token to query."
            )
        query_pos = lang_start + int(valid_idx[-1])

        # Last-instruction-token row over keys, per layer and head, with the
        # content-independent (query-averaged) sink component removed.
        per_layer = []
        for probs in captured:
            a = probs[0]                                    # (H, S, S)
            row = a[:, query_pos, :].to(torch.float32).cpu().numpy()
            marg = a.to(torch.float32).mean(dim=1).cpu().numpy()
            per_layer.append(np.clip(row - marg, 0.0, None))
        weights = np.stack(per_layer, axis=0)               # (L, H, prefix_len)

        first_row = weights[0, 0, :]
        if np.allclose(first_row, first_row[0], atol=1e-9):
            raise RuntimeError(
                "SmolVLAAdapter.extract_attention: prefix attention is "
                "uniform across all keys — degenerate output, refusing to "
                "return it."
            )

        return AttentionMaps(
            weights=weights,
            query_position=query_pos,
            n_keys=prefix_len,
            image_token_ranges=image_token_ranges,
            image_grid_sides=image_grid_sides,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "n_vlm_layers": self._num_vlm_layers,
                "n_heads": self._n_heads,
                "tokens_per_image": num_img,
                "side_per_image": side,
                "query_token": "last instruction token (SmolVLM2 prefix)",
                "attention_source": (
                    "SmolVLA SmolVLM2 prefix self-attention: instruction "
                    "token -> image tokens (localization heads)"
                ),
            },
        )
