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

from emboviz_wire import ActionResult, AttentionMaps, Scene, TokenSelector
from emboviz_wire import Capability, RequiredInputs, VLAModel


# Demo values (the shipped GR00T-N1.7 checkpoint + DROID embodiment).
# These are documentation only — they are NOT constructor defaults: a
# silent default here would load the wrong model on the wrong embodiment.
# The adapter spec's default_actor_kwargs carries them explicitly, and a
# user's fine-tune overrides via the run config's model.kwargs.
DEMO_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
DEMO_MODEL_PATH = "nvidia/GR00T-N1.7-3B"

# Pinned RNG seed for prediction. GR00T denoises the action chunk from
# ``torch.randn(...)`` noise with no explicit generator, so an unseeded
# ``predict`` varies run-to-run. emboviz's comparative diagnostics measure how
# the action CHANGES under an intervention (mask the target, occlude a patch,
# swap a modality); against a stochastic predict that change is confounded with
# sampling jitter — the noise floor that forces "inconclusive" verdicts.
# Pinning the seed makes ``predict`` a deterministic function of its input, so
# the baseline and the intervention draw IDENTICAL noise and it cancels exactly
# in their difference (common random numbers): one forward pass per arm yields a
# clean causal measurement, with no averaging. The deployed policy is untouched
# — this is an analysis-time setting only.
_ANALYSIS_SEED = 0


class Gr00tAdapter(VLAModel):
    """Wraps `gr00t.policy.Gr00tPolicy` as an Emboviz `VLAModel`.

    Construction:
        Gr00tAdapter(model_path="nvidia/GR00T-N1.7-3B",
                     embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        Gr00tAdapter(model_path="...", embodiment_tag="GR1",
                     camera_mapping={"primary": "video.ego_view"})

    ``model_path`` and ``embodiment_tag`` are REQUIRED — there is no
    silent default. The embodiment selects how state/action are
    interpreted, so defaulting it is a wrong-model bug. ``camera_mapping``
    translates our Scene camera names into GR00T's embodiment-specific
    video keys; it auto-maps only when the embodiment declares exactly
    one video key (otherwise it is required).

    ``model_path`` accepts the base model, a user's single-checkpoint
    finetune of it, or a local dir — all loaded directly. ``model_subfolder``
    (optional) is only for multi-checkpoint repos that ship one checkpoint
    per subfolder (e.g. nvidia/GR00T-N1.7-LIBERO → libero_spatial/, …): set
    it to the subfolder and we fetch just that checkpoint. Left unset (the
    common case), loading is unchanged.
    """

    _CAPS = Capability.INFERENCE | Capability.ATTENTION

    # GR00T-N1.7 is DUAL-SYSTEM: a FROZEN Qwen3-VL backbone (System-2) perceives
    # the scene; a Diffusion-Transformer action head (System-1) generates the
    # action by CROSS-ATTENDING to the backbone's VL token embeddings. The
    # action-relevant "where is the model looking?" map is therefore the DiT's
    # IMAGE cross-attention (action queries → image VL tokens) — NOT the frozen
    # VLM's last-token self-attention (generic perception, dominated by Qwen
    # corner sinks, and not what conditions the action). See extract_attention.
    #
    # The image-cross blocks have no early/late "stage structure" like a VLM
    # transformer (each is action grounding, and attends to image tokens
    # directly so there are no positional sinks to avoid). All blocks are valid
    # candidates; image_weights_clean picks the block whose head-mean is most
    # concentrated on the object interior.
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.0, 1.0),
        "literature_citation":
            "GR00T N1 (arXiv:2503.14734): the System-1 DiT conditions on the "
            "backbone's intermediate-layer (config.select_layer) VL token "
            "embeddings via cross-attention. The 'where it looks to act' map is "
            "the DiT image cross-attention (action queries → image VL tokens), "
            "meaned over denoise steps + heads. Raw attention, no gradient.",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        device: str = "cuda:0",
        camera_mapping: Optional[dict[str, str]] = None,
        model_subfolder: Optional[str] = None,
    ):
        if not model_path:
            raise ValueError(
                "Gr00tAdapter requires an explicit model_path (e.g. "
                "'nvidia/GR00T-N1.7-3B' or your fine-tune) — no silent "
                "default. Set it in --model-kwargs / the run config's "
                "model.kwargs."
            )
        if not embodiment_tag:
            raise ValueError(
                "Gr00tAdapter requires an explicit embodiment_tag (e.g. "
                "'OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT') — no silent "
                "default. The embodiment determines how state/action are "
                "interpreted; defaulting it would run the wrong model."
            )

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
        # The checkpoint's normalization stats (norm_params) are keyed by the
        # embodiment's .value (e.g. LIBERO_PANDA.value == "libero_sim"), NOT
        # its .name — _state_key_dim looks up dims under this key.
        self._embodiment_value = embodiment_enum.value

        # Resolve model_path → a LOCAL checkpoint dir. Gr00tPolicy does
        # ``AutoModel.from_pretrained(Path(model_path))``, which loads a
        # local dir or a *plain* HF repo — it cannot reach an HF subfolder.
        # Multi-suite repos (e.g. nvidia/GR00T-N1.7-LIBERO) ship one
        # checkpoint per suite in a subfolder (libero_spatial/, ...). When
        # model_subfolder is set we fetch just that subfolder and point the
        # policy at the local copy. A plain repo / local dir passes through.
        resolved_path = model_path
        if model_subfolder:
            import os
            from huggingface_hub import snapshot_download
            local_root = snapshot_download(
                model_path, allow_patterns=f"{model_subfolder}/*",
            )
            resolved_path = os.path.join(local_root, model_subfolder)

        self.policy = Gr00tPolicy(
            model_path=resolved_path,
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
        import torch
        torch.manual_seed(_ANALYSIS_SEED)   # deterministic denoise — see _ANALYSIS_SEED
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

        if not chunks_per_key:
            # No declared action key produced an array. Refuse to emit a
            # zero-length action: a 0-dim vector would feed a meaningless
            # (or silently broadcast) comparison into the diagnostics. Raise
            # with the mismatch instead of fabricating an empty verdict.
            raise RuntimeError(
                f"GR00T returned no arrays for any declared action key "
                f"{list(self._action_keys)} (policy output keys: "
                f"{sorted(action_dict)}). Cannot build an action — check the "
                f"embodiment's action modality vs. the policy output."
            )
        # Align time dim across keys (take min T) then concat along D.
        min_t = min(c.shape[0] for c in chunks_per_key)
        chunk = np.concatenate(
            [c[:min_t] for c in chunks_per_key], axis=-1,
        ).astype(np.float32)
        action = chunk[0]

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
        """Per-camera attention from GR00T's System-1 DiT image cross-attention.

        GR00T-N1.7 is DUAL-SYSTEM: a FROZEN Qwen3-VL backbone (System-2,
        Cosmos-Reason2) perceives the scene, and a Diffusion-Transformer
        action head (System-1) generates the action by CROSS-ATTENDING to the
        backbone's vision-language token embeddings (the hidden states of
        ``config.select_layer``; see gr00t_n1d7.py: the DiT is called with
        ``encoder_hidden_states = vl_embeds`` and an ``image_mask``).

        So the action-relevant "where is the model looking?" signal is the
        DiT's IMAGE cross-attention — action/noise queries attending to the
        image VL tokens — NOT the backbone's last-text-token self-attention.
        The frozen VLM's self-attention is generic perception attention
        dominated by Qwen's corner sinks; it is NOT what conditions the action
        (visualizing it produced misleading corner blobs). The DiT
        cross-attention is the grounded, action-relevant map (GR00T N1 paper,
        arXiv:2503.14734: the DiT conditions on intermediate-layer VLM
        embeddings via cross-attention).

        ``AlternateVLDiT`` (dit.py) interleaves action self-attention,
        text-cross and image-cross blocks; we hook ``attn1`` on the image-cross
        blocks, capture the action-query → VL-token probabilities across the
        denoise steps, mean over steps + action queries (keeping heads), and map
        the image-token columns to per-camera patch grids. The diffusion noise is
        SEEDED so the map is reproducible. ``query`` is unused: the DiT's queries
        ARE the action tokens, not a text position.

        CAVEAT (load-bearing): this is GR00T's DiT MOTOR pathway. Unlike
        single-stack VLAs (OpenVLA/OFT/π0) whose action flows THROUGH the VLM and
        yields focused object attention, GR00T generates actions in a SEPARATE
        diffusion head whose cross-attention is the spatially DISPERSED motor
        program — it is NOT a reliable object localizer. This is the documented
        "dispersed VLA attention" phenomenon (ReconVLA arXiv:2508.10333; survey
        arXiv:2507.10672) and matches the GR00T-N1.5 mechanistic study
        (arXiv:2603.19233: the expert pathway encodes the motor program; goals
        live in VLM FEATURES, not attention). Read it as "where the action
        pathway attends across the workspace," not a tight object map. The VLM's
        own last-token self-attention is worse (frozen, attention-sink dominated),
        and its instruction↔image attention is likewise dispersed. See README +
        LITERATURE.md.

        Multi-camera handling: image tokens are INLINE in ``input_ids`` (marked
        by ``image_token_id``, one id per merged patch); ``image_grid_thw`` gives
        per-image ``(T, H, W)`` so each image consumes ``T*(H//merge)*(W//merge)``
        tokens. Tiles map to cameras in ``video.modality_keys`` order. The VL
        token sequence shares the ``input_ids`` length, so these column indices
        index the DiT cross-attention keys directly.
        """
        import torch

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Gr00tAdapter.extract_attention: {reason}")

        observation = self._build_observation(scene)
        model = self.policy.model
        qwen_model = model.backbone.model        # Qwen3-VL VLM (System-2)
        dit = model.action_head.model            # AlternateVLDiT (System-1)

        # Capture the VLM input_ids / image_grid_thw only — needed to map image
        # tokens back to per-camera patch grids. We do NOT need the VLM's own
        # attention; the action-grounding signal is the DiT cross-attention,
        # captured below. The wrapper leaves the return unchanged.
        captured_meta: dict = {"input_ids": None, "image_grid_thw": None}
        original_qwen_forward = qwen_model.forward

        def meta_capture_forward(*args, **kwargs):
            captured_meta["input_ids"] = kwargs.get("input_ids", args[0] if args else None)
            captured_meta["image_grid_thw"] = kwargs.get("image_grid_thw")
            return original_qwen_forward(*args, **kwargs)

        # AlternateVLDiT (gr00t/model/modules/dit.py) interleaves blocks:
        #   idx % 2 == 1                       -> action self-attention
        #   idx % 2 == 0 and idx % (2*N) == 0  -> cross-attn to TEXT tokens
        #   idx % 2 == 0 and idx % (2*N) != 0  -> cross-attn to IMAGE tokens
        # (N = attend_text_every_n_blocks). We hook attn1 on the IMAGE-cross
        # blocks and record the (head, action-query, vl-token) probabilities at
        # every denoise step.
        n_text_every = int(getattr(dit, "attend_text_every_n_blocks", 2))
        image_block_idxs = [
            i for i in range(len(dit.transformer_blocks))
            if (i % 2 == 0) and (i % (2 * n_text_every) != 0)
        ]
        if not image_block_idxs:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: no DiT image-cross-attention "
                f"blocks found (attend_text_every_n_blocks={n_text_every}, "
                f"n_blocks={len(dit.transformer_blocks)})."
            )
        n_blocks = len(image_block_idxs)
        dit_attns: list = []   # call order: [step0: blk0..blkN][step1: ...]...

        class _CaptureCross:
            """Diffusers AttentionProcessor: record the image-cross attention
            probabilities (honouring the encoder image-mask), then run the real
            attention so ``get_action`` proceeds unchanged."""

            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, temb=None, **kwargs):
                q = attn.to_q(hidden_states)
                ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
                if encoder_hidden_states is not None and attn.norm_cross:
                    ehs = attn.norm_encoder_hidden_states(ehs)
                k = attn.to_k(ehs)
                v = attn.to_v(ehs)
                q = attn.head_to_batch_dim(q)
                k = attn.head_to_batch_dim(k)
                v = attn.head_to_batch_dim(v)
                # Compute UNMASKED scores (mask=None): the DiT's image-cross
                # mask is a BOOL tensor for the SDPA path, but get_attention_scores
                # (baddbmm) needs an additive float mask. We only ever USE the
                # image-token columns afterward, and softmax over image keys is
                # identical up to a constant factor whether or not text keys are
                # masked (exp(s_i)/Z_all vs exp(s_i)/Z_img) — so the per-camera
                # spatial pattern (and the interior-fraction block selection) is
                # unchanged after normalization. Avoids fragile mask plumbing.
                probs = attn.get_attention_scores(q, k, None)
                bh, tq, tk = probs.shape
                h = attn.heads
                dit_attns.append(
                    probs.detach().float().cpu().reshape(bh // h, h, tq, tk)[0].numpy()
                )
                out = torch.bmm(probs, v)
                out = attn.batch_to_head_dim(out)
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)
                return out

        original_procs = {}
        for i in image_block_idxs:
            m = dit.transformer_blocks[i].attn1
            original_procs[i] = m.processor
            m.set_processor(_CaptureCross())
        qwen_model.forward = meta_capture_forward
        try:
            with torch.inference_mode():
                # GR00T denoises the action chunk from RANDOM noise
                # (gr00t_n1d7.py get_action_with_features: ``actions =
                # torch.randn(...)``, no generator). Seed it so the captured
                # cross-attention is REPRODUCIBLE — a diagnostic must not vary
                # run-to-run on an unseeded RNG. (GR00T-specific: OFT/OpenVLA/π0
                # attention is the VLM forward, already deterministic.)
                torch.manual_seed(0)
                _ = self.policy.get_action(observation)
        finally:
            qwen_model.forward = original_qwen_forward
            for i, proc in original_procs.items():
                dit.transformer_blocks[i].attn1.set_processor(proc)

        input_ids = captured_meta["input_ids"]
        image_grid_thw = captured_meta["image_grid_thw"]
        if input_ids is None or image_grid_thw is None or not dit_attns:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: DiT image-cross attention or "
                "VLM meta not captured (the cross-attention processor or the "
                "qwen forward hook did not fire)."
            )

        # n_keys is the VL-token sequence length (== len(input_ids);
        # backbone_features shares the input_ids length), so image-token columns
        # line up with the per-camera map built below.
        n_steps = len(dit_attns) // n_blocks
        if n_steps == 0:
            raise RuntimeError(
                "Gr00tAdapter.extract_attention: fewer than one full denoise "
                "sweep of the image-cross blocks was captured."
            )
        n_heads = int(dit_attns[0].shape[0])
        n_keys = int(dit_attns[0].shape[2])
        # Aggregate per image-cross block: mean over denoise steps + action
        # queries, keep heads -> weights (n_blocks, H, n_keys). image_weights_clean
        # then selects the most-interior block and means over heads.
        # NOTE (load-bearing): this is GR00T's DiT (System-1) action->image
        # cross-attention — the MOTOR pathway. It is known to be spatially
        # DISPERSED, NOT a tight object localizer (see the class docstring +
        # README + LITERATURE.md). Step aggregation barely changes that; mean is
        # the stable, representative choice.
        per_block = []
        for b in range(n_blocks):
            steps = [dit_attns[s * n_blocks + b].mean(axis=1) for s in range(n_steps)]  # (H, n_keys)
            per_block.append(np.mean(steps, axis=0))      # (H, n_keys)
        weights = np.stack(per_block, axis=0)             # (n_blocks, H, n_keys)

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
            query_position=0,   # N/A: DiT queries are the action tokens (pooled), not a text position
            n_keys=n_keys,
            image_token_ranges=image_token_ranges,
            image_grid_sides=image_grid_sides,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "attention_source":
                    "GR00T System-1 DiT image cross-attention (MOTOR pathway): "
                    "action tokens (mean) -> image VL tokens, mean over (seeded) "
                    "denoise steps, per image-cross block x head.",
                "caveat":
                    "DISPERSED motor-pathway attention, NOT a reliable object "
                    "localizer (dispersed-VLA-attention: ReconVLA arXiv:2508.10333; "
                    "GR00T-N1.5 mechanistic study arXiv:2603.19233). See README.",
                "query_token": "DiT action/noise tokens (pooled)",
                "n_dit_image_blocks": int(n_blocks),
                "n_denoise_steps": int(n_steps),
                "n_heads": int(n_heads),
                "image_grid_thw": thw_np.tolist(),
                "merge_size": merge_size,
                "n_image_tokens_total": int(image_positions.size),
            },
        )

    def extract_attention_trace(self, scene: Scene):
        """Per-denoise-step, per-head DiT image cross-attention (action → image).

        GR00T's System-1 DiT cross-attends to the image VL tokens to denoise the
        action chunk; like π0, that attention sharpens across denoise steps. We
        capture the image-cross-attention at EVERY denoise step, keep the head
        axis, and map to the per-camera grid — so the visualizer can scrub the
        denoise steps and toggle heads. No averaging over steps, no head-mean.
        """
        import torch
        from emboviz_wire import AttentionTrace

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Gr00tAdapter.extract_attention_trace: {reason}")

        observation = self._build_observation(scene)
        model = self.policy.model
        qwen_model = model.backbone.model
        dit = model.action_head.model

        captured_meta: dict = {"input_ids": None, "image_grid_thw": None}
        original_qwen_forward = qwen_model.forward

        def meta_capture_forward(*args, **kwargs):
            captured_meta["input_ids"] = kwargs.get("input_ids", args[0] if args else None)
            captured_meta["image_grid_thw"] = kwargs.get("image_grid_thw")
            return original_qwen_forward(*args, **kwargs)

        n_text_every = int(getattr(dit, "attend_text_every_n_blocks", 2))
        image_block_idxs = [
            i for i in range(len(dit.transformer_blocks))
            if (i % 2 == 0) and (i % (2 * n_text_every) != 0)
        ]
        if not image_block_idxs:
            raise RuntimeError("extract_attention_trace: no DiT image-cross blocks found.")
        n_blocks = len(image_block_idxs)

        dit_attns: list = []   # appended in call order: [step0 blocks...][step1 blocks...]...

        class _Capture:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, temb=None, **kwargs):
                q = attn.to_q(hidden_states)
                ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
                if encoder_hidden_states is not None and attn.norm_cross:
                    ehs = attn.norm_encoder_hidden_states(ehs)
                k = attn.to_k(ehs)
                v = attn.to_v(ehs)
                q = attn.head_to_batch_dim(q)
                k = attn.head_to_batch_dim(k)
                v = attn.head_to_batch_dim(v)
                probs = attn.get_attention_scores(q, k, None)
                bh, tq, tk = probs.shape
                h = attn.heads
                dit_attns.append(probs.detach().float().cpu().reshape(bh // h, h, tq, tk)[0].numpy())
                out = torch.bmm(probs, v)
                out = attn.batch_to_head_dim(out)
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)
                return out

        original_procs = {}
        for i in image_block_idxs:
            m = dit.transformer_blocks[i].attn1
            original_procs[i] = m.processor
            m.set_processor(_Capture())
        qwen_model.forward = meta_capture_forward
        try:
            with torch.inference_mode():
                _ = self.policy.get_action(observation)
        finally:
            qwen_model.forward = original_qwen_forward
            for i, proc in original_procs.items():
                dit.transformer_blocks[i].attn1.set_processor(proc)

        input_ids = captured_meta["input_ids"]
        image_grid_thw = captured_meta["image_grid_thw"]
        if input_ids is None or image_grid_thw is None or not dit_attns:
            raise RuntimeError("extract_attention_trace: VLM meta / DiT attention not captured.")

        n_steps = len(dit_attns) // n_blocks
        if n_steps == 0:
            raise RuntimeError("extract_attention_trace: <1 full denoise sweep of image blocks.")
        n_heads = int(dit_attns[0].shape[0])
        # per step: mean over action queries (Tq) then mean over image blocks → (H, Tk)
        per_step = []
        for s in range(n_steps):
            block_maps = [dit_attns[s * n_blocks + b].mean(axis=1) for b in range(n_blocks)]  # (H,Tk)
            per_step.append(np.mean(block_maps, axis=0))
        per_step = np.stack(per_step, axis=0)   # (n_steps, H, Tk)

        # per-camera image-token columns (same mapping as extract_attention)
        image_token_id = qwen_model.config.image_token_id
        ids_row = input_ids[0].cpu().numpy()
        image_positions = np.where(ids_row == image_token_id)[0]
        try:
            merge_size = int(qwen_model.config.vision_config.spatial_merge_size)
        except AttributeError:
            merge_size = 2
        thw_np = image_grid_thw.cpu().numpy().astype(int)
        video_keys = list(self._modality_configs["video"].modality_keys)
        gr00t_key_to_user = {v: k for k, v in self._camera_mapping.items()}
        num_cameras = len(video_keys)

        ranges: dict[str, list] = {}
        sides: dict[str, int] = {}
        cursor = 0
        for tile_i, (t, h, w) in enumerate(thw_np):
            tpt = int(t * (h // merge_size) * (w // merge_size))
            run_start = int(image_positions[cursor])
            run_end = int(image_positions[cursor + tpt - 1]) + 1
            cursor += tpt
            h_eff = int(h // merge_size)
            user_cam = gr00t_key_to_user.get(video_keys[tile_i % num_cameras], video_keys[tile_i % num_cameras])
            ranges.setdefault(user_cam, []).append((run_start, run_end))
            sides[user_cam] = h_eff

        per_camera, grid_sides = {}, {}
        for cam, tiles in ranges.items():
            side = sides[cam]
            tile_maps = [per_step[:, :, s:e].reshape(n_steps, n_heads, side, side) for s, e in tiles]
            per_camera[cam] = np.sum(tile_maps, axis=0) if len(tile_maps) > 1 else tile_maps[0]
            grid_sides[cam] = side

        return AttentionTrace(
            per_camera=per_camera, grid_sides=grid_sides,
            n_steps=n_steps, n_heads=n_heads,
            source="gr00t DiT image cross-attention",
            query_desc="action tokens (mean) × image-cross blocks (mean), per head",
            metadata={"image_grid_thw": thw_np.tolist(), "merge_size": merge_size,
                      "n_denoise_steps": n_steps, "n_image_blocks": n_blocks},
        )

    def _state_key_dim(self, state_key: str) -> int:
        """Dim of a GR00T state key, read from the checkpoint's normalization
        metadata — the single source of truth. We REFUSE to guess.

        The processor exposes per-(embodiment, modality, key) stats in
        ``state_action_processor.norm_params``, keyed by the embodiment's
        ``.value`` (e.g. LIBERO_PANDA.value == "libero_sim"), each group
        carrying a ``"dim"`` field (and matching min/max arrays). This is the
        value ``_build_observation`` cross-checks each modality.json-declared
        state slice against. If the path/key is absent we RAISE rather than
        fabricate a dim — a wrong dim would mis-route the state vector.
        """
        try:
            norm = self.policy.processor.state_action_processor.norm_params
        except AttributeError as e:
            raise RuntimeError(
                f"Cannot read GR00T normalization metadata for state key "
                f"'{state_key}': {type(e).__name__}: {e}. The "
                "policy.processor.state_action_processor.norm_params path is "
                "absent in this gr00t version — update this introspection path."
            ) from e
        emb = norm.get(self._embodiment_value) if norm else None
        if not emb or "state" not in emb or state_key not in emb["state"]:
            avail = sorted(emb["state"]) if (emb and "state" in emb) else []
            raise RuntimeError(
                f"GR00T norm_params has no state entry for key '{state_key}' "
                f"under embodiment '{self._embodiment_value}' (available state "
                f"keys: {avail}). Cannot determine its dim; refusing to guess. "
                "Check the embodiment_tag matches the checkpoint."
            )
        group = emb["state"][state_key]
        dim = group.get("dim")
        if dim is None:
            mn = group.get("min")
            dim = len(mn) if mn is not None else None
        if dim is None:
            raise RuntimeError(
                f"GR00T norm_params state entry for '{state_key}' has neither "
                "a 'dim' field nor a length-bearing 'min'; cannot read its dim."
            )
        return int(dim)

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
