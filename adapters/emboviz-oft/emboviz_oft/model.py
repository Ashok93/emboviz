"""Adapter for OpenVLA-OFT (Optimized Fine-Tuning) checkpoints.

OFT is Stanford's enhanced fine-tuning recipe for OpenVLA, adding
parallel decoding + L1-regression action head + first-class proprioception
input. It significantly outperforms OpenVLA-7B on LIBERO benchmarks.

**Install (separate virtualenv required):**

OpenVLA-OFT pins a FORK of `transformers` (moojink/transformers-openvla-oft)
for bidirectional attention support. This fork is incompatible with both
mainline `transformers` and other adapters' transformers pins, so OFT
must live in its own virtualenv:

    git clone https://github.com/moojink/openvla-oft.git
    cd openvla-oft && pip install -e .
    pip install -e /path/to/emboviz   # add emboviz on top

Then construct the adapter with a checkpoint id (e.g.
"moojink/openvla-7b-oft-finetuned-libero-spatial").

Capabilities: INFERENCE. Internal-introspection (attention, hidden
states, patching) is not exposed through the OFT inference utilities;
capability-gated diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, AttentionMaps, Scene, TokenSelector
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel


_DEFAULT_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial"
_SPACE_TOKEN_ID = 29871   # Llama tokenizer's leading-space token


class OpenVLAOFTAdapter(VLAModel):
    """Wraps the openvla-oft inference utilities as a VLAModel.

    Construction:
        OpenVLAOFTAdapter()  # default LIBERO-spatial checkpoint
        OpenVLAOFTAdapter(checkpoint="moojink/openvla-7b-oft-finetuned-libero-10")
        OpenVLAOFTAdapter(checkpoint="...", unnorm_key="libero_10_no_noops",
                          num_images=2, use_proprio=True)
    """

    _CAPS = Capability.INFERENCE | Capability.ATTENTION

    # Same LLaMA-7B backbone as OpenVLA-7B; literature attention profile
    # is identical (LLaVA stage analysis applies to any LLaMA-2-based
    # VLA). See OpenVLAAdapter.ATTENTION_PROFILE for the source citations.
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.25, 0.75),
        "sink_top_pct_to_mask": 0.0,
        "literature_citation":
            "Same LLaMA-2 7B backbone as OpenVLA-7B; see that adapter "
            "for the source citations. LLaVA stage analysis "
            "(arXiv:2508.20279) for layer range; no image-patch sinks "
            "for LLaMA family.",
    }

    def __init__(
        self,
        checkpoint: str = _DEFAULT_REPO,
        unnorm_key: str = "libero_spatial_no_noops",
        num_images: int = 2,
        use_proprio: bool = True,
        use_l1_regression: bool = True,
        use_film: bool = False,
        center_crop: bool = True,
        wrist_camera: str = "wrist",
    ):
        self.wrist_camera = wrist_camera
        self.num_images = num_images
        try:
            # NOTE: we deliberately do NOT import from
            # ``experiments.robot.libero.run_libero_eval`` — that module hard-imports
            # ``libero.libero`` (the LIBERO simulator) at module load, which we
            # don't need for pure inference. We inline a minimal config dataclass
            # below that duck-types with what OFT's helpers actually read.
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_processor,
                get_proprio_projector,
                get_vla,
                get_vla_action,
            )
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
        except ImportError as e:
            raise ImportError(
                "OpenVLA-OFT requires the openvla-oft repository (separate venv).\n"
                "Setup:\n"
                "    git clone https://github.com/moojink/openvla-oft.git\n"
                "    cd openvla-oft && pip install -e .\n"
                "Then install emboviz on top of that environment."
            ) from e

        from dataclasses import dataclass as _dataclass
        from typing import Union as _Union

        @_dataclass
        class _InferenceConfig:
            """Minimal GenerateConfig replacement — only inference fields.

            Mirrors the fields OFT's helpers read from cfg. We do NOT inherit
            from the upstream GenerateConfig because that one carries LIBERO
            simulator references; we want this adapter to work on any source
            of Scenes, not just LIBERO sim rollouts.
            """
            model_family: str = "openvla"
            pretrained_checkpoint: _Union[str, "Path"] = ""
            use_l1_regression: bool = True
            use_diffusion: bool = False
            num_diffusion_steps_train: int = 50
            num_diffusion_steps_inference: int = 50
            use_film: bool = False
            num_images_in_input: int = 2
            use_proprio: bool = True
            center_crop: bool = True
            num_open_loop_steps: int = 8
            lora_rank: int = 32
            unnorm_key: _Union[str, "Path"] = ""
            load_in_8bit: bool = False
            load_in_4bit: bool = False
            # Set by override below to avoid the LIBERO-coupled TaskSuite enum.
            task_suite_name: str = "libero_spatial"
        GenerateConfig = _InferenceConfig

        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        self.use_proprio = use_proprio

        cfg = GenerateConfig(
            pretrained_checkpoint=checkpoint,
            use_l1_regression=use_l1_regression,
            use_diffusion=False,
            use_film=use_film,
            num_images_in_input=num_images,
            use_proprio=use_proprio,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=center_crop,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key=unnorm_key,
        )
        self._cfg = cfg

        self._vla = get_vla(cfg)
        self._processor = get_processor(cfg)
        self._action_head = get_action_head(cfg, llm_dim=self._vla.llm_dim)
        self._proprio_projector = (
            get_proprio_projector(cfg, llm_dim=self._vla.llm_dim, proprio_dim=PROPRIO_DIM)
            if use_proprio else None
        )
        self._get_vla_action = get_vla_action
        self._proprio_dim = PROPRIO_DIM
        self._action_dim = 7   # standard 7-DOF for OpenVLA-class

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.checkpoint.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # OFT consumes ``num_images`` cameras: the primary camera always,
        # plus the wrist camera when num_images >= 2. We declare BOTH so
        # the framework's Scene validator catches a missing wrist camera
        # at the boundary instead of silently feeding primary as wrist.
        cams = {"primary"}
        if self.num_images >= 2:
            cams.add(self.wrist_camera)
        return RequiredInputs(
            cameras=frozenset(cams),
            instruction=True,
            state=self.use_proprio,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"OpenVLAOFTAdapter.predict: {reason}")
        observation = self._build_observation(scene)
        actions = self._get_vla_action(
            self._cfg, self._vla, self._processor,
            observation, observation["task_description"],
            self._action_head, self._proprio_projector,
        )
        # `actions` is a chunk (chunk_len, action_dim); expose the full
        # chunk via action_chunk and the first row as the immediate action.
        chunk = np.asarray(actions, dtype=np.float32).reshape(-1, self._action_dim)
        action = chunk[0]
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "checkpoint": self.checkpoint,
                "unnorm_key": self.unnorm_key,
                "chunk_size": int(chunk.shape[0]),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []

    # ----- attention extraction -----------------------------------------

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        """Extract per-camera attention from OFT's language-model forward.

        Strategy: replicate the input-prep pipeline of ``predict_action``
        (vision encoder + proprio + multimodal-embedding stacking) and
        run the underlying ``language_model`` with ``output_attentions=True``
        instead of ``False``. We skip the action head entirely — attention
        comes out of the LM forward pass; we don't need to generate
        actions to inspect attention.

        Multimodal sequence layout when num_images=2 and use_proprio=True:
            [primary patches (P) | wrist patches (P) | proprio (1) |
             prompt tokens (NUM_PROMPT_TOKENS) |
             action tokens (NUM_ACTIONS_CHUNK * ACTION_DIM) |
             stop token]

        Returns one AttentionMaps with per-camera image-token slices so
        callers can compute per-camera heatmaps without re-running.

        Args:
            query: ``TokenSelector(relative="before_action")`` reads
                attention from the first action-token position (the
                conventional "what is the model looking at when deciding
                the first action" probe). ``"last"`` reads from the
                final position; ``"first"`` from BOS-equivalent.
        """
        import torch

        from prismatic.vla.constants import IGNORE_INDEX
        from experiments.robot.openvla_utils import normalize_proprio

        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"OpenVLAOFTAdapter.extract_attention: {reason}")
        observation = self._build_observation(scene)

        # Build prompt EXACTLY as get_vla_action does.
        prompt = (
            f"In: What action should the robot take to "
            f"{observation['task_description'].lower()}?\nOut:"
        )

        # Process primary image.
        from PIL import Image as _Image
        primary_pil = _Image.fromarray(observation["full_image"])
        device = next(self._vla.parameters()).device
        inputs = self._processor(prompt, primary_pil).to(device, dtype=torch.bfloat16)

        # If multi-image: process wrist + concat pixel_values along the
        # image-stream axis (same dim that get_vla_action concatenates).
        if self.num_images >= 2:
            wrist_pil = _Image.fromarray(observation["wrist_image"])
            wrist_inputs = self._processor(prompt, wrist_pil).to(device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1,
            )

        # Normalize proprio (predict_action does this via the helper).
        proprio = None
        if self.use_proprio:
            proprio_norm_stats = self._vla.norm_stats[self.unnorm_key]["proprio"]
            proprio = normalize_proprio(observation["state"], proprio_norm_stats)

        # ---- replicate the predict_action prep pipeline ----
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if not torch.all(input_ids[:, -1] == _SPACE_TOKEN_ID):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[_SPACE_TOKEN_ID]],
                                          device=input_ids.device,
                                          dtype=input_ids.dtype)),
                dim=1,
            )

        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        with torch.inference_mode():
            input_ids, attention_mask = self._vla._prepare_input_for_action_prediction(
                input_ids, attention_mask,
            )
            labels = self._vla._prepare_labels_for_action_prediction(labels, input_ids)

            input_embeddings = self._vla.get_input_embeddings()(input_ids)
            all_actions_mask = self._vla._process_action_masks(labels)

            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2],
            )
            projected_patch_embeddings = self._vla._process_vision_features(
                pixel_values, language_embeddings, use_film=False,
            )

            use_proprio = self.use_proprio and proprio is not None
            if use_proprio:
                proprio_tensor = torch.as_tensor(
                    proprio,
                    device=projected_patch_embeddings.device,
                    dtype=projected_patch_embeddings.dtype,
                )
                projected_patch_embeddings = self._vla._process_proprio_features(
                    projected_patch_embeddings, proprio_tensor, self._proprio_projector,
                )

            patches_per_image = self._vla.vision_backbone.get_num_patches()
            num_images_used = self._vla.vision_backbone.get_num_images_in_input()
            NUM_PATCHES = patches_per_image * num_images_used
            if use_proprio:
                NUM_PATCHES += 1

            # Zero out action-token embeddings (regression-path convention).
            all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
            input_embeddings = input_embeddings * ~all_actions_mask_3d

            multimodal_embeddings, multimodal_attention_mask = (
                self._vla._build_multimodal_attention(
                    input_embeddings, projected_patch_embeddings, attention_mask,
                )
            )

            # ---- the one knob we change vs predict_action: attention ON ----
            outputs = self._vla.language_model(
                inputs_embeds=multimodal_embeddings,
                attention_mask=multimodal_attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )

        full_seq = int(multimodal_embeddings.shape[1])
        first_action_pos = NUM_PATCHES + NUM_PROMPT_TOKENS

        # Resolve query position. "before_action" maps to the position
        # of the first action token — that's where the model decides
        # action[0], analogous to OpenVLA's "next-token" semantics.
        if query.position is not None:
            query_pos = int(query.position)
        elif query.relative == "last":
            query_pos = full_seq - 1
        elif query.relative == "first":
            query_pos = 0
        elif query.relative == "before_action":
            query_pos = first_action_pos
        elif query.word is not None:
            raise NotImplementedError(
                "OFT does not support word-anchored attention extraction "
                "yet (would need tokenizer-aware position search through "
                "the prompt span)."
            )
        else:
            query_pos = first_action_pos

        # Query-position attention to all keys, per layer & head, with the
        # CONTENT-INDEPENDENT attention-sink component removed: subtract the
        # query-averaged attention (any token attended-to regardless of query —
        # BOS/sink — cancels; query-specific grounding survives). Same pipeline
        # as OpenVLA/π0/GR00T. (Xiao et al. 2309.17453.)
        per_layer = []
        for layer_attn in outputs.attentions:
            a = layer_attn[0]                               # (n_heads, seq, seq)
            row = a[:, query_pos, :].float().cpu().numpy()  # (H, seq)
            marg = a.float().mean(dim=1).cpu().numpy()      # (H, seq) query-averaged (sink)
            per_layer.append(np.clip(row - marg, 0.0, None))
        weights = np.stack(per_layer, axis=0)

        # Per-camera image-token ranges. SigLIP+DINOv2 backbone is square.
        grid_side = int(np.sqrt(patches_per_image))
        if grid_side * grid_side != patches_per_image:
            raise RuntimeError(
                f"OFT vision_backbone produced {patches_per_image} patches; "
                f"expected a square grid (side² = patches). Refusing to "
                "fabricate a non-square grid_side."
            )

        # OFT cameras are single-tile per camera. The multimodal sequence
        # is laid out as: [BOS at position 0 | vision patches at
        # positions 1..1+P | proprio? | prompt tokens | action tokens |
        # stop]. Vision starts AFTER BOS at position 1. See
        # prismatic/extern/hf/modeling_prismatic.py::_build_multimodal_attention
        # ("insert embeddings after <BOS> token (1:)") in the openvla-oft
        # repo for proof.
        image_token_ranges: dict = {"primary": [(1, 1 + patches_per_image)]}
        image_grid_sides: dict = {"primary": grid_side}
        if self.num_images >= 2:
            image_token_ranges[self.wrist_camera] = [(
                1 + patches_per_image, 1 + 2 * patches_per_image,
            )]
            image_grid_sides[self.wrist_camera] = grid_side

        return AttentionMaps(
            weights=weights,
            query_position=query_pos,
            n_keys=full_seq,
            image_token_ranges=image_token_ranges,
            image_grid_sides=image_grid_sides,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "num_images":     num_images_used,
                "num_patches":    NUM_PATCHES,
                "num_prompt":     NUM_PROMPT_TOKENS,
                "first_action":   first_action_pos,
                "use_proprio":    use_proprio,
            },
        )

    # ----- helpers -------------------------------------------------------

    def _build_observation(self, scene: Scene) -> dict:
        """Format Scene into the dict OFT's `get_vla_action` expects.

        Strict: primary + (optionally) wrist must be present (already
        enforced by required_inputs.validate); state must be the exact
        ``PROPRIO_DIM`` shape OFT was trained on. No silent fallback.
        """
        primary = np.asarray(scene.observations.images["primary"].data, dtype=np.uint8)
        obs: dict = {"full_image": primary}
        if self.num_images >= 2:
            obs["wrist_image"] = np.asarray(
                scene.observations.images[self.wrist_camera].data, dtype=np.uint8,
            )
        if self.use_proprio:
            state_vals = np.asarray(
                scene.observations.state.values, dtype=np.float32,
            ).reshape(-1)
            if state_vals.size != self._proprio_dim:
                raise ValueError(
                    f"OpenVLA-OFT requires proprio dim {self._proprio_dim} "
                    f"but scene provides {state_vals.size}. Fix the dataset "
                    "adapter's state extractor — we never pad/truncate silently."
                )
            obs["state"] = state_vals
        if not scene.instruction:
            raise ValueError(
                "OpenVLA-OFT requires a non-empty instruction but "
                "scene.instruction is empty."
            )
        obs["task_description"] = scene.instruction
        return obs
