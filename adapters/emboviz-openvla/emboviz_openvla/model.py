"""OpenVLA-7B adapter — implements the ``VLAModel`` protocol.

This is the ONLY file where OpenVLA-specific code lives. Adding a new
VLA family means writing an analogous adapter package; no other
module in emboviz mentions OpenVLA by name.

Heavy imports (torch, transformers) are deferred to ``__init__`` so
just importing ``emboviz_openvla.model`` doesn't load 7B parameters —
critical because Ray imports the actor class on the driver side
(emboviz core's main venv, which has no torch) before materialising
the instance on the worker side (inside the runtime venv).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz_wire import (
    ActionResult,
    AttentionMaps,
    FFNActivations,
    HiddenStates,
    ImageLike,
    Scene,
    TokenSelector,
)
from emboviz_wire import (
    Capability,
    NotSupported,
    RequiredInputs,
    VLAModel,
)


HF_REPO_DEFAULT = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
SPACE_TOKEN_ID = 29871  # Llama tokenizer's leading-space token


class OpenVLAAdapter(VLAModel):
    """Adapter for OpenVLA-7B (and its OFT variants — change `hf_repo`)."""

    _CAPS = (
        Capability.INFERENCE
        | Capability.ATTENTION
        | Capability.HIDDEN_STATES
        | Capability.FFN_ACTIVATIONS
        | Capability.FFN_VALUE_VECTORS
        | Capability.VOCAB_LOGIT_LENS
        | Capability.NEURON_ABLATION
        | Capability.ACTIVATION_PATCHING
    )

    # Per-model attention-extraction profile, used by
    # ``AttentionMaps.image_weights_clean()`` to apply the literature-
    # backed default visualization for THIS backbone (instead of a
    # one-size-fits-all heuristic). See the citation field for the
    # source paper(s) that grounded each value.
    ATTENTION_PROFILE = {
        # LLaVA stage analysis ("How Multimodal LLMs Solve Image Tasks",
        # arXiv:2508.20279) finds that LLaMA-based VLMs (LLaVA-1.5, OpenVLA
        # via Llama-2 7B) have visual-grounding heads concentrated in
        # mid-layers 8-23 of 32; early layers do token grouping, late
        # layers do prediction summarization. Middle half = literature-
        # backed default for spatial attention extraction.
        "recommended_layer_range_fraction": (0.25, 0.75),
        # (LLaMA-2 7B has no documented image-patch spatial sinks — the
        # documented sink is the BOS *text* token, not image positions —
        # so the layer-adaptive map needs no per-cell sink masking here.)
        "literature_citation":
            "Layer range: 'How Multimodal LLMs Solve Image Tasks' "
            "(arXiv:2508.20279) — visual grounding in mid-layers. "
            "Sink 0%: LLaMA-2 has no documented image-patch spatial "
            "sinks; the BOS-token sink (Xiao et al. 'Efficient Streaming "
            "Language Models with Attention Sinks', arXiv:2309.17453) "
            "applies to text positions, not image patches.",
    }

    def __init__(
        self,
        hf_repo: str = "",
        unnorm_key: str = DEFAULT_UNNORM_KEY,
        device: str = "cuda",
        attn_implementation: str = "eager",  # eager-attn keeps gradients
    ):
        if not hf_repo:
            raise ValueError(
                "OpenVLAAdapter requires an explicit hf_repo (e.g. "
                "'openvla/openvla-7b' or your fine-tune) — no silent "
                "default. Set it in --model-kwargs / the run config's "
                "model.kwargs."
            )
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.hf_repo = hf_repo
        self.unnorm_key = unnorm_key
        self.device = device
        self._dtype = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            hf_repo,
            attn_implementation=attn_implementation,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # Cache invariants we'll need across diagnostics
        self._action_dim = int(self.model.get_action_dim(self.unnorm_key))
        # action_scale is OPTIONAL — some checkpoints lack norm stats.
        # If get_action_stats raises a known "no such norm stat" error
        # we set action_scale to None with a warning so the user knows
        # normalized_l2 metrics will fall back to plain L2. Any other
        # error (e.g. malformed checkpoint) propagates.
        import warnings as _warnings
        try:
            stats = self.model.get_action_stats(self.unnorm_key)
            self._action_scale = (
                np.asarray(stats["q99"]) - np.asarray(stats["q01"])
            ).astype(np.float32)
        except (KeyError, AttributeError, ValueError) as e:
            _warnings.warn(
                f"OpenVLA action_scale unavailable for unnorm_key="
                f"'{self.unnorm_key}': {type(e).__name__}: {e}. "
                "Normalized-L2 metrics will fall back to plain L2.",
                stacklevel=2,
            )
            self._action_scale = None

    # ---- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.hf_repo.split("/")[-1]

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        # OpenVLA-7B consumes one primary RGB camera + a text instruction.
        # It ignores state/gripper/action_history — which is precisely why
        # the state-side diagnostics are interesting on it.
        return RequiredInputs(
            cameras=frozenset({"primary"}),
            instruction=True,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_scale(self) -> Optional[np.ndarray]:
        return self._action_scale

    @property
    def num_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    @property
    def num_heads(self) -> int:
        return self.model.language_model.config.num_attention_heads

    @property
    def hidden_dim(self) -> int:
        return self.model.language_model.config.hidden_size

    # ---- inference -----------------------------------------------------

    def _validated_inputs(self, scene: Scene) -> tuple:
        """Validate the scene against required_inputs; return (image, instruction).

        Centralises the strict-contract check so every public entrypoint
        fails the same way on missing primary camera or empty instruction.
        Never silently substitutes "".
        """
        reason = self.required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"OpenVLAAdapter: {reason}")
        return scene.observations.images["primary"].data, scene.instruction

    def predict(self, scene: Scene) -> ActionResult:
        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)
        action, tokens = self._generate_action(ids, pixel_values)
        return ActionResult(
            action=action.astype(np.float32),
            action_dim=self._action_dim,
            action_tokens=tokens.detach().cpu().numpy(),
            metadata={
                "prompt_len": int(ids.shape[1]),
                "unnorm_key": self.unnorm_key,
            },
        )

    # ---- attention extraction ------------------------------------------

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        import torch

        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                pixel_values=pixel_values,
                output_attentions=True,
            )
        # Resolve which query position to read from.
        query_pos = self._resolve_query_position(query, ids, outputs.logits.shape[1])

        full_seq = outputs.attentions[0].shape[-1]
        text_seq = ids.shape[1]
        n_image = full_seq - text_seq
        grid_side = int(np.sqrt(n_image))

        # Query-position attention to all keys, per layer & head, with the
        # CONTENT-INDEPENDENT attention-sink component removed: subtract the
        # query-averaged attention so any token attended-to regardless of query
        # (BOS/sink) cancels and the query-specific grounding survives. (Xiao
        # et al. 2309.17453; near no-op for LLaMA, which has weak image sinks.)
        per_layer_per_head = []
        for layer_attn in outputs.attentions:
            a = layer_attn[0]                               # (n_heads, seq, seq)
            row = a[:, query_pos, :].float().cpu().numpy()  # (H, seq)
            marg = a.float().mean(dim=1).cpu().numpy()      # (H, seq) query-averaged (sink)
            per_layer_per_head.append(np.clip(row - marg, 0.0, None))
        weights = np.stack(per_layer_per_head, axis=0)  # (L, H, n_keys)
        # OpenVLA is single-camera (the "primary" alias), single-tile. The
        # image-token slice is one contiguous run starting at position 1.
        return AttentionMaps(
            weights=weights,
            query_position=int(query_pos),
            n_keys=full_seq,
            image_token_ranges={"primary": [(1, 1 + n_image)]},
            image_grid_sides={"primary": grid_side},
            metadata={"attention_profile": self.ATTENTION_PROFILE},
        )

    # ---- hidden states + FFN activations -------------------------------

    def extract_hidden_states(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> HiddenStates:
        import torch
        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        full_seq = outputs.logits.shape[1]
        query_pos = self._resolve_query_position(query, ids, full_seq)
        hs_list = []
        for li in layer_indices:
            h = outputs.hidden_states[li]  # (1, seq, hidden)
            hs_list.append(h[0, query_pos].float().cpu().numpy())
        return HiddenStates(
            states=np.stack(hs_list, axis=0),
            query_position=int(query_pos),
            layer_indices=list(layer_indices),
            hidden_dim=self.hidden_dim,
        )

    def extract_ffn_activations(
        self, scene: Scene, layer_indices: list[int], query: TokenSelector,
    ) -> FFNActivations:
        import torch
        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)

        captured: dict[int, torch.Tensor] = {}
        handles = []
        layers = self.model.language_model.model.layers

        def make_hook(li):
            def hook(module, inputs, output):
                captured[li] = inputs[0].detach()
            return hook

        for li in layer_indices:
            h = layers[li].mlp.down_proj.register_forward_hook(make_hook(li))
            handles.append(h)
        try:
            with torch.inference_mode():
                outputs = self.model(input_ids=ids, pixel_values=pixel_values)
        finally:
            for h in handles:
                h.remove()

        full_seq = outputs.logits.shape[1]
        query_pos = self._resolve_query_position(query, ids, full_seq)   # already multimodal
        by_layer = {
            li: captured[li][0, query_pos].float().cpu().numpy()
            for li in layer_indices
        }
        return FFNActivations(by_layer=by_layer, query_position=int(query_pos))

    # ---- FFN value vectors + vocab logit lens --------------------------

    def get_ffn_value_vector_norms(self, layer_indices: list[int]) -> dict[int, np.ndarray]:
        layers = self.model.language_model.model.layers
        out: dict[int, np.ndarray] = {}
        for li in layer_indices:
            # down_proj.weight: (hidden_dim, intermediate_dim) — each COLUMN
            # is a value vector pointing into residual-stream space.
            w = layers[li].mlp.down_proj.weight
            out[li] = w.norm(dim=0).detach().float().cpu().numpy()
        return out

    def get_ffn_value_vectors(self, layer_idx: int) -> np.ndarray:
        """(intermediate_dim, hidden_dim) — columns of down_proj.weight."""
        w = self.model.language_model.model.layers[layer_idx].mlp.down_proj.weight
        # transpose so each ROW is one value vector
        return w.detach().float().cpu().numpy().T

    def project_to_vocab(self, vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Logit-lens projection onto the LLM's vocabulary."""
        import torch
        with torch.no_grad():
            embed = self.model.language_model.get_input_embeddings().weight
            v = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(embed.device, dtype=embed.dtype)
            scores = embed @ v          # (vocab_size,)
            top = scores.topk(top_k)
        ids = top.indices.cpu().tolist()
        vals = top.values.float().cpu().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(ids)
        return [(t.replace("▁", " ").strip() if t else "<?>", float(v)) for t, v in zip(tokens, vals)]

    # ---- residual-stream patching --------------------------------------

    def predict_with_residual_patch(
        self, scene: Scene, patches: dict, patch_position=None,
    ):
        """Patch the residual-stream output of each named layer at `patch_position`
        with the provided vector, then run inference.

        Implementation: register forward hooks on each Llama decoder layer
        we need to patch. Each hook overwrites the layer's output[0]
        (the residual output) at `patch_position` with the patch tensor.
        """
        import torch
        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)
        layers = self.model.language_model.model.layers

        # `patches` keys are layer indices; values are (hidden_dim,) numpy arrays.
        # Convert once, on device, in model dtype.
        patch_tensors: dict[int, torch.Tensor] = {
            int(li): torch.as_tensor(v, device=self.device, dtype=self._dtype)
            for li, v in patches.items()
        }

        # The PATCH POSITION is the absolute multimodal index of the
        # action-prediction position (= last token of the prefix). We compute
        # it here as `prefix_full_seq − 1` by doing one cheap forward pass to
        # measure the multimodal sequence length. This makes patching
        # robust to:
        #   • generation calling forward multiple times (one per action token)
        #   • KV-cache enabled forwards where seq_len = 1 in continuation steps
        #   • non-KV forwards where seq_len grows each step
        with torch.no_grad():
            probe = self.model(input_ids=ids, pixel_values=pixel_values)
        prefix_len_mm = int(probe.logits.shape[1])
        target_pos = (
            patch_position if (patch_position is not None and patch_position >= 0)
            else prefix_len_mm - 1
        )

        # Hook: patch THIS layer's residual output at the target absolute
        # position. The hook fires on every forward pass; we explicitly skip
        # if the current forward's sequence doesn't contain target_pos
        # (continuation forwards during generation see a small slice; the
        # patched activation has already been committed to the KV-cache so
        # we don't need to re-apply).
        handles = []

        def make_hook(li, vec, target):
            # `seen_prefix` flips True on the first multi-token forward (prefix).
            state = {"applied": False}

            def hook(module, inputs, output):
                if state["applied"]:
                    return None        # passthrough on continuation steps
                h_out = output[0] if isinstance(output, tuple) else output
                seq_len = h_out.shape[1]
                # Patch only on the prefix-processing forward (seq_len > 1).
                # Single-token forwards during generation skip — the patched
                # residual has already been baked into the KV-cache.
                if seq_len < 2 or target >= seq_len:
                    return None
                h_new = h_out.clone()
                h_new[:, target, :] = vec
                state["applied"] = True
                if isinstance(output, tuple):
                    return (h_new,) + tuple(output[1:])
                return h_new

            return hook

        for li, vec in patch_tensors.items():
            handles.append(
                layers[li].register_forward_hook(make_hook(li, vec, target_pos))
            )

        try:
            action, tokens = self._generate_action(ids, pixel_values)
        finally:
            for h in handles:
                h.remove()

        return ActionResult(
            action=action.astype(np.float32),
            action_dim=self._action_dim,
            action_tokens=tokens.detach().cpu().numpy(),
            metadata={"patch_position_mm": target_pos, "prefix_len_mm": prefix_len_mm},
        )

    # ---- neuron ablation -----------------------------------------------

    def predict_with_neuron_ablation(
        self, scene: Scene, ablations: dict[tuple[int, int], float],
    ) -> ActionResult:
        import torch
        image, instruction = self._validated_inputs(scene)
        ids, pixel_values = self._tokenize(image, instruction)
        layers = self.model.language_model.model.layers
        handles = []

        def make_hook(li: int, neuron_overrides: dict[int, float]):
            def hook(module, inputs, output):
                # `inputs[0]` is (B, seq, intermediate_dim) entering down_proj
                t = inputs[0].clone()
                for n, val in neuron_overrides.items():
                    t[..., n] = val
                return (t,) + tuple(inputs[1:])
            return hook

        by_layer: dict[int, dict[int, float]] = {}
        for (li, ni), val in ablations.items():
            by_layer.setdefault(li, {})[ni] = val
        for li, ovr in by_layer.items():
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(make_hook(li, ovr)))

        try:
            action, tokens = self._generate_action(ids, pixel_values)
        finally:
            for h in handles:
                h.remove()

        return ActionResult(
            action=action.astype(np.float32),
            action_dim=self._action_dim,
            action_tokens=tokens.detach().cpu().numpy(),
        )

    # ---- tokenization helpers ------------------------------------------

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        """Find sub-token positions of `word` in the PROMPT (after templating)."""
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        tok_ids = self.processor.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(tok_ids)
        needle = word.lower().strip()

        positions: list[int] = []
        i = 0
        while i < len(tokens):
            t = tokens[i] or ""
            if t.startswith("▁"):
                j = i
                piece = t.replace("▁", "")
                while j + 1 < len(tokens) and not (tokens[j + 1] or "").startswith("▁"):
                    j += 1
                    piece += tokens[j] or ""
                if piece.lower().strip(".,!?:;") == needle:
                    positions.extend(range(i, j + 1))
                i = j + 1
            else:
                i += 1
        return positions

    # ---- internals -----------------------------------------------------

    def _tokenize(self, image: ImageLike, instruction: str):
        import torch
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = self.processor(prompt, image).to(self.device, dtype=self._dtype)
        ids = inputs["input_ids"]
        if not torch.all(ids[:, -1] == SPACE_TOKEN_ID):
            ids = torch.cat(
                [ids, torch.tensor([[SPACE_TOKEN_ID]], device=self.device, dtype=ids.dtype)],
                dim=1,
            )
        return ids.detach().clone(), inputs["pixel_values"].detach().clone()

    def _generate_action(self, ids, pixel_values):
        import torch
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=ids,
                pixel_values=pixel_values,
                max_new_tokens=self._action_dim,
                do_sample=False,
            )
        action_tokens = generated[0, -self._action_dim:].detach().clone()
        action = self._decode_action(action_tokens.cpu().numpy())
        return action, action_tokens

    def _decode_action(self, action_token_ids: np.ndarray) -> np.ndarray:
        m = self.model
        discretized = m.vocab_size - action_token_ids
        discretized = np.clip(discretized - 1, 0, m.bin_centers.shape[0] - 1)
        normalized = m.bin_centers[discretized]
        stats = m.get_action_stats(self.unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)

    def _resolve_query_position(self, q: TokenSelector, ids, full_seq_len: int) -> int:
        """Resolve a TokenSelector to a *multimodal-sequence* position.

        OpenVLA inserts image tokens after BOS, so text-sequence position P
        (for P >= 1) corresponds to multimodal position P + n_image. Every
        extract_*() method reads from `outputs.*` which use the multimodal
        coordinate system — this function consistently returns multimodal.
        """
        text_seq_len = int(ids.shape[1])
        n_image = full_seq_len - text_seq_len

        def text_to_mm(text_pos: int) -> int:
            return text_pos if text_pos == 0 else text_pos + n_image

        if q.position is not None:
            return int(q.position)             # caller provided multimodal directly
        if q.relative is not None:
            if q.relative == "last":
                return full_seq_len - 1
            if q.relative == "first":
                return 0
            if q.relative == "before_action":
                # Last position in the multimodal sequence — where the next
                # action token is predicted.
                return full_seq_len - 1
        if q.word is not None:
            text_positions = self.find_token_positions(
                self._extract_instruction_from_ids(ids), q.word,
            )
            if not text_positions:
                raise ValueError(
                    f"_resolve_query_position: word={q.word!r} did not "
                    "match any token position in the prompt. We refuse to "
                    "silently fall back to the last position — a "
                    "word-anchored diagnostic that gets last-position "
                    "attention silently looks like 'model attended to the "
                    "word' when actually we never found the word. Caller "
                    "must verify the word appears in the instruction."
                )
            return text_to_mm(text_positions[0])
        return full_seq_len - 1

    def _extract_instruction_from_ids(self, ids) -> str:
        """Recover the original instruction by decoding ids and unwrapping the
        prompt template.

        Strict: the OpenVLA prompt template has two well-known anchors
        ('take to ' and '?\\nOut'). If either is missing the prompt
        template has been changed, and silently returning the raw decoded
        text would give downstream tokenization-based diagnostics a
        polluted instruction that includes the template wrapping. We
        raise so the caller knows the adapter is misconfigured.
        """
        decoded = self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        if "take to " in decoded and "?\nOut" in decoded:
            return decoded.split("take to ")[1].split("?\nOut")[0]
        raise ValueError(
            "OpenVLAAdapter._extract_instruction_from_ids: decoded prompt "
            "does not contain the expected anchors 'take to ' and "
            "'?\\nOut'. This means PROMPT_TEMPLATE was changed or the "
            "tokenizer decoded special tokens differently from what we "
            "expect. Fix the template anchors before running "
            "word-anchored diagnostics on this model."
        )
