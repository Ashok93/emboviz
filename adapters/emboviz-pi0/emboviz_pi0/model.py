"""Adapter for Physical Intelligence's π0 / π0.5 / π0-FAST via openpi.

The `openpi` repository (https://github.com/Physical-Intelligence/openpi)
is PI's official open-source inference path for the π0 family. Each
checkpoint is paired with a platform-specific observation format (DROID,
ALOHA, LIBERO, UR5, custom). This adapter wraps openpi's
`create_trained_policy` and maps our typed `Scene` into the format the
chosen config expects.

**Install (its own virtualenv):**

    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    uv pip install --no-deps -e /path/to/emboviz   # add emboviz on top

Then construct with a config name (e.g. "pi0_aloha_sim", "pi0_libero",
"pi0_fast_droid", "pi05_libero", "pi05_droid").

Strict contract:
  • Each platform builder REQUIRES the cameras / state shape its
    upstream policy was trained on. Missing cameras or mis-shaped state
    raise ValueError — we never silently feed the primary camera into
    the wrist slot or zero-pad an unexpected state vector.
  • Each platform declares its own ``RequiredInputs`` so the framework's
    Scene-validation can surface the failure at the boundary, before
    we ever call the model.

Capabilities: INFERENCE only. openpi's inference path doesn't expose
hidden states or attention through a stable API; capability-gated
diagnostics auto-skip.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from emboviz_wire import ActionResult, AttentionMaps, Scene, TokenSelector
from emboviz_wire import Capability, RequiredInputs, VLAModel


# Default checkpoint URI prefix from openpi's published checkpoints
_GCS_PREFIX = "gs://openpi-assets/checkpoints/"


def _to_chw_uint8(pil_or_arr) -> np.ndarray:
    """Convert PIL/HWC array to (3, H, W) uint8."""
    arr = np.asarray(pil_or_arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:   # HWC → CHW
        arr = arr.transpose(2, 0, 1)
    return arr.astype(np.uint8)


def _require_image(scene: Scene, *names: str) -> "np.ndarray":
    """Return the image data for the FIRST name present in the scene.

    Raises KeyError listing the names tried + the cameras actually
    available — no silent substitution.
    """
    images = scene.observations.images
    for n in names:
        if n in images:
            return images[n].data
    raise KeyError(
        f"None of the required cameras {list(names)} are in the scene. "
        f"Available cameras: {sorted(images)}. The dataset adapter must "
        "load one of the listed cameras under the expected name (rename "
        "via image_keys) — we never substitute another camera silently."
    )


def _require_state_exact(scene: Scene, expected_dim: int, platform: str) -> "np.ndarray":
    """Return the state vector, raising if missing or the wrong shape.

    Pads and truncates were previously silent; now the caller is told
    EXACTLY what shape mismatch they have so they can fix the loader.
    """
    state = scene.observations.state
    if state is None:
        raise ValueError(
            f"π0 ({platform}) requires proprioceptive state with dim "
            f"{expected_dim}, but scene.observations.state is None. "
            "Either populate state in the dataset adapter or use a config "
            "that doesn't need state."
        )
    vec = np.asarray(state.values, dtype=np.float32).reshape(-1)
    if vec.size != expected_dim:
        raise ValueError(
            f"π0 ({platform}) requires state dim {expected_dim} but scene "
            f"provides {vec.size}. Fix the dataset adapter's state_key / "
            "gripper_extractor to emit the expected layout — we never "
            "silently pad or truncate."
        )
    return vec


def _aloha_observation_builder(scene: Scene) -> dict:
    """openpi ALOHA observation format — bimanual, 4 cameras, state(14)."""
    state = _require_state_exact(scene, 14, "aloha")
    return {
        "state": state,
        "images": {
            "cam_high":        _to_chw_uint8(_require_image(scene, "cam_high", "head")),
            "cam_low":         _to_chw_uint8(_require_image(scene, "cam_low", "front")),
            "cam_left_wrist":  _to_chw_uint8(_require_image(scene, "cam_left_wrist", "wrist_left")),
            "cam_right_wrist": _to_chw_uint8(_require_image(scene, "cam_right_wrist", "wrist_right")),
        },
        "prompt": _require_instruction(scene, "aloha"),
    }


def _droid_observation_builder(scene: Scene) -> dict:
    """openpi DROID observation format — single arm + wrist + state(7)+gripper."""
    obs: dict = {
        "observation/exterior_image_1_left":
            np.asarray(_require_image(scene, "primary"), dtype=np.uint8),
        "observation/wrist_image_left":
            np.asarray(_require_image(scene, "wrist_left", "wrist"), dtype=np.uint8),
        "observation/joint_position":
            _require_state_exact(scene, 7, "droid"),
        "prompt": _require_instruction(scene, "droid"),
    }
    if scene.observations.gripper is None:
        raise ValueError(
            "π0 (droid) requires gripper state but scene.observations.gripper "
            "is None. Populate gripper in the dataset adapter."
        )
    obs["observation/gripper_position"] = np.array(
        [scene.observations.gripper.value], dtype=np.float32,
    )
    return obs


def _libero_observation_builder(scene: Scene) -> dict:
    """openpi LIBERO observation format — 2 cameras (HWC uint8) + state(8).

    Per openpi.policies.libero_policy.make_libero_example, LiberoInputs
    expects ``observation/image`` and ``observation/wrist_image`` as HWC
    ``(H, W, 3)`` uint8 arrays — NOT CHW. Feeding CHW silently produces
    garbage outputs (openpi's image pipeline reinterprets the bytes
    without erroring) and the model falls back to near-mean predictions.
    """
    return {
        "observation/image":
            np.asarray(_require_image(scene, "primary"), dtype=np.uint8),
        "observation/wrist_image":
            np.asarray(_require_image(scene, "wrist", "wrist_left"), dtype=np.uint8),
        "observation/state":
            _require_state_exact(scene, 8, "libero"),
        "prompt": _require_instruction(scene, "libero"),
    }


def _require_instruction(scene: Scene, platform: str) -> str:
    instr = scene.instruction
    if not instr:
        raise ValueError(
            f"π0 ({platform}) requires a non-empty instruction but "
            "scene.instruction is None or empty. The dataset adapter "
            "must produce a task string."
        )
    return instr


# Built-in observation builders keyed by config-name fragment.
_BUILDER_REGISTRY: dict[str, Callable[[Scene], dict]] = {
    "aloha":  _aloha_observation_builder,
    "droid":  _droid_observation_builder,
    "libero": _libero_observation_builder,
}


# Per-platform RequiredInputs declarations. Matches what the builders
# actually consume so RequiredInputs.validate() catches missing fields
# at the framework boundary instead of inside the builder.
_REQUIRED_INPUTS_REGISTRY: dict[str, RequiredInputs] = {
    "aloha": RequiredInputs(
        cameras=frozenset({"cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"}),
        instruction=True,
        state=True,
    ),
    "droid": RequiredInputs(
        cameras=frozenset({"primary", "wrist_left"}),
        instruction=True,
        state=True,
        gripper=True,
    ),
    "libero": RequiredInputs(
        cameras=frozenset({"primary", "wrist"}),
        instruction=True,
        state=True,
    ),
}


def _resolve_platform(config_name: str) -> str:
    """Pick a platform key based on substring match in config_name.

    Raises ValueError if the config doesn't match a known platform — we
    do not silently default to DROID (the old behaviour quietly fed
    DROID-shaped observations to a non-DROID checkpoint).
    """
    low = config_name.lower()
    for key in _BUILDER_REGISTRY:
        if key in low:
            return key
    raise ValueError(
        f"Cannot infer π0 platform from config_name='{config_name}'. "
        f"Known platforms: {sorted(_BUILDER_REGISTRY)}. Pass "
        "observation_builder + required_inputs explicitly for a custom "
        "platform."
    )


class Pi0Adapter(VLAModel):
    """Wraps `openpi`'s trained policy as a VLAModel.

    Construction:
        Pi0Adapter(config_name="pi0_libero")
        Pi0Adapter(config_name="pi05_droid",
                   observation_builder=my_custom_builder,
                   required_inputs=my_custom_inputs)

    ``config_name`` is REQUIRED — there is no silent default. It selects
    both the openpi checkpoint AND the platform (DROID/ALOHA/LIBERO),
    which determine the exact cameras and state shape consumed. A silent
    default here is a wrong-model bug (it previously defaulted to
    ``pi0_fast_droid``, quietly loading the DROID platform on non-DROID
    data). For known platforms the builder + required-inputs are picked
    from config_name; for a custom platform pass both explicitly.
    """

    def __init__(
        self,
        config_name: Optional[str] = None,
        checkpoint_uri: Optional[str] = None,
        observation_builder: Optional[Callable[[Scene], dict]] = None,
        required_inputs: Optional[RequiredInputs] = None,
        use_pytorch: bool = False,
    ):
        """Construct π0 adapter.

        Args:
            config_name: openpi config name (e.g. "pi0_libero",
                "pi0_fast_droid", "pi05_droid").
            checkpoint_uri: explicit checkpoint URI. Defaults to the GCS
                URL ``gs://openpi-assets/checkpoints/{config_name}`` for
                the JAX backend. When ``use_pytorch=True`` and this is
                None, we append ``_pytorch`` to the default URI — that's
                the convention ``examples/convert_jax_model_to_pytorch.py``
                produces.
            observation_builder, required_inputs: custom platform pair —
                pass BOTH or NEITHER (we never partially trust the
                platform autodetect).
            use_pytorch: when True, load the PyTorch-converted checkpoint
                (``create_trained_policy`` auto-detects via
                ``model.safetensors``). The PyTorch path is REQUIRED for
                attention extraction — π0's JAX inference path is
                JIT-compiled and doesn't surface intermediate attention.
                Convert your JAX checkpoint once with
                ``examples/convert_jax_model_to_pytorch.py`` from openpi.
        """
        if not config_name:
            raise ValueError(
                "Pi0Adapter requires an explicit config_name (e.g. "
                "'pi0_libero', 'pi05_droid', 'pi0_aloha_sim') — there is NO "
                "silent default. Set it in --model-kwargs or your run "
                "config's model.kwargs. (It selects both the checkpoint and "
                "the platform's required cameras/state; defaulting it would "
                "silently run the wrong model on your data.)"
            )

        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.shared import download
            from openpi.training import config as _config
        except ImportError as e:
            raise ImportError(
                "Pi0Adapter requires the openpi package (separate venv).\n"
                "Setup:\n"
                "    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
                "    cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync\n"
                "Then install emboviz on top of openpi's venv."
            ) from e

        self.config_name = config_name
        self.use_pytorch = use_pytorch
        cfg = _config.get_config(config_name)
        if checkpoint_uri is None:
            ckpt_uri = f"{_GCS_PREFIX}{config_name}"
            if use_pytorch:
                # Convention used by examples/convert_jax_model_to_pytorch.py.
                ckpt_uri = ckpt_uri + "_pytorch"
        else:
            ckpt_uri = checkpoint_uri
        checkpoint_dir = download.maybe_download(ckpt_uri)
        self._policy = _policy_config.create_trained_policy(cfg, checkpoint_dir)
        self._cfg = cfg

        # Sanity: create_trained_policy auto-detects PyTorch via
        # model.safetensors. If the user asked for pytorch but the loaded
        # policy is JAX (or vice versa), surface that loudly.
        loaded_is_pytorch = bool(getattr(self._policy, "_is_pytorch_model", False))
        if use_pytorch and not loaded_is_pytorch:
            raise RuntimeError(
                f"Pi0Adapter: use_pytorch=True but the policy loaded from "
                f"{ckpt_uri!r} is JAX (no model.safetensors found). Did "
                "you convert the checkpoint? Run: "
                "examples/convert_jax_model_to_pytorch.py --checkpoint-dir "
                f"... --output-path {ckpt_uri} --config-name {config_name}"
            )
        if not use_pytorch and loaded_is_pytorch:
            raise RuntimeError(
                f"Pi0Adapter: use_pytorch=False but the policy loaded from "
                f"{ckpt_uri!r} is PyTorch. Either pass use_pytorch=True or "
                "point checkpoint_uri at the JAX checkpoint directory."
            )

        if observation_builder is not None and required_inputs is not None:
            self._observation_builder = observation_builder
            self._required_inputs = required_inputs
        elif observation_builder is None and required_inputs is None:
            platform = _resolve_platform(config_name)
            self._observation_builder = _BUILDER_REGISTRY[platform]
            self._required_inputs = _REQUIRED_INPUTS_REGISTRY[platform]
        else:
            raise ValueError(
                "Pi0Adapter: pass BOTH observation_builder and required_inputs "
                "together (for a custom platform), or NEITHER (auto-pick from "
                "config_name). Mixing one custom + one auto would silently lie "
                "about what the model actually consumes."
            )
        self._action_dim = 0

    # ---- capabilities ---------------------------------------------------

    _BASE_CAPS = Capability.INFERENCE

    # π0's VLM is PaliGemma (Gemma-2B language model + SigLIP vision tower).
    # ``AttentionMaps.image_weights_clean`` uses the layer-adaptive
    # last-instruction-token map: within the mid-to-late band, pick the one
    # layer whose attention is most concentrated on the image INTERIOR
    # (where objects are) rather than the BOS/prefix sink, then mean over
    # heads. Gemma-2B's grounding signal sits mid-to-late; the very early
    # and very late layers dump on the prefix sink — hence the 0.25–0.85
    # band. Only ``recommended_layer_range_fraction`` is read by the cleaner.
    ATTENTION_PROFILE = {
        "recommended_layer_range_fraction": (0.25, 0.85),
        "literature_citation":
            "Layer-adaptive last-token attention (arXiv:2602.04304; 'How "
            "Multimodal LLMs Solve Image Tasks', arXiv:2508.20279) on "
            "PaliGemma/Gemma-2B: query = last instruction token of the "
            "prefix; select the mid-stack layer most concentrated on the "
            "image interior; mean over heads. Raw attention, no gradient.",
    }

    @property
    def _CAPS(self) -> Capability:
        # Attention extraction needs the PyTorch backend (JAX nnx doesn't
        # expose intermediate attention through a stable API). We only
        # advertise ATTENTION when we know we can deliver it.
        if self.use_pytorch:
            return self._BASE_CAPS | Capability.ATTENTION
        return self._BASE_CAPS

    # ----- identification ------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.config_name

    @property
    def capabilities(self) -> Capability:
        return self._CAPS

    @property
    def required_inputs(self) -> RequiredInputs:
        return self._required_inputs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ----- inference -----------------------------------------------------

    def predict(self, scene: Scene) -> ActionResult:
        reason = self._required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Pi0Adapter.predict: {reason}")
        observation = self._observation_builder(scene)
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        # openpi returns an action chunk (chunk_len, action_dim). Expose
        # the full chunk via action_chunk so ChunkConsistencyDiagnostic
        # can test chunk[t][1] vs chunk[t+1][0] coherence — that's the
        # actual chunk-planning quality test, not just adjacent-frame
        # single-step delta.
        if actions.ndim >= 2:
            chunk = actions if actions.ndim == 2 else actions.reshape(-1, actions.shape[-1])
            action = chunk[0]
        else:
            chunk = actions[np.newaxis, :]
            action = actions
        self._action_dim = int(action.size)
        return ActionResult(
            action=action,
            action_dim=self._action_dim,
            action_chunk=chunk,
            metadata={
                "config_name": self.config_name,
                "chunk_shape": list(actions.shape),
            },
        )

    def find_token_positions(self, instruction: str, word: str) -> list[int]:
        return []

    # ---- attention extraction (PyTorch backend only) ----

    def extract_attention(
        self, scene: Scene, query: TokenSelector,
    ) -> AttentionMaps:
        """Extract per-camera attention from π0's PaliGemma backbone.

        Only available when ``use_pytorch=True``. The JAX path uses JIT-
        compiled nnx and doesn't surface intermediate attention through a
        stable API; the PyTorch port (loaded from a checkpoint converted
        by openpi's ``examples/convert_jax_model_to_pytorch.py``) computes
        attention via openpi's custom layer-by-layer loop that joins
        PaliGemma + action-expert streams.

        **Correct strategy (capture-as-side-effect, not re-run):**

        openpi's ``PaliGemmaWithExpertModel.forward`` does NOT call
        ``paligemma.language_model.forward`` — it walks layers manually
        and computes attention via
        ``transformers.models.gemma.modeling_gemma.eager_attention_forward(...)``,
        passing Q/K/V from BOTH streams concatenated along the sequence
        axis. The function returns ``(attn_output, attn_weights)``;
        openpi discards the weights (``att_output, _ = ...``).

        We monkey-patch ``modeling_gemma.eager_attention_forward`` to
        capture ``attn_weights`` as a side effect — same exact computation
        openpi runs, no re-call, no mask reconstruction, no risk of mask
        mismatch. The first ``num_hidden_layers`` captured entries are the
        PREFIX forward (image+text, bidirectional); subsequent entries are the
        per-denoise-step action-expert forwards.

        **Which signal: instruction-token → image (the PREFIX), per the
        visual-grounding literature.** Per *Your LVLM Only Needs A Few
        Attention Heads For Visual Grounding* (CVPR 2025, arXiv:2503.06287)
        the object-localization signal in a VLM is the LAST input TEXT
        token's attention over the image patches — picked out by a few
        "localization heads", NOT a head-average. This is exactly what the
        OpenVLA adapter queries, and why its map is clean. So we read the
        PREFIX forward's last-instruction-token row over the image columns
        (images lead the prefix). The action expert's suffix→prefix attention
        — although it is what literally drives the action — is diffuse and
        positional ("inverted", object-cold) and is NOT used for the
        user-facing "where does the model look" map. Head selection happens
        downstream in ``AttentionMaps.image_weights_clean``.

        Why not re-run paligemma standalone:
          • PaliGemma's bidirectional-prefix masking lives on the parent
            PaliGemma model, NOT on the inner ``language_model``. Calling
            ``language_model.forward`` directly gives plain causal Gemma
            attention with the wrong mask (we verified this — produces
            uniform softmax at the query position).
          • Re-running adds a second full forward of the 3B backbone for
            no benefit when the real attention is already being computed
            once by openpi inside ``policy.infer``.
        """
        if not self.use_pytorch:
            from emboviz_wire import NotSupported
            raise NotSupported(
                "Pi0Adapter.extract_attention requires use_pytorch=True. "
                "The JAX inference path is JIT-compiled and doesn't expose "
                "intermediate attention. Convert your checkpoint once with "
                "openpi's examples/convert_jax_model_to_pytorch.py and "
                "re-construct with use_pytorch=True."
            )

        import torch

        reason = self._required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Pi0Adapter.extract_attention: {reason}")

        pi0_model = self._policy._model
        paligemma_with_expert = pi0_model.paligemma_with_expert
        n_layers = int(paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)

        # Phase 1: patch embed_prefix (capture shape/pad mask metadata)
        # AND eager_attention_forward (capture per-layer attention).
        from openpi.models_pytorch import gemma_pytorch as _gp
        from transformers.models.gemma import modeling_gemma as _mg

        captured_attns: list = []
        capture_meta: dict = {}

        original_embed = pi0_model.embed_prefix
        original_eager = _mg.eager_attention_forward

        def patched_embed(images, img_masks, lang_tokens, lang_masks):
            result = original_embed(images, img_masks, lang_tokens, lang_masks)
            capture_meta["prefix_embs_shape"] = tuple(result[0].shape)
            capture_meta["prefix_pad_masks"]  = result[1].detach()
            capture_meta["n_images"]          = len(images)
            return result

        def patched_eager(module, query_states, key_states, value_states,
                          attention_mask=None, scaling=None, **kwargs):
            attn_output, attn_weights = original_eager(
                module, query_states, key_states, value_states,
                attention_mask=attention_mask, scaling=scaling, **kwargs,
            )
            # attn_weights: (B, H, S, S) for the joint Q/K stream.
            captured_attns.append(attn_weights.detach())
            return attn_output, attn_weights

        pi0_model.embed_prefix = patched_embed
        _mg.eager_attention_forward = patched_eager
        # The gemma_pytorch module imported the symbol by name; rebind
        # there too so openpi's call site picks up our patched function.
        _gp_orig_eager = getattr(_gp, "eager_attention_forward", None)
        if hasattr(_gp, "modeling_gemma"):
            _gp.modeling_gemma.eager_attention_forward = patched_eager

        try:
            obs = self._observation_builder(scene)
            with torch.inference_mode():
                _ = self._policy.infer(obs)
        finally:
            pi0_model.embed_prefix = original_embed
            _mg.eager_attention_forward = original_eager
            if hasattr(_gp, "modeling_gemma"):
                _gp.modeling_gemma.eager_attention_forward = original_eager
            if _gp_orig_eager is not None:
                _gp.eager_attention_forward = _gp_orig_eager

        if "prefix_embs_shape" not in capture_meta:
            raise RuntimeError(
                "Pi0Adapter.extract_attention: embed_prefix was not "
                "invoked during policy.infer — the patched call never "
                "ran. openpi's inference path must have changed."
            )
        if len(captured_attns) < n_layers:
            raise RuntimeError(
                f"Pi0Adapter.extract_attention: only {len(captured_attns)} "
                f"attention captures observed but expected at least "
                f"{n_layers} (one per prefix-forward layer). The eager "
                "attention function was not invoked the expected number "
                "of times — openpi's layer loop may have changed."
            )

        # ── Last-instruction-token → image attention (PaliGemma prefix) ──
        # The visual-grounding signal is the LAST input TEXT token's attention
        # over the image tokens (arXiv:2503.06287; this is exactly what the
        # OpenVLA adapter queries via `before_action` → full_seq_len-1, and why
        # OpenVLA's map is clean). It is NOT the action expert's suffix→prefix
        # attention, which is diffuse and positional ("inverted", object-cold).
        #
        # π0's PaliGemma prefix is processed FIRST (captured_attns[:n_layers]);
        # its sequence is [image tokens (all cameras) ; language tokens] and is
        # bidirectional. We take the LAST VALID prefix token (the final
        # instruction token, which has aggregated the whole sentence's grounding
        # under bidirectional attention) as the query row, and read its
        # attention over the image key columns (images lead the prefix). The
        # localization-head selection in AttentionMaps.image_weights_clean then
        # isolates the few grounding heads.
        n_images = int(capture_meta["n_images"])

        prefix_attns = captured_attns[:n_layers]
        prefix_len = int(prefix_attns[0].shape[-1])
        for li, attn in enumerate(prefix_attns):
            if attn.shape[-2] != prefix_len or attn.shape[-1] != prefix_len:
                raise RuntimeError(
                    f"Pi0Adapter.extract_attention: prefix capture {li} has "
                    f"shape {tuple(attn.shape)}; expected square "
                    f"({prefix_len},{prefix_len}) prefix self-attention."
                )
        pad = capture_meta.get("prefix_pad_masks")
        if pad is None:
            raise RuntimeError(
                "Pi0Adapter.extract_attention: prefix pad mask not captured; "
                "cannot locate the last instruction token."
            )
        valid_positions = np.where(np.asarray(pad[0].cpu().numpy()).astype(bool))[0]
        if valid_positions.size == 0:
            raise RuntimeError(
                "Pi0Adapter.extract_attention: prefix pad mask is all-False — "
                "no valid prefix token to query."
            )
        query_pos = int(valid_positions[-1])    # last instruction token

        # Last instruction token's attention to image, per layer & head, with
        # the CONTENT-INDEPENDENT component removed (subtract the query-averaged
        # attention; sinks are high for every query and cancel,
        # instruction-specific grounding survives). Attention sinks are
        # documented by Xiao et al. 2309.17453 (whose remedy is KV-retention);
        # the mean-over-queries subtraction is our adapter-local heuristic.
        per_layer = []
        for layer in range(n_layers):
            a = prefix_attns[layer][0]                          # (H, S, S)
            row = a[:, query_pos, :].float().cpu().numpy()      # (H, S) last-token → keys
            marg = a.float().mean(dim=1).cpu().numpy()          # (H, S) query-averaged (sink)
            per_layer.append(np.clip(row - marg, 0.0, None))
        weights = np.stack(per_layer, axis=0)                   # (L, H, prefix_len)
        full_seq = prefix_len
        action_horizon = int(pi0_model.config.action_horizon)   # informational

        # Per-image token grid side (SigLIP vision tower). Image tokens lead
        # the prefix, so the image key columns are [0, tokens_per_image*N).
        vt = paligemma_with_expert.paligemma.model.vision_tower
        vis_cfg = vt.config if hasattr(vt, "config") else \
                  paligemma_with_expert.paligemma.config.vision_config
        image_size = int(vis_cfg.image_size)
        patch_size = int(vis_cfg.patch_size)
        side_per_image = image_size // patch_size
        tokens_per_image = side_per_image * side_per_image

        # Sanity gate: the instruction-token→image attention must not be
        # uniform across keys (uniform = degenerate mask/impl, not a verdict).
        first_row = weights[0, 0, :]
        if np.allclose(first_row, first_row[0], atol=1e-9):
            raise RuntimeError(
                "Pi0Adapter.extract_attention: prefix attention is uniform "
                "across all keys — degenerate mask/impl. Refusing to return "
                "degenerate output."
            )

        # Per-camera image-token ranges. Cameras appear in the order
        # embed_prefix iterates over `images` — which mirrors openpi's
        # platform input-transform. openpi pads to a fixed image count
        # per platform (e.g. LIBERO platform always feeds 3 image slots
        # — base + left_wrist + right_wrist — and zero-fills slots the
        # task doesn't actually have). We label each slot honestly: real
        # user-facing cameras get their name; padding slots are labeled
        # as "<padding_N>" so the diagnostic can report how much
        # attention falls on garbage tokens.
        image_token_ranges: dict[str, list[tuple[int, int]]] = {}
        image_grid_sides: dict[str, int] = {}
        cursor = 0
        camera_order = self._image_order_for_platform()
        if len(camera_order) != n_images:
            raise RuntimeError(
                f"Pi0Adapter.extract_attention: platform expects "
                f"{len(camera_order)} image slots ({camera_order}) but "
                f"embed_prefix received {n_images} images. Adapter/"
                "observation_builder mismatch — fix _image_order_for_platform."
            )
        for slot_name in camera_order:
            image_token_ranges[slot_name] = [(cursor, cursor + tokens_per_image)]
            image_grid_sides[slot_name] = side_per_image
            cursor += tokens_per_image

        return AttentionMaps(
            weights=weights,
            query_position=query_pos,
            n_keys=full_seq,
            image_token_ranges=image_token_ranges,
            image_grid_sides=image_grid_sides,
            metadata={
                "attention_profile": self.ATTENTION_PROFILE,
                "config_name":   self.config_name,
                "n_images":      n_images,
                "tokens_per_image": tokens_per_image,
                "side_per_image":   side_per_image,
                "n_prefix_layers":  int(n_layers),
                "action_horizon":   int(action_horizon),
                "query_token":      "last_instruction_token (PaliGemma prefix)",
                "attention_source": "instruction-token->image (PaliGemma prefix self-attention, localization heads)",
            },
        )

    def extract_attention_trace(self, scene: Scene):
        """Per-denoise-step, per-head action-expert cross-attention to the image.

        This is the signal the pi0.5 attention visualizer shows: the action
        expert only sees the image THROUGH cross-attention, and that attention
        SHARPENS across the flow-matching denoise steps (diffuse at t=0, locked
        onto task objects at the last step). We capture the action-expert
        attention at EVERY denoise step and keep the head axis, so the
        visualizer can scrub t=0..last and toggle heads. No averaging over
        steps, no head-mean baked in, no calibration — raw attention.
        """
        if not self.use_pytorch:
            from emboviz_wire import NotSupported
            raise NotSupported("extract_attention_trace requires use_pytorch=True.")
        import torch
        from emboviz_wire import AttentionTrace

        reason = self._required_inputs.validate(scene)
        if reason is not None:
            raise ValueError(f"Pi0Adapter.extract_attention_trace: {reason}")

        pi0_model = self._policy._model
        pwe = pi0_model.paligemma_with_expert
        n_layers = int(pwe.paligemma.config.text_config.num_hidden_layers)
        from openpi.models_pytorch import gemma_pytorch as _gp
        from transformers.models.gemma import modeling_gemma as _mg

        captured: list = []
        meta: dict = {}
        orig_embed = pi0_model.embed_prefix
        orig_eager = _mg.eager_attention_forward

        def p_embed(images, img_masks, lang_tokens, lang_masks):
            meta["n_images"] = len(images)
            return orig_embed(images, img_masks, lang_tokens, lang_masks)

        def p_eager(module, q, k, v, attention_mask=None, scaling=None, **kw):
            out, w = orig_eager(module, q, k, v, attention_mask=attention_mask,
                                scaling=scaling, **kw)
            captured.append(w.detach())
            return out, w

        pi0_model.embed_prefix = p_embed
        _mg.eager_attention_forward = p_eager
        _gp_orig = getattr(_gp, "eager_attention_forward", None)
        if hasattr(_gp, "modeling_gemma"):
            _gp.modeling_gemma.eager_attention_forward = p_eager
        try:
            with torch.inference_mode():
                _ = self._policy.infer(self._observation_builder(scene))
        finally:
            pi0_model.embed_prefix = orig_embed
            _mg.eager_attention_forward = orig_eager
            if hasattr(_gp, "modeling_gemma"):
                _gp.modeling_gemma.eager_attention_forward = orig_eager
            if _gp_orig is not None:
                _gp.eager_attention_forward = _gp_orig

        if "n_images" not in meta:
            raise RuntimeError("extract_attention_trace: embed_prefix was not invoked.")
        n_images = int(meta["n_images"])

        # captured = [prefix layers] + [n_steps × expert layers]; keep the
        # denoise (action-expert) sweeps only.
        denoise = captured[n_layers:]
        if not denoise:
            raise RuntimeError("extract_attention_trace: no denoise-step attention captured.")
        n_steps = len(denoise) // n_layers
        if n_steps == 0:
            raise RuntimeError("extract_attention_trace: <1 full expert sweep captured.")
        suffix_len = int(denoise[0].shape[-2])
        n_heads = int(denoise[0].shape[1])
        action_horizon = int(pi0_model.config.action_horizon)
        action_rows = slice(suffix_len - action_horizon, suffix_len)

        # SigLIP grid; image tokens lead the prefix (cols [0, tpi*N)).
        vt = pwe.paligemma.model.vision_tower
        vis = vt.config if hasattr(vt, "config") else pwe.paligemma.config.vision_config
        side = int(vis.image_size) // int(vis.patch_size)
        tpi = side * side
        cams = self._image_order_for_platform()
        if len(cams) != n_images:
            raise RuntimeError(
                f"extract_attention_trace: {len(cams)} image slots ({cams}) vs "
                f"{n_images} images fed to embed_prefix."
            )

        # Per denoise step: mean over expert LAYERS, mean over the action-query
        # rows, KEEP heads → (n_steps, H, key_len).
        per_step = []
        for s in range(n_steps):
            layer_maps = [
                denoise[s * n_layers + l][0, :, action_rows, :].float().cpu().numpy().mean(axis=1)
                for l in range(n_layers)
            ]
            per_step.append(np.mean(layer_maps, axis=0))   # (H, key_len)
        per_step = np.stack(per_step, axis=0)              # (n_steps, H, key_len)

        per_camera, grid_sides = {}, {}
        cursor = 0
        for slot in cams:
            cols = per_step[:, :, cursor:cursor + tpi]
            per_camera[slot] = cols.reshape(n_steps, n_heads, side, side)
            grid_sides[slot] = side
            cursor += tpi

        return AttentionTrace(
            per_camera=per_camera, grid_sides=grid_sides,
            n_steps=n_steps, n_heads=n_heads,
            source="pi0 action-expert cross-attention",
            query_desc="action chunk tokens (mean) × expert layers (mean), per head",
            metadata={
                "config_name": self.config_name,
                "tokens_per_image": tpi, "side": side,
                "n_denoise_steps": n_steps, "action_horizon": action_horizon,
            },
        )

    def _image_order_for_platform(self) -> list[str]:
        """Image-slot names in the order openpi's input transform packs
        them into embed_prefix.

        π0 architectures fix a number of image slots per platform. Slots
        with no real camera in the source observation are zero-padded by
        the input transform. We label REAL camera slots with the user-
        facing name (matching ``required_inputs.cameras``); zero-pad
        slots get the literal name ``"<padding_*_rgb>"`` so attention
        falling on them is honestly attributed.

        Source: ``openpi/src/openpi/policies/libero_policy.py`` and the
        analogous droid/aloha policy files — read those to keep this
        method in sync.
        """
        platform = _resolve_platform(self.config_name)
        if platform == "libero":
            # libero_policy.py packs 3 slots:
            #   base_0_rgb       ← observation/image       (primary)
            #   left_wrist_0_rgb ← observation/wrist_image (wrist)
            #   right_wrist_0_rgb ← zeros_like(base)       (padding)
            return ["primary", "wrist", "<padding_right_wrist_rgb>"]
        if platform == "droid":
            # droid_policy.py packs 3 slots:
            #   base_0_rgb       ← observation/exterior_image_1_left (primary)
            #   left_wrist_0_rgb ← observation/wrist_image_left      (wrist_left)
            #   right_wrist_0_rgb ← zeros_like(base)                 (padding)
            return ["primary", "wrist_left", "<padding_right_wrist_rgb>"]
        if platform == "aloha":
            # ALOHA has 4 real cameras; π0 ALOHA platform uses all 4 slots.
            return ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        raise ValueError(
            f"Pi0Adapter._image_order_for_platform: no known image-slot "
            f"layout for platform {platform!r}. Add it here after reading "
            f"the corresponding openpi policy file."
        )
