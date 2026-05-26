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

from emboviz.core.types import ActionResult, AttentionMaps, Scene, TokenSelector
from emboviz.models.protocol import Capability, RequiredInputs, VLAModel
from emboviz.models.registry import register_model


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


@register_model("pi0")
@register_model("pi05")
class Pi0Adapter(VLAModel):
    """Wraps `openpi`'s trained policy as a VLAModel.

    Construction:
        Pi0Adapter()                                        # pi0_fast_droid default
        Pi0Adapter(config_name="pi0_libero")
        Pi0Adapter(config_name="pi05_droid",
                   observation_builder=my_custom_builder,
                   required_inputs=my_custom_inputs)

    For known platforms (DROID/ALOHA/LIBERO) the builder + required-inputs
    are picked automatically. For a custom platform pass both explicitly —
    we do not fall back to DROID silently.
    """

    def __init__(
        self,
        config_name: str = "pi0_fast_droid",
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
        mismatch. The first ``num_hidden_layers`` captured entries are
        the PREFIX-only forward (the first ``paligemma_with_expert.forward``
        call inside ``sample_actions``, with ``suffix_embs=None``);
        subsequent entries are the per-denoise-step forwards which
        include the action expert stream — we ignore those.

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
            from emboviz.models.protocol import NotSupported
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

        # First n_layers captures = the prefix-only forward. The
        # subsequent denoise_step forwards include the action expert
        # stream and have a longer key sequence (we'd need to slice
        # them — defer for now; prefix attention is what user-facing
        # diagnostics want).
        prefix_attns = captured_attns[:n_layers]
        # Verify all prefix-forward attentions have the same key length
        # (== prefix sequence length).
        prefix_seq_len = int(capture_meta["prefix_embs_shape"][1])
        for li, attn in enumerate(prefix_attns):
            if attn.shape[-1] != prefix_seq_len:
                raise RuntimeError(
                    f"Pi0Adapter.extract_attention: prefix layer {li} "
                    f"attention has key-length {attn.shape[-1]} but "
                    f"prefix_embs has seq-length {prefix_seq_len}. The "
                    "captured tensor is from a non-prefix forward."
                )

        n_images = int(capture_meta["n_images"])
        prefix_pad_masks = capture_meta["prefix_pad_masks"]

        # Resolve query position.
        full_seq = prefix_seq_len
        # Image tokens come first in embed_prefix output. Per-image
        # token count comes from the SigLIP vision tower config.
        vt = paligemma_with_expert.paligemma.model.vision_tower
        vis_cfg = vt.config if hasattr(vt, "config") else \
                  paligemma_with_expert.paligemma.config.vision_config
        image_size = int(vis_cfg.image_size)
        patch_size = int(vis_cfg.patch_size)
        side_per_image = image_size // patch_size
        tokens_per_image = side_per_image * side_per_image

        # openpi pads the language sequence to a fixed length. The
        # tail of the prefix is PADDING (prefix_pad_masks=False), not
        # a real token, so its hidden state / attention is uninformative
        # — reading from that position gives degenerate uniform softmax.
        # We resolve "last" / "before_action" to the LAST VALID position
        # (the last True index in prefix_pad_masks).
        valid_mask_np = prefix_pad_masks[0].cpu().numpy().astype(bool)
        valid_positions = np.where(valid_mask_np)[0]
        if valid_positions.size == 0:
            raise RuntimeError(
                "Pi0Adapter.extract_attention: prefix_pad_masks has no "
                "valid positions — entire prefix is padding. Adapter / "
                "observation_builder bug."
            )
        last_valid_pos = int(valid_positions[-1])

        if query.position is not None:
            query_pos = int(query.position)
        elif query.relative == "last" or query.relative == "before_action":
            # Last VALID prefix token — the one whose hidden state
            # actually conditions the action head.
            query_pos = last_valid_pos
        elif query.relative == "first":
            query_pos = 0
        else:
            query_pos = last_valid_pos

        per_layer = [
            attn[0, :, query_pos, :].float().cpu().numpy()
            for attn in prefix_attns
        ]
        weights = np.stack(per_layer, axis=0)   # (L, H, n_keys)

        # Sanity gate: confirm attention at the chosen query position is
        # not uniform across keys (which would indicate a degenerate
        # mask / impl rather than a real verdict).
        first_row = weights[0, 0, :]
        if np.allclose(first_row, first_row[0], atol=1e-9):
            raise RuntimeError(
                f"Pi0Adapter.extract_attention: attention at query "
                f"position {query_pos} (last_valid={last_valid_pos}) is "
                f"uniform across all keys (constant "
                f"{float(first_row[0]):.6e}). Mask / attn-impl / "
                "position is wrong — refusing to return degenerate output."
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
                "config_name":   self.config_name,
                "n_images":      n_images,
                "tokens_per_image": tokens_per_image,
                "side_per_image":   side_per_image,
                "n_layers":      int(len(prefix_attns)),
                "query_pos":     int(query_pos),
                "last_valid_pos": int(last_valid_pos),
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
