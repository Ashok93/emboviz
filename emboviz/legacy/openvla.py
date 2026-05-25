"""OpenVLA-7B loading + a thin attribution-friendly inference layer.

The HF wrapper hides too much. For attribution we need to (a) reconstruct the
exact `input_ids` used at inference so we can teacher-force the predicted
action tokens, (b) expose logits for those 7 action positions as a single
scalar target, and (c) preserve gradients into `pixel_values` and the text
embedding layer. This module is the thin shim that does just that.

Single source of truth for the OpenVLA inference contract — anything else
needing model internals goes through `OpenVLAInference`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_REPO = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
SPACE_TOKEN_ID = 29871  # Llama tokenizer's space token — added before action tokens


@dataclass
class VLAPrediction:
    """One predict_action call's worth of differentiable bookkeeping."""

    action: np.ndarray              # (7,) — final unnormalized action
    action_token_ids: torch.Tensor  # (7,) — generated token IDs (long, on device)
    full_input_ids: torch.Tensor    # (1, prompt_len + 8) — prompt + space + action tokens
    prompt_len: int                 # length of prompt+image+space prefix (before action tokens)
    pixel_values: torch.Tensor      # (1, ...) — exactly what was fed to the model
    instruction_text: str           # original instruction string


class OpenVLAInference:
    """Wraps OpenVLA-7B with two modes:

      • `predict(image, instruction)`  → fast path (calls native predict_action)
      • `scalar_target_from_logits(...)`  → differentiable scalar for captum

    Loading is eager + bf16 + eager-attn (not flash-attn) so gradients flow
    cleanly during attribution. Flash-attention can be turned on for the
    `predict` fast path later if speed matters; we keep things simple here.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_REPO,
            attn_implementation="eager",   # gradient-friendly
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    # ---- public API --------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: str = DEFAULT_UNNORM_KEY,
    ) -> VLAPrediction:
        """Run the standard OpenVLA action prediction and return the bookkeeping
        we need to drive captum afterwards.
        """
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        # Mimic the model's internal pre-action space-token insertion so our
        # `full_input_ids` matches the generation-time prefix exactly.
        if not torch.all(input_ids[:, -1] == SPACE_TOKEN_ID):
            input_ids = torch.cat(
                [input_ids, torch.tensor([[SPACE_TOKEN_ID]], device=self.device, dtype=input_ids.dtype)],
                dim=1,
            )

        action_dim = self.model.get_action_dim(unnorm_key)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=action_dim,
                do_sample=False,
            )
        # Clone everything that crosses out of generate() so no
        # inference-mode tensors leak into captum's autograd graph later.
        action_token_ids = generated_ids[0, -action_dim:].detach().clone()
        action = self._decode_action(action_token_ids.cpu().numpy(), unnorm_key)

        full_input_ids = torch.cat(
            [input_ids.detach().clone(), action_token_ids.unsqueeze(0)], dim=1
        )
        prompt_len = int(input_ids.shape[1])
        pixel_values = pixel_values.detach().clone()

        return VLAPrediction(
            action=action,
            action_token_ids=action_token_ids,
            full_input_ids=full_input_ids,
            prompt_len=prompt_len,
            pixel_values=pixel_values,
            instruction_text=instruction,
        )

    def scalar_attribution_target(
        self,
        pixel_values: torch.Tensor,
        full_input_ids: torch.Tensor,
        action_token_ids: torch.Tensor,
        prompt_len: int,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass + scalar target for captum.

        Target = sum of log-probabilities of the *chosen* action tokens.
        Higher = model more confident in its action. Gradient of this w.r.t.
        pixels (or text embeddings) tells us which inputs the model relied on.

        Pass `inputs_embeds` instead of relying on `full_input_ids` when doing
        text-embedding-layer attribution.
        """
        if inputs_embeds is not None:
            outputs = self.model(
                pixel_values=pixel_values,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.model(
                input_ids=full_input_ids,
                pixel_values=pixel_values,
            )
        logits = outputs.logits  # (B, seq, vocab)
        action_dim = int(action_token_ids.shape[-1])

        # The token at position t predicts the token at position t+1. The
        # action tokens live at indices [prompt_len .. prompt_len+action_dim-1]
        # in `full_input_ids`, so the predicting positions are one earlier.
        pred_positions = slice(prompt_len - 1, prompt_len - 1 + action_dim)
        action_logits = logits[:, pred_positions, :]                 # (B, 7, V)
        log_probs = torch.log_softmax(action_logits.float(), dim=-1)  # (B, 7, V)

        # Gather log-prob of the chosen token at each of the 7 positions.
        targets = action_token_ids.view(1, -1, 1).expand(log_probs.shape[0], -1, 1)
        chosen = log_probs.gather(dim=-1, index=targets).squeeze(-1)  # (B, 7)
        return chosen.sum(dim=-1)                                     # (B,)

    # ---- helpers -----------------------------------------------------------

    def _decode_action(self, action_token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
        """Mirror of HF predict_action's token→action math."""
        m = self.model
        discretized = m.vocab_size - action_token_ids
        discretized = np.clip(discretized - 1, 0, m.bin_centers.shape[0] - 1)
        normalized = m.bin_centers[discretized]
        stats = m.get_action_stats(unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)

    @property
    def llm_embedding_layer(self) -> torch.nn.Module:
        """The text-token embedding layer — captum's LayerIntegratedGradients
        operates on this for token attribution. Lives inside the LLM backbone.
        """
        return self.model.language_model.get_input_embeddings()
