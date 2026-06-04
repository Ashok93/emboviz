"""Cosmos 3 instruction-based image editing — the stress-test perturbation.

The closed-loop stress test starts each clip from a *perturbed* version of a real
frame: "rotate the cup 90 degrees so the handle faces away", "replace the cup
with a rubber duck". Cosmos 3 produces these with its ``image2image`` mode — an
input image plus a natural-language editing instruction yields an edited image
(cosmos-framework ``inference/inference.py::_get_image_edit_sample_data``; the
model applies the editing system prompt *"You are a helpful assistant who will
edit images based on the user's instructions."* internally).

Served over the vLLM-Omni OpenAI-compatible API. Image editing uses the chat
endpoint with the image as a base64 data URL and generation parameters in
``extra_body`` (vLLM-Omni ``examples/online_serving/image_to_image``). The exact
endpoint Cosmos 3 registers for ``image2image`` is the one detail that can only be
confirmed against a running server (``GET {server_url}/v1/models``); it is
isolated here behind :attr:`CosmosImageEditor.endpoint` so a difference is a
one-line change, never a redesign. This client carries no torch and holds no GPU.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np

#: The editing system prompt Cosmos applies to image2image (cosmos-framework
#: ``model/vfm/vlm/qwen3_vl/utils.py``). The server-side pipeline adds it; it is
#: recorded here for provenance and for callers that must pass it explicitly.
EDIT_SYSTEM_PROMPT = "You are a helpful assistant who will edit images based on the user's instructions."


class CosmosImageEditor:
    """Thin HTTP client for Cosmos 3 ``image2image`` editing.

    Parameters
    ----------
    server_url
        Base URL of the running vLLM-Omni Cosmos 3 server.
    num_inference_steps, guidance_scale, seed
        Generation settings forwarded in ``extra_params`` / ``extra_body``.
    request_timeout
        Per-request HTTP timeout in seconds.
    endpoint_path
        Path appended to ``server_url`` for the edit call. Default is the
        documented vLLM-Omni image-edit chat endpoint; override if a running
        Cosmos 3 server registers ``image2image`` elsewhere.
    """

    def __init__(
        self,
        server_url: str,
        *,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        seed: int = 0,
        request_timeout: float = 300.0,
        endpoint_path: str = "/v1/chat/completions",
    ):
        if not server_url:
            raise ValueError("CosmosImageEditor: server_url is required.")
        self._server_url = server_url.rstrip("/")
        self._endpoint = f"{self._server_url}{endpoint_path}"
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._seed = int(seed)
        self._request_timeout = float(request_timeout)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def edit(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Apply ``instruction`` to ``image`` and return the edited RGB frame.

        ``image`` is ``(H, W, 3)`` uint8 RGB; the returned array has the same
        dtype and channel layout (size may differ if the model re-scales). Raises
        on an empty instruction or a malformed server response — there is no
        fallback to the unedited image, which would silently run the stress test
        on the wrong scene.
        """
        instruction = (instruction or "").strip()
        if not instruction:
            raise ValueError("CosmosImageEditor.edit: a non-empty instruction is required.")
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"CosmosImageEditor.edit: image must be (H, W, 3) uint8 RGB, got "
                f"dtype={arr.dtype} shape={arr.shape}."
            )

        height, width = int(arr.shape[0]), int(arr.shape[1])
        data_url = f"data:image/png;base64,{self._encode_png(arr)}"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "extra_body": {
                "height": height,
                "width": width,
                "num_inference_steps": self._num_inference_steps,
                "guidance_scale": self._guidance_scale,
                "seed": self._seed,
            },
        }

        import requests

        response = requests.post(self._endpoint, json=payload, timeout=self._request_timeout)
        response.raise_for_status()
        return self._decode_response(response.json())

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _decode_response(body: dict) -> np.ndarray:
        """Extract the edited image (base64 data URL) from the chat response."""
        url = _dig_image_url(body)
        if url is None:
            raise RuntimeError(
                "CosmosImageEditor: server response did not contain an edited image "
                f"under choices[0].message.content[*].image_url.url. Body keys: {sorted(body)}."
            )
        b64 = url.split(",", 1)[1] if url.startswith("data:") else url
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def _dig_image_url(body: dict) -> Optional[str]:
    """Pull the first ``image_url.url`` out of an OpenAI-style chat completion."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url")
                if isinstance(url, str) and url:
                    return url
    if isinstance(content, str) and content.startswith("data:image"):
        return content
    return None


__all__ = ["EDIT_SYSTEM_PROMPT", "CosmosImageEditor"]
