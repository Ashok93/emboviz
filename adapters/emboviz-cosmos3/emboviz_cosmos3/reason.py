"""Cosmos 3 Reason — the per-clip verdict for the stress test.

After the policy flies a perturbed scene in the world model, the question is
semantic — *did it grasp the cup or close on air?* — not a pixel number. Cosmos 3
Reason is the physical-AI reasoning VLM that answers it: given frames from the
generated clip plus a question, it returns a natural-language judgement (its
cookbook ships "robot next action" and "physical plausibility" examples).

Reason is a distinct model from the generator and may be served at its own URL
(vLLM or the ``cosmos3-reasoner`` NIM), so this client takes its own
``server_url``. OpenAI-compatible chat: a text question + one or more image
frames as base64 data URLs → a text answer. Carries no torch, holds no GPU.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np


class CosmosReasoner:
    """Thin HTTP client for Cosmos 3 Reason (OpenAI-compatible chat VLM).

    Parameters
    ----------
    server_url
        Base URL of the running reasoner server (may differ from the generator).
    model
        Optional model name sent in the request; ``None`` lets the server use its
        loaded default.
    max_frames
        Cap on how many clip frames are attached to one judgement request — a
        handful (start, middle, end) is enough to read the outcome and keeps the
        prompt bounded. Frames passed beyond this are evenly subsampled.
    request_timeout
        Per-request HTTP timeout in seconds.
    endpoint_path
        Chat endpoint path appended to ``server_url``.
    """

    def __init__(
        self,
        server_url: str,
        *,
        model: Optional[str] = None,
        max_frames: int = 6,
        request_timeout: float = 300.0,
        endpoint_path: str = "/v1/chat/completions",
    ):
        if not server_url:
            raise ValueError("CosmosReasoner: server_url is required.")
        if int(max_frames) < 1:
            raise ValueError(f"CosmosReasoner: max_frames must be >= 1, got {max_frames}.")
        self._server_url = server_url.rstrip("/")
        self._endpoint = f"{self._server_url}{endpoint_path}"
        self._model = model
        self._max_frames = int(max_frames)
        self._request_timeout = float(request_timeout)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def judge(self, frames: list[np.ndarray], question: str) -> str:
        """Return Cosmos Reason's text answer to ``question`` about ``frames``.

        ``frames`` are ``(H, W, 3)`` uint8 RGB images from the generated clip
        (evenly subsampled to ``max_frames``). Raises on no frames, an empty
        question, or a response with no text — never returns a fabricated verdict.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("CosmosReasoner.judge: a non-empty question is required.")
        if not frames:
            raise ValueError("CosmosReasoner.judge: at least one frame is required.")

        selected = _subsample(frames, self._max_frames)
        content: list[dict] = [{"type": "text", "text": question}]
        for frame in selected:
            content.append({"type": "image_url", "image_url": {"url": self._data_url(frame)}})

        payload: dict = {"messages": [{"role": "user", "content": content}]}
        if self._model is not None:
            payload["model"] = self._model

        import requests

        response = requests.post(self._endpoint, json=payload, timeout=self._request_timeout)
        response.raise_for_status()
        return self._decode_text(response.json())

    @staticmethod
    def _data_url(frame: np.ndarray) -> str:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"CosmosReasoner: each frame must be (H, W, 3) uint8 RGB, got "
                f"dtype={arr.dtype} shape={arr.shape}."
            )
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _decode_text(body: dict) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"CosmosReasoner: response had no choices. Body keys: {sorted(body)}.")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined
        raise RuntimeError("CosmosReasoner: response contained no text answer.")


def _subsample(frames: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Evenly pick at most ``k`` frames, always keeping the first and last."""
    n = len(frames)
    if n <= k:
        return frames
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [frames[i] for i in idx]


__all__ = ["CosmosReasoner"]
