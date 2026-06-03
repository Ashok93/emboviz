"""AdapterSpec for NVIDIA Cosmos3-Nano (world model).

Registered under the ``emboviz.world_models`` entry-point group — the
world-model analogue of ``emboviz.adapters`` (policies) and
``emboviz.readers`` (datasets). Core imports only this small spec module to
learn how to build the worker venv and launch the ZMQ worker; it never
imports the adapter's HTTP/decode code.

Unlike the VLA adapters, this worker carries **no torch and needs no GPU**.
Cosmos's action-conditioned forward dynamics is served by a separate
vLLM-Omni process (``vllm/vllm-omni:cosmos3``) that holds the BF16 model on
its own GPU; the worker here is a thin HTTP client. Its runtime venv needs
only the HTTP + video-decode stack.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="cosmos3",
    server_module="emboviz_cosmos3.server",
    # The worker is a pure HTTP client to a vLLM-Omni Cosmos 3 server. It
    # POSTs the conditioning frame + action chunk and decodes the returned
    # MP4 — so the runtime needs an HTTP client (requests), a video decoder
    # (imageio + PyAV, whose wheels bundle the ffmpeg libraries), and PIL to
    # encode the conditioning frame. No torch, no CUDA: the GPU work happens
    # in the vLLM-Omni server, not here.
    runtime_pip=(
        "requests>=2.31",
        "imageio>=2.34",
        "av>=12",
        "Pillow>=10",
        "numpy>=1.26",
        # Core wire contract + this shim. The lifecycle layer rewrites both
        # to ``-e <local_path>`` in dev mode; user-mode pulls from PyPI.
        "emboviz-wire",
        "emboviz-cosmos3",
    ),
    default_actor_kwargs={
        "server_url": "http://localhost:8000",
    },
    description=(
        "NVIDIA Cosmos3-Nano world model — action-conditioned forward "
        "dynamics via a vLLM-Omni server. Thin HTTP client; no GPU here."
    ),
    requires_python="3.12",
    # The worker holds no model and touches no CUDA device; the GPU lives in
    # the separate vLLM-Omni server.
    needs_gpu=False,
)
