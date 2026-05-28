"""AdapterSpec for Meta SAM 3.

SAM 3's runtime venv pins:

  • Python 3.12+ (SAM 3 reference repo requirement).
  • torch >= 2.7 (the version the reference repo validates against).
  • transformers >= 4.56 (added the ``Sam3Model`` integration).

These constraints do not coexist with any VLA adapter's pins (OpenVLA
on 4.40-4.49, OFT on a vendored fork, π0 on 4.53, GR00T on 4.57), so
the SAM 3 worker has to live in its own runtime venv — which is fine
because the ZMQ wire is bytes and cross-Python-version safe.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="sam3",
    server_module="emboviz_sam3.server",
    runtime_pip=(
        "torch>=2.7,<2.10",
        # SAM 3's processor uses ``Sam3VideoProcessor`` which is
        # backed by torchvision IO ops. Without torchvision installed,
        # ``AutoProcessor.from_pretrained("facebook/sam3")`` fails at
        # import time. The image processor itself also gates on the
        # torchvision-only path now (see
        # ``IMAGE_PROCESSOR_MAPPING_NAMES["sam3"] = {torchvision: ...}``).
        "torchvision>=0.22",
        "transformers>=4.56",
        "accelerate>=1.0",
        "safetensors>=0.5",
        "huggingface-hub>=0.28",
        "tokenizers>=0.21",
        "Pillow>=10",
        # Core (carries pyzmq + msgpack + msgpack-numpy + the
        # adapter base classes) and this shim. The lifecycle layer
        # rewrites both to ``-e <local_path>`` in dev mode.
        "emboviz-wire",
        "emboviz-sam3",
    ),
    default_actor_kwargs={
        "model_id": "facebook/sam3",
        "device_map": "auto",
        "preload": True,
    },
    description="Meta SAM 3 — open-vocabulary text→mask detector for memorization & target-aware diagnostics.",
    requires_python="3.12",
    needs_gpu=True,
)
