"""AdapterSpec for the Stable Diffusion text-guided inpainting worker.

RUNTIME-SPEC adapter (cf. ``AdapterSpec.runtime_pip``): the model is a diffusers
inpainting pipeline loaded by repo id from the HuggingFace Hub — there is no
installable provider package to drive deps, so the list below is the inference
RUNTIME the worker code needs, not a dependency mirror.

The default checkpoint is ``stabilityai/stable-diffusion-2-inpainting`` (SD 2.0
inpainting, ~512 px, a few GB in fp16) — deliberately small so the whole scene-
swap flow can be exercised cheaply. Override the model with the ``model_id``
actor kwarg (any diffusers-compatible inpainting checkpoint, e.g.
``diffusers/stable-diffusion-xl-1.0-inpainting-0.1`` for higher quality).
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="sd-inpaint",
    server_module="emboviz_sd_inpaint.server",
    runtime_pip=(
        # Upper-bound torch to exclude the cu13-only 2.12 wheel; the CUDA
        # build is pinned to cu126 via runtime_env_vars below so the GPU is
        # used on the common cloud hosts. Mirrors emboviz-lama.
        "torch>=2.2,<2.11",
        # diffusers ships AutoPipelineForInpainting + the SD inpaint
        # pipelines; transformers provides the CLIP text encoder/tokenizer
        # the pipeline loads; accelerate + safetensors for fast/ safe load.
        "diffusers>=0.31",
        "transformers>=4.44",
        "accelerate>=0.30",
        "safetensors>=0.4",
        "huggingface-hub>=0.24",
        "Pillow>=10",
        "numpy>=1.26",
        # Core wire (pyzmq + msgpack + msgpack-numpy + the adapter base
        # classes) and this shim. The lifecycle layer rewrites both to
        # ``-e <local_path>`` in dev mode.
        "emboviz-wire",
        "emboviz-sd-inpaint",
    ),
    # Pin torch to the cu126 PyTorch index (driver >= 12.6) via the stable
    # UV_EXTRA_INDEX_URL (extra index ranks above default PyPI in uv, so
    # torch resolves from cu126; other deps from PyPI). install_venv applies
    # runtime_env_vars to the install subprocess; core stays torch-blind.
    runtime_env_vars={"UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu126"},
    default_actor_kwargs={
        "device": "auto",
        "preload": True,
    },
    description="Stable Diffusion text-guided inpainting — object-insertion backend for the dream scene swap.",
    requires_python="3.11",
    needs_gpu=True,
)
