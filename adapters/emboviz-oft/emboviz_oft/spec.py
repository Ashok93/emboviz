"""AdapterSpec for OpenVLA-OFT.

OFT's runtime venv installs ``openvla-oft`` (research code, NOT on
PyPI; the import package is ``prismatic`` / ``openvla`` / ``experiments``)
WITH its own pinned dependency closure — its moojink ``transformers``
fork, torch/torchvision, ``draccus==0.8``, peft, timm, einops,
diffusers, json-numpy, tensorflow, etc. — plus emboviz core + this shim.

Two reasons OFT must live in its own isolated venv:
  • moojink's transformers fork claims the ``transformers`` distribution
    name, so it cannot coexist with mainline transformers in one venv.
  • openvla-oft pins ``draccus==0.8``, which conflicts with the
    ``draccus==0.10`` that lerobot pulls. The isolated venv sidesteps
    that: this is an inference-only worker, it does NOT load datasets
    (Scenes arrive pre-decoded over ZMQ), so lerobot is simply not
    installed here and there is nothing to conflict.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="oft",
    server_module="emboviz_oft.server",
    # This is an ISOLATED, inference-only worker venv. It does NOT load
    # datasets — that happens in the user's main venv and Scenes arrive
    # pre-decoded over the ZMQ wire. So lerobot (and its av/pandas/
    # torchcodec companions) have no business here. Their presence was
    # the ONLY thing dragging ``draccus==0.10`` into the venv, which
    # collided with openvla-oft's pinned ``draccus==0.8`` and forced an
    # ugly ``--no-deps`` install of openvla-oft — which then silently
    # dropped openvla-oft's real deps (json-numpy, tensorflow, peft) and
    # broke the worker at import.
    #
    # The clean fix is to let ``openvla-oft`` install WITH its own pinned
    # dependency closure — exactly what an isolated venv is for — and put
    # only emboviz core + this shim on top. emboviz core is dependency-
    # light (numpy/pyzmq/msgpack/rerun/...; no torch/transformers/lerobot)
    # so it cannot fight openvla-oft's stack.
    runtime_pip=(
        # openvla-oft drives the entire ML stack via its own metadata:
        # the moojink transformers fork, torch/torchvision, draccus 0.8,
        # peft, timm, einops, diffusers, json-numpy, tensorflow, ... .
        "openvla-oft @ git+https://github.com/moojink/openvla-oft.git",
        # Core + this shim. The lifecycle layer rewrites both to
        # ``-e <local_path>`` in dev mode.
        "emboviz-wire",
        "emboviz-oft",
    ),
    default_actor_kwargs={
        "checkpoint":         "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "unnorm_key":         "libero_spatial_no_noops",
        "num_images":         2,
        "use_proprio":        True,
        "use_l1_regression":  True,
        "use_film":           False,
        "center_crop":        True,
        "wrist_camera":       "wrist",
    },
    description="OpenVLA-OFT (Stanford). LLaMA-2 7B with parallel decoding + L1 action head; LIBERO fine-tunes.",
    requires_python="3.11",
    needs_gpu=True,
)
