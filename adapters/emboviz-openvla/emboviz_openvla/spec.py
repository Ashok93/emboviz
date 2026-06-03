"""AdapterSpec for OpenVLA-7B.

Discovered by emboviz core through the ``emboviz.adapters`` entry-
point group declared in this package's ``pyproject.toml``. Carries:

  • The Python ``-m`` target that launches the ZMQ worker.
  • The pip requirement specs that must exist in the runtime venv
    (torch + transformers 4.40-4.49 + lerobot 0.3.x + prismatic).
  • Default kwargs forwarded to the underlying ``OpenVLAAdapter``
    constructor.

This module must stay IMPORT-LIGHT — emboviz core imports it from
the user's main venv to read SPEC. No torch, no transformers.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="openvla",
    server_module="emboviz_openvla.server",
    # RUNTIME-SPEC adapter (cf. AdapterSpec.runtime_pip): OpenVLA-7B ships
    # as HF-hub remote code, loaded via ``AutoModelForVision2Seq.
    # from_pretrained("openvla/openvla-7b", trust_remote_code=True)``.
    # There is NO installable ``openvla`` package whose pyproject could
    # drive the dependency closure, so the list below is the inference
    # RUNTIME the hub modeling code requires — not a mirror of a package's
    # transitive deps. The runtime venv needs:
    #
    #   • torch — cap below 2.10 to keep transformers/timm in OpenVLA's
    #     tested window; the cu13-only 2.12 wheel is excluded by this cap.
    #     The CUDA *build* (which must match the host driver) is pinned to
    #     cu126 via ``runtime_env_vars`` below — NOT by version-juggling —
    #     so the wheel runs on any driver >= 12.6 (every cloud A40/A6000/
    #     A100 in 2026). OpenVLA's modeling code is torch-version-agnostic
    #     (built on 2.2); the upper bound is an API window, not a CUDA one.
    #   • transformers — OpenVLA's modeling code targets the 4.40–4.49
    #     window; 4.50 dropped APIs it still uses.
    #   • timm 0.9.x — OpenVLA's prismatic-vlm hard-checks this range.
    #   • Core + this shim. The lifecycle layer rewrites both to
    #     ``-e <local_path>`` if installed editable in the caller's
    #     main venv (dev mode); user-mode pulls both from PyPI.
    #
    # NO lerobot / torchcodec / av / pandas: this is an inference-only
    # worker. It does NOT read datasets — Scenes arrive pre-decoded over
    # the ZMQ wire — and OpenVLA's hub modeling + processing code import
    # none of them (only torch/transformers/timm/tokenizers). They were
    # dead weight here, same as the OFT spec already documents.
    runtime_pip=(
        "torch>=2.2,<2.10",
        "torchvision>=0.17",
        "transformers>=4.40,<4.50",
        "accelerate>=0.30",
        "peft>=0.11",
        "timm>=0.9.10,<1.0",
        "tokenizers>=0.19,<0.22",
        "sentencepiece>=0.2",
        "einops>=0.8",
        "safetensors>=0.4",
        "emboviz-wire",
        "emboviz-openvla",
    ),
    # Pin torch to the cu126 PyTorch index (driver >= 12.6 — every cloud GPU
    # in 2026) via the stable UV_EXTRA_INDEX_URL: uv ranks the extra index
    # above default PyPI, so torch resolves from cu126, the rest from PyPI.
    # NOT the preview --torch-backend flag. install_venv applies it at install
    # time; core stays torch-blind.
    runtime_env_vars={"UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu126"},
    default_actor_kwargs={
        "hf_repo":             "openvla/openvla-7b",
        "unnorm_key":          "bridge_orig",
        "device":              "cuda",
        "attn_implementation": "eager",
    },
    description="OpenVLA-7B (Stanford). LLaMA-2 7B + SigLIP+DINOv2 vision tower.",
    requires_python="3.11",
    needs_gpu=True,
)
