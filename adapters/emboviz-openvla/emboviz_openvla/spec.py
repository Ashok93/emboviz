"""AdapterSpec for OpenVLA-7B.

Discovered by emboviz core through the ``emboviz.adapters`` entry-
point group declared in this package's ``pyproject.toml``. Carries:

  • The actor class to instantiate (``emboviz_openvla.actor:OpenVLAActor``).
  • The pip requirement specs that must exist in the runtime venv
    (torch + transformers 4.49 + lerobot 0.3 + prismatic checkpoint code).
  • Default kwargs forwarded to the actor's ``__init__``.

This module must stay IMPORT-LIGHT — emboviz core imports it from
the user's main venv to read SPEC. No torch, no transformers.
"""

from __future__ import annotations

from emboviz.adapters import AdapterSpec


SPEC = AdapterSpec(
    name="openvla",
    actor_import_path="emboviz_openvla.actor:OpenVLAActor",
    # The runtime venv needs:
    #
    #   • torch — cap below 2.10 to avoid the cu13-only 2.12 wheel that
    #     breaks on every cloud A40/A6000/A100 driver in 2026.
    #   • transformers — OpenVLA's modeling code targets the 4.40–4.49
    #     window; 4.50 dropped APIs it still uses.
    #   • timm 0.9.x — OpenVLA's prismatic-vlm hard-checks this range.
    #   • lerobot 0.3.x — Bridge / LIBERO datasets are exposed through
    #     LeRobotDataset; pinned below 0.5 because 0.5 reshuffled the
    #     dataset module hierarchy.
    #   • The adapter SHIM itself so the actor module is importable in
    #     the runtime venv.
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
        "lerobot>=0.3,<0.5",
        "torchcodec>=0.5",
        "av>=14",
        "pandas>=2.0",
        # Core and the shim package itself. The lifecycle layer
        # rewrites both to ``-e <local_path>`` if they're installed
        # editable in the caller's main venv (dev mode); user-mode
        # installs pull both from PyPI.
        "emboviz",
        "emboviz-openvla",
    ),
    default_actor_kwargs={
        "hf_repo":             "openvla/openvla-7b",
        "unnorm_key":          "bridge_orig",
        "device":              "cuda",
        "attn_implementation": "eager",
    },
    description="OpenVLA-7B (Stanford). LLaMA-2 7B + SigLIP+DINOv2 vision tower.",
    requires_python="3.10",
    needs_gpu=True,
)
