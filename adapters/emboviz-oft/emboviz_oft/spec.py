"""AdapterSpec for OpenVLA-OFT.

OFT's runtime venv installs:

  • torch (capped <2.10 — see openvla spec for the cu13 cap rationale).
  • moojink's transformers fork (bidirectional attention support — NOT
    on PyPI; the fork's distribution name is ``transformers`` so it
    cannot coexist with mainline transformers in one venv).
  • ``openvla-oft`` (research code, NOT on PyPI; the import package is
    still called ``prismatic`` / ``openvla``).
  • lerobot for LIBERO datasets.

The transformers fork takes its name slot — you cannot install both
mainline transformers and this fork at the same time in one venv,
which is precisely why we have to isolate OFT into its own runtime
venv. (And why no static pyproject extra could ever ship OFT alongside
mainline-transformers-using adapters.)
"""

from __future__ import annotations

from emboviz.adapters import AdapterSpec


SPEC = AdapterSpec(
    name="oft",
    server_module="emboviz_oft.server",
    runtime_pip=(
        "torch>=2.2,<2.10",
        "torchvision>=0.17",
        "accelerate>=0.30",
        "peft>=0.11",
        "timm>=0.9.10,<1.0",
        "einops>=0.8",
        "safetensors>=0.4",
        "diffusers>=0.30,<0.40",
        "lerobot>=0.3,<0.5",
        "av>=14",
        "pandas>=2.0",
        # moojink's transformers fork (claims the ``transformers``
        # distribution name).
        "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git",
        # openvla-oft research code. The pip distribution name is
        # ``openvla-oft``; the import package is ``prismatic`` /
        # ``openvla``. Without explicit naming on the LHS uv refuses
        # to resolve direct URLs in extras.
        "openvla-oft @ git+https://github.com/moojink/openvla-oft.git",
        # Core + this shim. The lifecycle layer rewrites both to
        # ``-e <local_path>`` in dev mode.
        "emboviz",
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
