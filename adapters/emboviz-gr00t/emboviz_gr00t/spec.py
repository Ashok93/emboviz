"""AdapterSpec for NVIDIA GR00T-N1.7.

The gr00t package itself has a hard dep on ``flash-attn``, whose
``setup.py`` imports torch *before* pip has installed it — making it
impossible to install with build isolation. We work around this with
:attr:`AdapterSpec.runtime_pip_no_deps`, which gives install_venv a
second pass that runs ``uv pip install --no-deps`` for the gr00t
package itself. The adapter falls back to SDPA / eager attention at
runtime, so ``flash-attn`` is never actually invoked.
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="gr00t",
    server_module="emboviz_gr00t.server",
    runtime_pip=(
        "torch>=2.2,<2.10",
        # Qwen3-VL backbone needs transformers >= 4.57.
        "transformers>=4.57,<4.60",
        "accelerate>=0.30",
        "peft>=0.11",
        "diffusers>=0.30,<0.40",
        "einops>=0.8",
        "albumentations>=2.0",
        "av>=14",
        "decord>=0.6",
        "pandas>=2.0",
        "lerobot>=0.3,<0.5",
        "emboviz-wire",
        "emboviz-gr00t",
    ),
    # NVIDIA's gr00t package: installed with --no-deps so pip skips
    # flash-attn (whose build setup imports torch before installing it,
    # which doesn't work under pip's build isolation). The adapter
    # falls back to SDPA at runtime.
    runtime_pip_no_deps=(
        "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git",
    ),
    default_actor_kwargs={
        "model_path":       "nvidia/GR00T-N1.7-3B",
        "embodiment_tag":   "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        "device":           "cuda",
    },
    description="NVIDIA GR00T-N1.7 (3B). Qwen3-VL backbone + diffusion action expert.",
    requires_python="3.11",
    needs_gpu=True,
)
