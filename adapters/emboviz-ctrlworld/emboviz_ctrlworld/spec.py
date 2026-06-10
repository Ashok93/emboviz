"""AdapterSpec for Ctrl-World (world model).

Registered under the ``emboviz.world_models`` entry-point group, alongside
``emboviz-cosmos3``. Core imports only this small spec module to learn how to
build the worker venv and launch the ZMQ worker; it never imports torch or the
vendored diffusion code.

Unlike the Cosmos worker (a thin HTTP client to a separate vLLM-Omni server),
this worker runs the model **locally**: the 1.5B SVD UNet, its VAE, and the
CLIP text encoder live in this process on the GPU. The released DROID
checkpoint, the SVD base, and the CLIP encoder are pulled from the Hugging
Face Hub on first load (none are gated).
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="ctrlworld",
    server_module="emboviz_ctrlworld.server",
    # RUNTIME-SPEC adapter (cf. AdapterSpec.runtime_pip): Ctrl-World is
    # research code with no installable package, vendored into this shim, so
    # the list below is the inference runtime its modules require — pinned to
    # the reference repo's requirements.txt (commit 99fb206):
    #
    #   • torch 2.7.x — the reference pin (torch==2.7.1); kept to the minor so
    #     patch fixes land while staying in the tested window.
    #   • diffusers 0.34.0 / transformers 4.48.1 — exact reference pins. The
    #     vendored UNet/pipeline files are copies of diffusers internals from
    #     this version; drifting either breaks the load_state_dict surface.
    #   • einops — used by the vendored model code and the latent reshapes.
    #   • huggingface_hub — checkpoint / SVD / CLIP downloads.
    #   • accelerate — diffusers' from_pretrained device handling imports it.
    runtime_pip=(
        "torch>=2.7,<2.8",
        "diffusers==0.34.0",
        "transformers==4.48.1",
        "accelerate>=1.0",
        "einops>=0.8",
        "huggingface_hub>=0.30",
        "numpy>=1.26",
        "Pillow>=10",
        # Core wire contract + this shim. The lifecycle layer rewrites both
        # to ``-e <local_path>`` in dev mode; user-mode pulls from PyPI.
        "emboviz-wire",
        "emboviz-ctrlworld",
    ),
    default_actor_kwargs={},
    description=(
        "Ctrl-World (ICLR 2026) — multi-view, pose-anchored-memory forward "
        "dynamics on DROID. Runs the 1.5B SVD-based checkpoint locally on GPU."
    ),
    requires_python="3.11",
    needs_gpu=True,
)
