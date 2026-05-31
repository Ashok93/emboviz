"""AdapterSpec for the LaMa inpainting worker.

RUNTIME-SPEC adapter (cf. ``AdapterSpec.runtime_pip``): the model is a
TorchScript export of big-lama loaded with ``torch.jit.load`` — there is
no installable provider package to drive deps, so the list below is the
inference RUNTIME the worker code needs, not a dependency mirror.

We deliberately do NOT depend on the ``simple-lama-inpainting`` PyPI
package even though we reuse its (Apache-2.0) preprocessing: that package
pins ``Pillow<10``, which conflicts with ``emboviz-wire``'s ``Pillow>=10``.
The ~40 lines of tested preprocessing are vendored into ``model.py``
instead (with attribution), which keeps the runtime to four packages and
lets us additionally fix the mod-8 crop the upstream wrapper omits.

LaMa is small (~27M params, ~200 MB checkpoint) and feed-forward, so it
runs on CPU in ~1 s; a GPU is optional (``needs_gpu=False``).
"""

from __future__ import annotations

from emboviz_wire import AdapterSpec


SPEC = AdapterSpec(
    name="lama",
    server_module="emboviz_lama.server",
    runtime_pip=(
        # The default PyPI torch wheel is CUDA-enabled on a CUDA host and
        # CPU-only otherwise — both work for LaMa, so we keep an unpinned
        # floor and let the platform's default wheel resolve (per the
        # "the default PyPI wheel works for everyone" rule).
        "torch>=2.1",
        # We fetch the pinned TorchScript checkpoint from the HF Hub.
        "huggingface-hub>=0.24",
        # Pillow + numpy for the vendored preprocessing. Compatible with
        # emboviz-wire's Pillow>=10 (the reason we vendor rather than
        # depend on simple-lama-inpainting, which caps Pillow<10).
        "Pillow>=10",
        "numpy>=1.26",
        # Core wire (pyzmq + msgpack + msgpack-numpy + the adapter base
        # classes) and this shim. The lifecycle layer rewrites both to
        # ``-e <local_path>`` in dev mode.
        "emboviz-wire",
        "emboviz-lama",
    ),
    default_actor_kwargs={
        "device": "auto",
        "preload": True,
    },
    description="LaMa (big-lama) inpainting — on-manifold mask fill for the memorization diagnostic.",
    requires_python="3.11",
    needs_gpu=False,
)
