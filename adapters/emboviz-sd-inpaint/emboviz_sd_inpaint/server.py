"""ZeroMQ worker entry-point for the SD text-guided inpainting backend.

Run from the runtime venv as one of::

    emboviz-sd-inpaint serve --sock /tmp/emboviz/sd-inpaint.sock
    python -m emboviz_sd_inpaint.server --sock /tmp/emboviz/sd-inpaint.sock

Default kwargs preload the pipeline on construction (and run a one-forward
self-test) so the first :meth:`fill` call doesn't pay the cold-load latency and
any checkpoint/API mismatch fails at startup. Pass ``--kwargs '{"preload": false}'``
to defer loading until first request — useful for tests.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import serve
    from emboviz_sd_inpaint.handler import SDInpaintHandler

    serve(
        SDInpaintHandler.from_kwargs,
        name="sd-inpaint",
        default_kwargs={"preload": True},
    )


if __name__ == "__main__":
    main()
