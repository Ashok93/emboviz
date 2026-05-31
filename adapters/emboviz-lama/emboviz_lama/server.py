"""ZeroMQ worker entry-point for the LaMa inpainting fill.

Run from the runtime venv as one of::

    emboviz-lama serve --sock /tmp/emboviz/lama.sock
    python -m emboviz_lama.server --sock /tmp/emboviz/lama.sock

Default kwargs preload the model on construction (and run a one-forward
self-test) so the first :meth:`inpaint` call doesn't pay the cold-load
latency and any checkpoint-incompatibility fails at startup. Pass
``--kwargs '{"preload": false}'`` to defer loading until first request —
useful for tests.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import serve
    from emboviz_lama.handler import LamaInpaintHandler

    serve(
        LamaInpaintHandler.from_kwargs,
        name="lama",
        default_kwargs={"preload": True},
    )


if __name__ == "__main__":
    main()
