"""ZeroMQ worker entry-point for SAM 3.

Run from the runtime venv as one of::

    emboviz-sam3 serve --sock /tmp/emboviz/sam3.sock
    python -m emboviz_sam3.server --sock /tmp/emboviz/sam3.sock

Default kwargs preload the SAM 3 model on construction so the first
:meth:`detect` call doesn't pay the ~30 s cold-load latency. Pass
``--kwargs '{"preload": false}'`` to defer loading until first
request — useful for tests.
"""

from __future__ import annotations


def main() -> None:
    from emboviz.adapters import serve
    from emboviz_sam3.handler import Sam3DetectorHandler

    serve(
        Sam3DetectorHandler.from_kwargs,
        name="sam3",
        default_kwargs={"preload": True},
    )


if __name__ == "__main__":
    main()
