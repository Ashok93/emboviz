"""ZeroMQ worker entry-point for OpenVLA-OFT.

Run from the runtime venv as one of::

    emboviz-oft serve --sock /tmp/emboviz/oft.sock
    python -m emboviz_oft.server --sock /tmp/emboviz/oft.sock

The actual model lives in :mod:`emboviz_oft.model` — that import is
deferred until ``main()`` runs so importing this module does NOT
require torch / the moojink transformers fork to be installed.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_oft.model import OpenVLAOFTAdapter

    def factory(**kwargs):
        return VLAModelHandler(OpenVLAOFTAdapter(**kwargs))

    serve(factory, name="oft")


if __name__ == "__main__":
    main()
