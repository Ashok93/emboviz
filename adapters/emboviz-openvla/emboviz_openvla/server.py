"""ZeroMQ worker entry-point for OpenVLA-7B.

Run from the runtime venv as one of::

    emboviz-openvla serve --sock /tmp/emboviz/openvla.sock
    python -m emboviz_openvla.server --sock /tmp/emboviz/openvla.sock

When ``emboviz analyze --config <file>`` runs (config's ``model.adapter``
= ``openvla``), the lifecycle layer spawns this in the background (or
attaches to it if a user already started it manually) and dispatches RPC
over a Unix-domain ZMQ socket.

The actual model lives in :mod:`emboviz_openvla.model` — that import is
deferred until ``main()`` runs so importing this module does NOT
require torch / transformers to be installed.
"""

from __future__ import annotations


def main() -> None:
    # Deferred imports — keeps ``import emboviz_openvla.server`` cheap
    # in the user's main venv during entry-point discovery.
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_openvla.model import OpenVLAAdapter

    def factory(**kwargs):
        return VLAModelHandler(OpenVLAAdapter(**kwargs))

    serve(factory, name="openvla")


if __name__ == "__main__":
    main()
