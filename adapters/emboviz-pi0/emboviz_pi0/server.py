"""ZeroMQ worker entry-point for π0 / π0.5 via Physical Intelligence's openpi.

Run from the runtime venv as one of::

    emboviz-pi0 serve --sock /tmp/emboviz/pi0.sock
    python -m emboviz_pi0.server --sock /tmp/emboviz/pi0.sock

The actual model lives in :mod:`emboviz_pi0.model` — that import is
deferred until ``main()`` runs so importing this module does NOT
require torch / openpi to be installed.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_pi0.model import Pi0Adapter

    def factory(**kwargs):
        return VLAModelHandler(Pi0Adapter(**kwargs))

    serve(factory, name="pi0")


if __name__ == "__main__":
    main()
