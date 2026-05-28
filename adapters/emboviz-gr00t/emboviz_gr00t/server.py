"""ZeroMQ worker entry-point for NVIDIA GR00T-N1.7.

Run from the runtime venv as one of::

    emboviz-gr00t serve --sock /tmp/emboviz/gr00t.sock
    python -m emboviz_gr00t.server --sock /tmp/emboviz/gr00t.sock

The actual model lives in :mod:`emboviz_gr00t.model` — that import is
deferred until ``main()`` runs so importing this module does NOT
require torch / the gr00t package to be installed.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_gr00t.model import Gr00tAdapter

    def factory(**kwargs):
        return VLAModelHandler(Gr00tAdapter(**kwargs))

    serve(factory, name="gr00t")


if __name__ == "__main__":
    main()
