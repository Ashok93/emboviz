"""ZeroMQ worker entry-point for the Ctrl-World world model.

Run from the runtime venv as one of::

    emboviz-ctrlworld serve --sock /tmp/emboviz/ctrlworld.sock
    python -m emboviz_ctrlworld.server --sock /tmp/emboviz/ctrlworld.sock

First start downloads the DROID checkpoint (~8 GB), the SVD base (~8 GB), and
the CLIP text encoder from the Hugging Face Hub, then loads them onto the GPU.
"""

from __future__ import annotations


def _build_handler(**kwargs):
    from emboviz_wire import WorldModelHandler

    from emboviz_ctrlworld.model import CtrlWorldModel

    return WorldModelHandler(CtrlWorldModel(**kwargs))


def main() -> None:
    from emboviz_wire import serve

    serve(_build_handler, name="ctrlworld")


if __name__ == "__main__":
    main()
