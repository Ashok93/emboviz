"""ZeroMQ worker entry-point for the ACT adapter.

Run from the runtime venv as one of::

    emboviz-act serve --kwargs '{"checkpoint": "<repo_or_dir>", "camera_mapping": {...}}'
    python -m emboviz_act.server serve --kwargs '...'

The constructor loads the checkpoint, so a successful start means the
model is ready.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_act.model import ACTAdapter

    serve(
        lambda **kw: VLAModelHandler(ACTAdapter(**kw)),
        name="act",
    )


if __name__ == "__main__":
    main()
