"""ZeroMQ worker entry-point for the SmolVLA adapter.

Run from the runtime venv as one of::

    emboviz-smolvla serve --kwargs '{"checkpoint": "<repo_or_dir>", "camera_mapping": {...}}'
    python -m emboviz_smolvla.server serve --kwargs '...'

The constructor loads the checkpoint, so a successful start means the
model is ready.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import VLAModelHandler, serve
    from emboviz_smolvla.model import SmolVLAAdapter

    serve(
        lambda **kw: VLAModelHandler(SmolVLAAdapter(**kw)),
        name="smolvla",
    )


if __name__ == "__main__":
    main()
