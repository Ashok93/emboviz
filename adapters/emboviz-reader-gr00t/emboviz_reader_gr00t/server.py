"""ZeroMQ worker entry-point for the GR00T-format dataset reader.

Run from the runtime venv as one of::

    emboviz-reader-gr00t serve --sock /tmp/emboviz/reader-gr00t.sock
    python -m emboviz_reader_gr00t.server serve --sock /tmp/emboviz/reader-gr00t.sock

When ``emboviz analyze --config <file>`` runs (config's ``dataset.format``
= ``gr00t``), the lifecycle layer spawns this in the background and
dispatches ``load_trajectory`` / ``load_episodes`` / ``all_instructions``
RPC over a Unix-domain ZMQ socket — exactly as for the lerobot reader, but
this worker reads LeRobot v2.1 + ``meta/modality.json`` datasets.

The dataset construction kwargs (path, camera bindings, state/action keys,
gripper, instruction) arrive via ``serve --kwargs <json>`` and are
forwarded to :func:`emboviz_reader_gr00t.source.build_gr00t_source`.

``lerobot`` is imported lazily (inside the source's methods), so importing
this module for entry-point discovery does NOT require lerobot.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import DatasetReaderHandler, serve
    from emboviz_reader_gr00t.source import build_gr00t_source

    def factory(**kwargs):
        return DatasetReaderHandler(build_gr00t_source(**kwargs))

    serve(factory, name="reader-gr00t")


if __name__ == "__main__":
    main()
