"""ZeroMQ worker entry-point for the LeRobot dataset reader.

Run from the runtime venv as one of::

    emboviz-lerobot serve --sock /tmp/emboviz/lerobot.sock
    python -m emboviz_lerobot.server serve --sock /tmp/emboviz/lerobot.sock

When ``emboviz analyze --config <file>`` runs (config's ``dataset.format``
= ``lerobot``), the lifecycle layer spawns this in the background and
dispatches ``load_trajectory`` / ``load_episodes`` / ``all_instructions``
RPC over a Unix-domain ZMQ socket — exactly as it does for a model
worker, but the handler exposes the EpisodeSource contract.

The dataset construction kwargs (path, camera bindings, state/action
keys, gripper, instruction) arrive via ``serve --kwargs <json>`` and are
forwarded to :func:`emboviz_lerobot.source.build_lerobot_source`.

``lerobot`` is imported lazily (inside ``build_lerobot_source``), so
importing this module for entry-point discovery does NOT require lerobot.
"""

from __future__ import annotations


def main() -> None:
    from emboviz_wire import DatasetReaderHandler, serve
    from emboviz_lerobot.source import build_lerobot_source

    def factory(**kwargs):
        return DatasetReaderHandler(build_lerobot_source(**kwargs))

    serve(factory, name="lerobot")


if __name__ == "__main__":
    main()
