"""emboviz-lerobot — isolated LeRobot dataset reader for emboviz.

Importing this package is cheap: no lerobot, no torch. The heavy reader
machinery (``lerobot``) is materialised inside the isolated runtime venv
(``~/.emboviz/venvs/lerobot``) when ``emboviz analyze --config <file>``
(whose ``dataset.format`` is ``lerobot``) spawns the ZeroMQ reader
worker. The worker reads the dataset with the canonical
``LeRobotDataset`` and ships universal ``Scene`` / ``Trajectory`` objects
to the host over the wire — the same mechanism a model worker uses.

The entry point ``emboviz.readers:lerobot`` resolves to
:data:`emboviz_lerobot.spec.SPEC`.
"""

__version__ = "0.3.0"
