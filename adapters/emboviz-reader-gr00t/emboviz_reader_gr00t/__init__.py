"""emboviz-reader-gr00t — isolated GR00T-format dataset reader for emboviz.

Importing this package is cheap: no lerobot, no torch. The heavy reader
machinery (``lerobot`` 0.3.x) is materialised inside the isolated runtime
venv (``~/.emboviz/venvs/reader-gr00t``) when ``emboviz analyze --config
<file>`` (whose ``dataset.format`` is ``gr00t``) spawns the ZeroMQ reader
worker.

A GR00T dataset is a standard LeRobot **v2.1** dataset plus one extra
file, ``meta/modality.json`` (NVIDIA Isaac-GR00T's state/action/video
layout). The worker reads it with the canonical v2.1 ``LeRobotDataset``
and ships universal ``Scene`` / ``Trajectory`` objects to the host over
the wire — the same mechanism the v3.0 ``emboviz-lerobot`` reader and the
model workers use.

This is a READER (sibling of ``emboviz-lerobot``), not a model adapter:
it never imports the GR00T model package, and the GR00T model adapter
(``emboviz-gr00t``) never imports this.

The entry point ``emboviz.readers:reader-gr00t`` resolves to
:data:`emboviz_reader_gr00t.spec.SPEC`.
"""

__version__ = "0.1.0"
