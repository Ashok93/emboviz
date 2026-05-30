"""emboviz-gr00t — NVIDIA GR00T-N1.7 adapter for emboviz.

Importing this package is cheap: no torch, no gr00t. The heavy
machinery is materialised inside the isolated runtime venv
(``~/.emboviz/venvs/gr00t``) when ``emboviz analyze --config <file>``
(whose ``model.adapter`` is ``gr00t``) spawns the ZeroMQ worker.
"""

__version__ = "0.3.0"
