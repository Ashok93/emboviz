"""emboviz-oft — OpenVLA-OFT adapter for emboviz.

Importing this package is cheap: no torch, no transformers fork, no
openvla-oft research code. The heavy machinery is materialised
inside the isolated runtime venv (``~/.emboviz/venvs/oft``) when
``emboviz analyze --config <file>`` (whose ``model.adapter`` is ``oft``)
spawns the ZeroMQ worker.
"""

__version__ = "0.3.0"
