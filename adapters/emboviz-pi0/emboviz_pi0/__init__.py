"""emboviz-pi0 — Physical Intelligence π0 / π0.5 adapter for emboviz.

Importing this package is cheap: no torch, no jax, no openpi.
The heavy machinery is materialised inside the isolated runtime venv
(``~/.emboviz/venvs/pi0``) when ``emboviz analyze --model pi0 ...``
spawns the ZeroMQ worker — or when the user runs ``emboviz-pi0 serve``
in that venv directly.
"""

__version__ = "0.3.0"
