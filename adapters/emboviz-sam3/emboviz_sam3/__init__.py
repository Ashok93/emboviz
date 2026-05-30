"""emboviz-sam3 — Meta SAM 3 text→mask detector for emboviz.

SAM 3 (Meta AI, released Nov 2025) is a single model that takes a
text concept and segments every instance in an image. We ship it as
an isolated ZeroMQ worker because:

  • SAM 3 requires Python 3.12+ and ``transformers>=4.56``. None of
    the four VLA adapter venvs (OpenVLA / OFT / π0 / GR00T) can host
    those constraints alongside their pinned adapter deps.
  • Multiple diagnostics need SAM 3 (memorization, future
    sensitivity-by-target). Loading it once per session beats
    cold-loading in every analyze invocation.

Importing this package is cheap: no torch, no transformers. The heavy
machinery is materialised inside the isolated runtime venv
(``~/.emboviz/venvs/sam3``) when ``emboviz install-sam3`` runs.
"""

__version__ = "0.3.0"
