"""emboviz-act — ACT (Action Chunking Transformer) adapter for emboviz.

ACT (Zhao et al. 2023, arXiv:2304.13705) is a DETR-style CVAE policy:
per-camera ResNet features + a proprioceptive-state token feed a
transformer encoder, and a fixed set of learned action queries
cross-attend to that encoder memory to produce an action chunk. It
consumes vision and robot state only — no language instruction.

The model runs in an isolated venv (lerobot + torch) as a ZeroMQ worker.
Importing this shim is cheap: no torch, no lerobot.
"""

__version__ = "0.3.0"
