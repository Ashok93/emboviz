"""emboviz-sam3 — SAM 3 sidecar service.

Runs in its own Python 3.12 venv (separate from any VLA adapter venv,
which are pinned to incompatible torch / transformers versions). Speaks
HTTP to adapter-side ``emboviz.perturb._target_detection.SAM3Detector``
clients.

Why a sidecar at all: the official ``facebookresearch/sam3`` repo needs
Python 3.12+ and torch 2.7+; the HuggingFace ``Sam3Model`` integration
landed in ``transformers >= 4.56``. None of our four VLA adapter venvs
(OpenVLA on 4.49, OFT on a vendored fork, π0 on 4.53, GR00T on 4.57)
can host all of those constraints simultaneously without breaking the
adapter. The sidecar isolates the SAM 3 runtime from every adapter and
gives every adapter the same default text-to-mask detector.
"""

__version__ = "0.1.0"
