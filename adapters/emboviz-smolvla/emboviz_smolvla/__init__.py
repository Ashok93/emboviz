"""emboviz-smolvla — SmolVLA adapter for emboviz.

SmolVLA (Shukor et al. 2025) is a compact vision-language-action model: a
SmolVLM2 backbone (SigLIP vision encoder + SmolLM2 decoder) processes
images, a language instruction, and the robot state, and a flow-matching
action expert produces an action chunk. Inference is stochastic (the
action expert samples noise and denoises), so per-frame predictions are
averaged over samples.

The model runs in an isolated venv (lerobot[smolvla] + torch) as a ZeroMQ
worker. Importing this shim is cheap: no torch, no lerobot.
"""

__version__ = "0.3.0"
