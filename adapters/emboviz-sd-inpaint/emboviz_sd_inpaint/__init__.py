"""Stable Diffusion text-guided inpainting adapter for emboviz.

The object-insertion backend for the closed-loop dream scene swap: given an
image, a binary mask, and a text prompt, it regenerates only the masked region
to contain the described object. Runs as an isolated-venv ZeroMQ worker, exactly
like the LaMa (removal) and SAM 3 (detection) adapters.
"""

__all__: list[str] = []
