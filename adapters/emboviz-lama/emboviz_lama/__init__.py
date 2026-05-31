"""emboviz-lama — LaMa (big-lama) inpainting fill for emboviz.

LaMa (Suvorov et al., *Resolution-robust Large Mask Inpainting with
Fourier Convolutions*, WACV 2022 — arXiv:2109.07161; Apache-2.0) is a
feed-forward GAN that fills a masked region with plausible background.
emboviz uses it as the **on-manifold third fill** of the memorization
diagnostic (LITERATURE.md §1): channel-mean and Gaussian-blur are both
OOD-leaning fills, so an inpainting fill is needed to span the
on-manifold/OOD axis the agreement gate is supposed to cover.

We ship it as an isolated ZeroMQ worker — the same pattern as
``emboviz-sam3`` — because its torch runtime can't share a venv with the
VLA adapters, and because it is DETERMINISTIC and feed-forward (unlike the
2025-era diffusion object-removers), which is exactly what a calibrated
diagnostic needs: a reproducible, conservative fill that does not
hallucinate new content into the hole.

Importing this package is cheap: no torch, no Pillow. The heavy machinery
is materialised inside the isolated runtime venv (``~/.emboviz/venvs/lama``)
when ``emboviz install-lama`` runs.
"""

__version__ = "0.3.0"
