# emboviz-sd-inpaint

Stable Diffusion **text-guided inpainting** worker for emboviz — the
object-**insertion** backend for the closed-loop dream scene swap. Given an
image, a binary mask, and a text prompt, it regenerates only the masked region to
contain the described object (e.g. paint *"a spoon"* where SAM 3 located the
marker). The counterpart to `emboviz-lama`, which *removes* objects.

Like the other perception adapters it is a **thin shim** in the main venv (zero
heavy deps) plus an isolated runtime venv that carries torch + diffusers. The
host side talks to it over ZeroMQ via `emboviz_sd_inpaint.client.SDInpaintClient`;
`emboviz.perturb.image._inpaint.SDInpaintInserter` wraps it for core.

## Install + run

```bash
emboviz install-sd-inpaint            # build the isolated runtime venv
~/.emboviz/venvs/sd-inpaint/bin/emboviz-sd-inpaint serve
```

`emboviz analyze` / the dream driver auto-install and auto-spawn the worker on
first use, so the explicit steps are only needed for manual debugging.

## Model

Default: `stabilityai/stable-diffusion-2-inpainting` (small, ~512 px) — chosen so
the whole flow is cheap to exercise. Override with the `model_id` actor kwarg or
`EMBOVIZ_SD_INPAINT_MODEL` (any diffusers inpainting checkpoint, e.g.
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` for higher quality).

Mask convention: WHITE = repaint, BLACK = preserve. The worker composites the
generated frame back onto the original **only inside the mask**, so every
non-target pixel is byte-identical to the input.
