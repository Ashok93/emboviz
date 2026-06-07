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

Default model: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Override with
the `model_id` actor kwarg or `EMBOVIZ_SD_INPAINT_MODEL` (any diffusers inpainting
checkpoint; `stable-diffusion-v1-5/stable-diffusion-inpainting` is a lighter
alternative).

Reliable object insertion needs a high guidance scale and a negative prompt that
excludes the background (configured per run via `scene_swap.edit_guidance_scale`
and `scene_swap.edit_negative_prompt`); without them the model tends to fill the
mask with surrounding context rather than the prompted object.

VRAM: SDXL at 1024 px uses ~30 GB. Set `EMBOVIZ_SD_INPAINT_RESOLUTION=512` to run
alongside a policy on a single GPU.

Mask convention: white = repaint, black = preserve. The worker composites the
generated region back onto the original inside the mask only, so non-target pixels
are unchanged.
