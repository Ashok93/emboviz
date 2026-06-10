# Vendored Ctrl-World reference implementation

Source: https://github.com/Robert-gyj/Ctrl-World
Commit: `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`
License: MIT (Copyright (c) 2025 Tsinghua University) — see `LICENSE.txt` in this directory.
Paper: Guo, Shi, Chen, Finn, *Ctrl-World: A Controllable Generative World Model
for Robot Manipulation*, ICLR 2026, arXiv:2510.10125.

Ctrl-World is research code with no installable package (no `setup.py` /
`pyproject.toml`), so the inference modules are vendored here, following the
same pattern as `emboviz_cosmos3._cosmos_action`.

| File | Upstream path | Modifications |
|---|---|---|
| `unet_spatio_temporal_condition.py` | `models/unet_spatio_temporal_condition.py` | none (verbatim) |
| `pipeline_stable_video_diffusion.py` | `models/pipeline_stable_video_diffusion.py` | `from models.…` import rewritten to the relative `from .…` |
| `pipeline_ctrl_world.py` | `models/pipeline_ctrl_world.py` | kept `svd_tensor2vid`, `_append_dims`, `CtrlWorldDiffusionPipeline` verbatim; dropped the unrelated `LatentToVideoPipeline` and `TextStableVideoDiffusionPipeline` classes and the imports only they used, plus the unused `from einops import rearrange, repeat` and `import PIL`; `from models.…` import rewritten relative |
| `ctrl_world.py` | `models/ctrl_world.py` | kept the sincos position-embedding helpers, `Action_encoder2`, and `CrtlWorld` verbatim; dropped module-level imports unused by those definitions (`accelerate`, `decord`, `wandb`, `swanlab`, `mediapy`, `datetime`, `os`, `tqdm`, `json`, the unused `CtrlWorldDiffusionPipeline` import); `from models.…` imports rewritten relative |
| `droid_stat.json` | `dataset_meta_info/droid/stat.json` | none (verbatim). The `state_01` / `state_99` per-dimension quantile bounds of DROID `[cartesian_position(6), gripper_position(1)]`, used by `normalize_bound` in training (`dataset/dataset_droid_exp33.py`) and rollout (`scripts/rollout_interact_pi.py`) |

No algorithmic line is altered. Anything emboviz-specific lives outside this
directory, in `emboviz_ctrlworld.model`.
