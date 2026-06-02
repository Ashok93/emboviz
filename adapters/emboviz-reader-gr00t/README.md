# emboviz-reader-gr00t

Isolated **GR00T-format dataset reader** for [emboviz](../../). Sibling of
`emboviz-lerobot` — a reader keyed to a dataset *format*, not a model.

## What it reads

A **GR00T dataset** is a standard **LeRobot v2.1** dataset (parquet + mp4 +
`meta/{info,episodes,tasks}.jsonl`) plus one extra file,
**`meta/modality.json`** — NVIDIA Isaac-GR00T's declaration of how the
packed `observation.state` / `action` vectors split into named fields
(see [Isaac-GR00T data_preparation.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)).

The reader wraps the **canonical** `lerobot.datasets.LeRobotDataset` for
all decoding (parquet, mp4 video, task lookup) and reimplements none of
it. Its only GR00T-specific work is reading `modality.json` to locate the
gripper inside the packed state. It emits emboviz's universal
`Scene` / `Trajectory` types, so every diagnostic consumes a GR00T dataset
exactly like any other.

## Why a separate package + venv

lerobot ≥ 0.4 reads only the v3.0 on-disk format and hard-refuses v2.x
(`BackwardCompatibilityError`). GR00T datasets are v2.1, so this reader's
isolated venv pins the **last v2.1-capable lerobot, `>=0.3.3,<0.4`**
(0.4.0 already flipped `CODEBASE_VERSION` to `v3.0`). The v3.0 datasets are
read by the separate `emboviz-lerobot` reader; the two lerobot versions are
mutually incompatible, which is precisely why they live in separate venvs.

This is a **reader**, not the GR00T model adapter. It never imports the
`gr00t` package; `emboviz-gr00t` (the model) is untouched. The reader's
`SPEC.name` is `reader-gr00t` (distinct venv from the model's `gr00t`).

## Install

Ships with [emboviz](../../README.md#installation) core — `uv sync` installs
it; you do not install it separately. The isolated worker venv builds
automatically on first use.

## Use

```yaml
# configs/<your>.yaml
dataset:
  format: gr00t          # → routed to this reader
  path: /path/to/gr00t_dataset   # LeRobot v2.1 dir with meta/modality.json
  ...
```

The IPEC LIBERO datasets ship plain LeRobot v2.1 **without** `modality.json`.
Add it the way Isaac-GR00T does before pointing the reader at the dataset:

```bash
hf download --repo-type dataset \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --local-dir <dir>
cp Isaac-GR00T/examples/LIBERO/modality.json <dir>/meta/
```

A user's own GR00T-finetuned dataset already carries `modality.json`
(training requires it), so it works directly.
