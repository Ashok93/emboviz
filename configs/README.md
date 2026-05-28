# emboviz run configs

One file = one `emboviz analyze` run. It declares **everything**: which model,
which checkpoint, the dataset mapping, the memorization target, which episodes,
and where to write the report. No CLI flag soup.

```bash
emboviz analyze --config configs/pi0-libero.yaml
```

## Bring your own model + dataset

Copy the template closest to your setup and edit three things:

1. **`model.kwargs.checkpoint`** → the path / HF id of *your* fine-tune.
2. **`dataset.path`** → *your* dataset (HF repo id or local dir).
3. **`dataset.cameras` / `dataset.state.convention` / `dataset.gripper`** → the
   bindings the dataset format does **not** encode (which physical camera is the
   model's "primary", whether state is joint-angles vs ee-pose, gripper units).
   Everything else (dims, per-dim names, fps) is read from the data.

```bash
cp configs/pi0-libero.yaml configs/my-run.yaml
$EDITOR configs/my-run.yaml
emboviz analyze --config configs/my-run.yaml
```

## The contract is identical for every format

`dataset.format` may be `lerobot | hdf5 | rlds` — the three self-describing
"saved episode" formats. The **schema you fill in is the same** regardless —
only the reader behind each `key` changes (a LeRobot key reads from parquet;
an HDF5 key reads from an h5 dataset; an RLDS key reads from the TFDS step
features). You never see that difference. The dims/per-dim-names are read from
each format's own schema (`meta/info.json` / the first demo's array shapes /
`builder.info.features`), never hand-typed.

For `rlds`, `dataset.path` is the **TFDS builder name** (e.g. `bridge`); put an
optional TFDS `data_dir` / `split` under `dataset.extra`, and use
`instruction.key: language_instruction` for the per-step instruction field.

(Rerun `.rrd` and MCAP/rosbag2 are *recording / debugging-viz* formats, not
dataset inputs — they are intentionally not accepted here.)

## Fields the format can never encode (you always provide them)

- `dataset.state.convention` — joint-angles vs ee-pose. We refuse to guess
  (mislabeling is a silent-wrong-answer bug).
- `dataset.cameras` — maps the model's logical roles (`primary`, `wrist_left`)
  to your dataset's actual camera keys.
- `dataset.gripper` — optional. Omit it and the gripper value just stays inside
  the state vector (no gripper-specific perturbation, nothing fabricated).
