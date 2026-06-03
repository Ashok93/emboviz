# emboviz run configs

One file = one `emboviz analyze` run. It declares **everything**: which model,
which checkpoint, the dataset mapping, the memorization target, which episodes,
and where to write the report. No CLI flag soup.

```bash
uv run emboviz analyze --config configs/pi0.yaml
```

There is one base config per model — `openvla`, `oft`, `pi0`, `gr00t`, `act`,
`smolvla` — each runnable as-is and meant to be copied and pointed at your own
checkpoint and dataset.

## Bring your own model + dataset

Copy the template closest to your setup and edit three things:

1. **`model.kwargs.checkpoint`** → the path / HF id of *your* fine-tune.
2. **`dataset.path`** → *your* dataset (HF repo id or local dir).
3. **`dataset.cameras` / `dataset.state.convention` / `dataset.gripper`** → the
   bindings the dataset format does **not** encode (which physical camera is the
   model's "primary", whether state is joint-angles vs ee-pose, gripper units).
   Everything else (dims, per-dim names, fps) is read from the data.

```bash
cp configs/pi0.yaml configs/my-run.yaml
$EDITOR configs/my-run.yaml
uv run emboviz analyze --config configs/my-run.yaml
```

## The contract is identical for every format

`dataset.format` may be `lerobot | gr00t | hdf5 | rlds` — the self-describing
"saved episode" formats (`gr00t` = LeRobot v2.1 + `modality.json`). The **schema you fill in is the same** regardless —
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

## Memorization mask-fill ensemble (`analysis.fills`)

The memorization diagnostic masks the target and checks whether the action
still moves. To avoid measuring the model's reaction to the *masking
artifact* instead of the *absence of the object*, it masks with an
**ensemble of fills** and requires agreement across them:

```yaml
analysis:
  fills: [channel_mean, gaussian_blur]      # default (no extra worker)
  # fills: [channel_mean, gaussian_blur, lama_inpaint]   # full ensemble
```

- `channel_mean`, `gaussian_blur` — pure-numpy, both OOD-leaning. Default.
- `lama_inpaint` — the **on-manifold** fill (plausible background via LaMa).
  Adding it makes the agreement gate span the on-manifold/OOD axis the
  literature prescribes (`LITERATURE.md` §1). It runs in the isolated
  `emboviz-lama` worker, which ships with emboviz core and is built
  automatically on first use — the analyze runner auto-starts it. Without it,
  every memorization result honestly flags
  `fill_ensemble.on_manifold_fill_present = false`.

## Fields the format can never encode (you always provide them)

- `dataset.state.convention` — joint-angles vs ee-pose. We refuse to guess
  (mislabeling is a silent-wrong-answer bug).
- `dataset.cameras` — maps the model's logical roles (`primary`, `wrist_left`)
  to your dataset's actual camera keys.
- `dataset.gripper` — optional. Omit it and the gripper value just stays inside
  the state vector (no gripper-specific perturbation, nothing fabricated).
