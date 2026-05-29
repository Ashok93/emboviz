# emboviz-lerobot

The LeRobot dataset reader for [emboviz](../../), isolated in its own venv.

Reading a LeRobot-format dataset needs the `lerobot` package, which pins
`rerun-sdk < 0.27` (it uses rerun for its own visualisers). emboviz core
also writes `.rrd` files and needs modern `rerun` — so `lerobot` cannot
live in the host venv without a dependency collision. This package puts
the reader where it belongs: **its own isolated venv**, talking to the
host over the same ZeroMQ wire that model workers use.

- **Host venv** installs only this thin shim (`emboviz-wire` + `numpy`)
  so emboviz can discover the reader's `AdapterSpec` via the
  `emboviz.readers` entry point.
- **Reader venv** (`~/.emboviz/venvs/lerobot`, built by
  `emboviz install-lerobot`) holds `lerobot 0.3.x` (codebase v2.1 →
  reads LeRobot v2.0 and v2.1 datasets) and its decode stack.

The worker wraps the canonical `lerobot.datasets.LeRobotDataset` — we do
not reimplement any decoding — and emits the universal emboviz `Scene` /
`Trajectory` types over the wire.

## Usage

You don't run this directly. Point a run config at a LeRobot dataset and
emboviz spawns the reader automatically:

```yaml
dataset:
  format: lerobot
  path: IPEC-COMMUNITY/bridge_orig_lerobot   # HF repo id or local path
  cameras:
    primary: observation.images.image_0
  state: {key: observation.state, convention: ee_pose}
  action: {key: action}
  instruction: {from: tasks}
```

```bash
uv pip install emboviz emboviz-lerobot
emboviz install-lerobot          # builds the isolated reader venv
emboviz analyze --config your-config.yaml
```
