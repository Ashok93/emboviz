# VM setup scripts — re-create the integration-test environment

These scripts rebuild the per-model venvs we use for end-to-end integration
tests on a fresh GPU pod (RunPod RTX 3090 or similar). Each VLA pins
incompatible transformers + lerobot versions, so each lives in its own
uv venv.

The layout: 7 independent runtime venvs, all under `/root/venvs/<name>`,
each spawned by `emboviz install-<name>`. The host main venv (core +
lightweight shims, no torch/lerobot) is `/root/.venv-emboviz`. Version
pins are NOT listed here — pyproject extras are the single source of truth.

| Venv path | Role |
|---|---|
| `/root/.venv-emboviz` | host main venv — core + lightweight shims only (no torch, no lerobot) |
| `/root/venvs/openvla` | OpenVLA-7B runtime venv |
| `/root/venvs/oft` | OpenVLA-OFT runtime venv |
| `/root/venvs/pi0` | π0 / π0.5 runtime venv |
| `/root/venvs/gr00t` | GR00T-N1.7 runtime venv |
| `/root/venvs/sam3` | SAM 3 detector runtime venv |
| `/root/venvs/lerobot` | LeRobot v3.0 dataset-reader venv |
| `/root/venvs/reader-gr00t` | GR00T-format dataset-reader venv (v2.1 + modality.json) |

## One-shot install

The bootstrap does NOT clone from GitHub — it requires the dev checkout
to be present at `/root/emboviz` first (so we test local, often unpushed,
changes) and errors out if it is absent.

```bash
# 1. From your dev machine, scp the checkout to the pod:
git archive --format=tar.gz HEAD -o /tmp/emboviz.tgz
scp -O /tmp/emboviz.tgz <pod>:/root/
ssh <pod> 'mkdir -p /root/emboviz && tar xzf /root/emboviz.tgz -C /root/emboviz'

# 2. Drop your HF token in /root/emboviz/.env (one line: HF_TOKEN=hf_...)

# 3. SSH into the pod and run the installer:
bash /root/emboviz/scripts/setup/install_all.sh

# 4. source /root/.bashrc.emboviz  (or restart shell)
# 5. ready to run integration tests
```

## Manual install (per venv)

If you only need one model, run just that script:

```bash
bash scripts/setup/01_install_openvla_venv.sh        # OpenVLA-7B on Bridge
bash scripts/setup/02_install_oft_venv.sh            # OpenVLA-OFT on LIBERO
bash scripts/setup/03_install_pi0_venv.sh            # pi0 on LIBERO
bash scripts/setup/04_install_gr00t_venv.sh          # GR00T-N1.7 on LIBERO
bash scripts/setup/05_install_sam3_venv.sh           # SAM 3 detector (memorization target)
bash scripts/setup/06_install_lerobot_venv.sh        # LeRobot v3.0 dataset reader (isolated)
bash scripts/setup/07_install_reader_gr00t_venv.sh   # GR00T-format dataset reader (v2.1 + modality.json)
```

Each model script also installs the lightweight host shims (`emboviz-wire`,
`emboviz`, `emboviz-lerobot`, `emboviz-<model>`); `06` builds the isolated
LeRobot reader venv. The reader is also auto-built on first `analyze` if you
skip `06`.

## What's NOT installed automatically

- **HF_TOKEN**: put your own in `/root/emboviz/.env`. Required for π0/GR00T (gated repos: `nvidia/Cosmos-Reason2-2B`).
- **Model checkpoints**: downloaded lazily on first inference (HF cache → `$HF_HOME=/root/hf_cache`).
- **Datasets**: same. The isolated LeRobot reader venv lazy-fetches Bridge/LIBERO from HF the first time we run.
- **droid_sample**: shipped with Isaac-GR00T; the GR00T install script clones the repo which brings it.

## After install, verify

```bash
source /root/.bashrc.emboviz
uv run python scripts/dev/verify_w2_batching.py    # batched-diagnostic gather/submit
uv run python scripts/dev/verify_reader_wire.py    # dataset reader wire round-trip
uv run python scripts/dev/deadcode_audit.py        # import-graph dead-code audit
```
