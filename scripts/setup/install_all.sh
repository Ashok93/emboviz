#!/usr/bin/env bash
# Run all venv installs in sequence. ~30-45 min total on a fresh pod
# (mostly waiting on pip + flash-attn build + first SAM 3 checkpoint).
#
# Five venvs:
#   00 bootstrap          system pkgs + uv + cache dirs
#   01 openvla            VLA adapter (Python 3.10)
#   02 oft                VLA adapter (Python 3.10, vendored transformers fork)
#   03 pi0                VLA adapter (Python 3.11)
#   04 gr00t              VLA adapter (Python 3.11)
#   05 sam3               SAM 3 sidecar (Python 3.12) — separate by design
#                         so every VLA adapter shares the same text→mask
#                         default detector over HTTP.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "########## 00 bootstrap ##########"
bash "$DIR/00_bootstrap.sh"
source /root/.bashrc.emboviz

echo "########## 01 openvla ##########"
bash "$DIR/01_install_openvla_venv.sh"

echo "########## 02 oft ##########"
bash "$DIR/02_install_oft_venv.sh"

echo "########## 03 pi0 ##########"
bash "$DIR/03_install_pi0_venv.sh"

echo "########## 04 gr00t ##########"
bash "$DIR/04_install_gr00t_venv.sh"

echo "########## 05 sam3 sidecar ##########"
bash "$DIR/05_install_sam3_venv.sh"

echo "########## ALL DONE ##########"
echo "Start the SAM 3 sidecar before any real run:"
echo "    /root/venvs/sam3/bin/emboviz-sam3 serve --preload &"
echo "Then verify the full diagnostic suite:"
echo "    bash scripts/final_integration_test.sh"
