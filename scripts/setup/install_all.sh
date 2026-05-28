#!/usr/bin/env bash
# Run all 4 venv installs in sequence. ~30-45 min total on a fresh pod
# (mostly waiting on pip + flash-attn build).
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

echo "########## ALL DONE ##########"
echo "Verify: bash scripts/final_integration_test.sh"
