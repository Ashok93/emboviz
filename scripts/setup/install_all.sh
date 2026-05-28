#!/usr/bin/env bash
# Run all venv installs in sequence. ~30-45 min total on a fresh pod
# (mostly pip + the first SAM 3 / OpenVLA / π0 checkpoint download).
#
# Layout (after running this script):
#   /root/.venv-emboviz                     main venv — core + adapter shims (Python 3.11, no torch)
#   /root/venvs/openvla                     OpenVLA-7B runtime venv
#   /root/venvs/oft                         OpenVLA-OFT runtime venv
#   /root/venvs/pi0                         π0 / π0.5 runtime venv
#   /root/venvs/gr00t                       GR00T-N1.7 runtime venv
#   /root/venvs/sam3                        SAM 3 detector runtime venv (Python 3.12)
#
# All five runtime venvs are independent — each pins its own Python +
# torch + transformers + adapter deps. ZeroMQ over Unix sockets is the
# only thing that talks between them; the wire is bytes / msgpack so
# Python versions don't have to match.
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

echo "########## 05 sam3 ##########"
bash "$DIR/05_install_sam3_venv.sh"

echo "########## ALL DONE ##########"
cat <<'EOM'

Start workers (each in its own background shell — they stay running
between analyze calls):

    /root/venvs/sam3/bin/emboviz-sam3 serve &
    /root/venvs/openvla/bin/emboviz-openvla serve &
    /root/venvs/oft/bin/emboviz-oft serve &
    /root/venvs/pi0/bin/emboviz-pi0 serve &
    /root/venvs/gr00t/bin/emboviz-gr00t serve &

Then run an analysis:

    /root/.venv-emboviz/bin/emboviz analyze \
        --model openvla --dataset bridge --episodes 537 \
        --target "the cloth" --output /root/outputs/openvla
EOM
