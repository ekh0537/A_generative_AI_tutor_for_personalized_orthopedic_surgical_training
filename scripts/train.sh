#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc-per-node=8 -m ag_tri_ct.cli validate-config configs/main.yaml

