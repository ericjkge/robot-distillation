#!/bin/bash
# The perturbation simulators and the general benchmarks, once, on a CPU machine: PEAR's install targets libero_pro and
# libero_plus (with Mesa and ImageMagick) and evalscope (its environment and the benchmarks' datasets), under the
# shared work directory beside the environments that are there (verl, libero).
#   DESC=... CODE_SHA=<sha> research/bin/submit on-policy-distillation \
#     research/experiments/on-policy-distillation/eval/setup.sbatch
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/eval-setup-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/${CODE_SHA:?the commit of the code to run}
[ -d "$CODE" ] || { git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"; git -C "$CODE" checkout -q "$CODE_SHA"; git -C "$CODE" submodule update -q --init; }
export WORK_DIR=$OUT/work
for target in libero_pro libero_plus evalscope; do
  bash "$CODE/third_party/pear/scripts/install.sh" "$target"
done
ls "$WORK_DIR/envs"
