#!/bin/bash
# Downloads the teacher (Qwen3.8-27B-PEAR-LIBERO_only) and the 9B student at their pinned revisions into the work
# directory, with PEAR's download script (the 4B is already there).
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/download-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/${CODE_SHA:?the commit of the code to run}
export WORK_DIR=$OUT/work
cd "$CODE"
for model_file in configs/models/teacher.sh configs/models/qwen3_5_9b.sh; do
  MODEL_FILE=$PWD/$model_file bash third_party/pear/scripts/download_model.sh
done
du -sh "$WORK_DIR"/models/*
