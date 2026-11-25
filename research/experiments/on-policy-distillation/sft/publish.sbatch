#!/bin/bash
# After SFT (afterok), upload its last checkpoint to the Hub as a private model in the robot-distillation collection
# (sft/publish.py).
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/publish-sft-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation
CKPT=$(find "$OUT/runs/sft-bf9b274/Qwen3.5-9B/ckpt" -maxdepth 2 -name huggingface -path '*global_step_*' | sort -V | tail -1)
EPISODES=$(find "$OUT/runs/collect-bf9b274/sft-set" -name '*.pt' | wc -l)
"$OUT/work/envs/verl/bin/python" "$HERE/sft/publish.py" "$CKPT" eddykang06/qwen3.5-9b-pear-libero-sft \
  eddykang06/robot-distillation-6b072adc11798b1c0c17c36c "$EPISODES"
