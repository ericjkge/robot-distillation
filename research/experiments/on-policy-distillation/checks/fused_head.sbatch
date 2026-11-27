#!/bin/bash
# checks/fused_head.py on one GPU, before SFT 60955095 runs the fused head
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-gc-17-[245-247]
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/check-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
nvidia-smi -L
"$OUT/work/envs/verl/bin/python" -I /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/checks/fused_head.py \
  "$OUT/code/bf9b274/third_party/pear/3rdparty/verl/src"
