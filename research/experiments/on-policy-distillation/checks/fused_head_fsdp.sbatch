#!/bin/bash
# checks/fused_head_fsdp.py on 2 GPUs: the lm_head's gradient under verl's FSDP2 wrapping of an untied model, with
# and without CPU offload, against one process's
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:2
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/check-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
nvidia-smi -L
"$OUT/work/envs/verl/bin/torchrun" --standalone --nproc_per_node=2 \
  /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/checks/fused_head_fsdp.py \
  "$OUT/code/bf9b274/third_party/pear/3rdparty/verl/src" 2>&1 | grep -v 'Warning\|warn('
