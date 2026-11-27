#!/bin/bash
# checks/crossnode.py: ROLE=head or join, one GPU each, the two jobs on different nodes
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/check-%x-%j.out
set -uo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then unset ROCR_VISIBLE_DEVICES; fi  # as scripts/common.sh does
hostname
RAY_TMPDIR=/var/tmp/$USER/ray-crossnode timeout 3000 "$OUT/work/envs/verl/bin/python" \
  /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/checks/crossnode.py \
  "$ROLE" "$OUT/checks/crossnode/address-$TAG" 2>&1 | grep -v Warning
"$OUT/work/envs/verl/bin/ray" stop --force > /dev/null 2>&1
