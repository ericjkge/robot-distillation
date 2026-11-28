#!/bin/bash
# smoke/split.sbatch's teacher or rollout job: one GPU (research/bin/submit takes GPUs from the header only)
#   ROLE=teacher|rollout TRAINER_JOB=<trainer id> CODE_SHA=<code>, --dependency=after:<trainer id>
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-split-%x-%j.out
exec bash /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/split.sbatch
