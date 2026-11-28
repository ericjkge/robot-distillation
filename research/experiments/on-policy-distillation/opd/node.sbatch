#!/bin/bash
# opd/run.sbatch's teacher or rollout job: two GPUs (research/bin/submit takes GPUs from the header only)
#   ROLE=teacher|rollout LAUNCH=<tag> ROLLOUT_JOBS=<n> CODE=... SFT_SHA=...; teacher -c 6 --mem=120G, rollout -c 14 --mem=160G
#SBATCH --partition=gpu_quad  # not preemptible: losing one job stops the whole cluster
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=5-00:00:00
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/opd-%x-%j.out
exec bash /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/opd/run.sbatch
