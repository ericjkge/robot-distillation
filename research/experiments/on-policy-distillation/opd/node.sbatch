#!/bin/bash
# opd/run.sbatch's teacher or rollout job: two GPUs (research/bin/submit takes GPUs from the header only)
#   ROLE=teacher|rollout LAUNCH=<tag> ROLLOUT_JOBS=<n> CODE=... SFT_SHA=...; a teacher job with -c 12 --mem=120G
#   (a teacher GPU's bundle takes 3 CPUs, and Ray's small actors can land on any machine)
#SBATCH --partition=gpu_requeue  # submitted with -p gpu_quad: not preemptible, as losing one job stops the cluster
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16  # a rollout job's (2 GPUs x 2, 8 simulators, room for Ray's small actors)
#SBATCH --mem=160G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/opd-%x-%j.out
exec bash /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/opd/run.sbatch
