#!/bin/bash
# SFT of one student (STUDENT=Qwen3.5-9B | Qwen3.5-4B) on the teacher's episodes: up to 25 per task, over every
# cluster's episodes on the Hub dataset when the job starts (sft/build_set.sh, seed 0; the first job to start draws the
# set), with configs/step/sft.yaml's settings (lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs), each episode split
# across the 4 GPUs, the update on the CPU.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5
#SBATCH --requeue
#SBATCH --gres=gpu:4,vram:48G  # L40S, A6000 or A100
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=20
# host memory: the 4B pilot peaked at 147 GB on 2 GPUs; the 9B scales to about 330 GB. Submit the 4B with --mem=224G
#SBATCH --mem=450G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/sft-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
STUDENT=${STUDENT:?Qwen3.5-9B | Qwen3.5-4B}
CODE_SHA=${CODE_SHA:?the commit of the code to run}
COLLECT=${COLLECT:?the collection runs directory, runs/collect-<sha>}
CODE=$OUT/code/$CODE_SHA
export WORK_DIR=$OUT/work
exec 9> "$OUT/code/.lock"
flock 9
if [ ! -d "$CODE" ]; then
  git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
  git -C "$CODE" checkout -q "$CODE_SHA"
  git -C "$CODE" submodule update -q --init
  bash "$CODE/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh"
fi
SET=$COLLECT/sft-set
if [ ! -d "$SET" ]; then
  bash /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/sft/build_set.sh
fi
flock -u 9
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname); $(ls "$SET"/*.pt | wc -l) episodes in $SET"

start=$SECONDS
MODEL=$WORK_DIR/models/$STUDENT EPISODES_DIR=$SET RUN=$OUT/runs/sft-${CODE_SHA:0:7}/$STUDENT NGPUS=4 \
  bash scripts/sft.sh trainer.save_freq=10 trainer.max_ckpt_to_keep=2  # a requeued job resumes from the latest
echo "sft $STUDENT: $((SECONDS - start)) s"
