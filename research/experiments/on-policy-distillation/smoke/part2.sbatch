#!/bin/bash
# GPU smoke test, part 2 of 3, after part 1, on two GPUs: SFT 2 updates on part 1's episodes (FSDP2 shards a 4B's fp32
# parameters and gradients, and the optimizer state comes onto the GPUs only for the update), saved also as a Hugging
# Face model; evaluate it on the 2 episodes. check.py verifies each step's outputs.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5: a short, restartable job
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-part2-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

stage sft env MODEL=$BASE EPISODES_DIR=$RUNS/collect/episodes RUN=$RUNS/sft NGPUS=2 \
  bash scripts/sft.sh trainer.total_training_steps=2 data.train_batch_size=2 data.micro_batch_size_per_gpu=1 \
  engine.optimizer_offload=true trainer.logger='[console]'
stage eval_sft env ROLLOUT_GPUS=1 MODEL_FILE=$STUDENT_FILE MODEL=$SFT_MODEL RUN=$RUNS/eval-sft \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE" "$UNITS"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID" sft eval-sft
