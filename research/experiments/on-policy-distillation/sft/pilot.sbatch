#!/bin/bash
# SFT pilot: the 4B on part 1's two smoke episodes with each episode split across 2 GPUs (Ulysses SP 2) and the
# fine-tuning optimizer of configs/step/sft.yaml, 2 updates. Checks that sequence parallelism runs for this model in
# verl's SFT trainer before the real SFT, whose episodes run to 164k tokens.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/sft-pilot-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

stage sft_sp env MODEL=$BASE EPISODES_DIR=$OUT/runs/smoke-2055b84/collect/episodes RUN=$RUNS/sft-sp NGPUS=2 \
  bash scripts/sft.sh trainer.total_training_steps=2 data.train_batch_size=2 data.micro_batch_size_per_gpu=1 \
  engine.optimizer_offload=true trainer.logger='[console]'
ls "$RUNS/sft-sp/ckpt/global_step_2/huggingface"
