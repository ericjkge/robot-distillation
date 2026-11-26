#!/bin/bash
# GPU smoke test, part 3 of 3, after part 2, on four GPUs: distill from part 2's SFT model, 2 updates after a
# validation pass, on two trainer GPUs, a rollout and a teacher GPU (one trainer GPU held 60461359's 4B in fp32 with
# too little room for its update); export the result with PEAR's script and evaluate it on the 2 episodes. check.py
# verifies each step's outputs. The actor keeps its fp32 state on the CPU (FSDP2's offload_policy, as SFT does): with
# PEAR's optimizer_offload, Adam's step brought it onto the trainer GPUs and ran out of memory (60471092). PEAR's
# optimizer_offload and optimizer_offload_step go off with it: verl allows them only without offload_policy.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5: a short, restartable job
#SBATCH --gres=gpu:4,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=16
# host memory: 96G was killed in the first update; the 4B SFT pilot with its state on the CPU peaked at 147 GB
#SBATCH --mem=192G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-part3-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh

stage distill env TRAINER_GPUS=2 ROLLOUT_GPUS=1 TEACHER_GPUS=1 TEACHER_TP=1 STEPS=2 BATCH=2 \
  MODEL_FILE=$STUDENT_FILE MODEL=$SFT_MODEL TEACHER=$BASE RUN=$RUNS/distill \
  bash scripts/run.sh scripts/distill.sh chat_template_file="$TEMPLATE" "$UNITS" \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=false
stage export bash third_party/pear/scripts/export_checkpoint.sh "$RUNS/distill/ckpt/global_step_2" "$RUNS/distilled"
stage eval_distilled env ROLLOUT_GPUS=1 MODEL_FILE=$STUDENT_FILE MODEL=$RUNS/distilled RUN=$RUNS/eval-distilled CKPT_STEP=2 \
  bash scripts/run.sh scripts/eval.sh chat_template_file="$TEMPLATE" "$UNITS"

"$WORK_DIR/envs/verl/bin/python" "$SMOKE/check.py" "$RUNS" "$CODE_SHA" "$SLURM_JOB_ID" distill eval-distilled
