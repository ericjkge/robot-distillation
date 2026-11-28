#!/bin/bash
# opd/node.sh end to end at smoke scale: part 3's distillation (Qwen3.5-4B student and teacher, 2 episodes, 2 updates)
# as three jobs on separate machines: trainer (2 GPUs), teacher (1) and rollout (1).
#   teacher, rollout: smoke/split-node.sbatch with ROLE=teacher|rollout LAUNCH=<tag> CODE_SHA=<code>
#   trainer: this script with ROLE=trainer LAUNCH=<tag> CODE_SHA=<code>, --dependency=after:<teacher id>:<rollout id>
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/smoke-split-%x-%j.out
source /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/smoke/env.sh
export VERL_CHANGE_IN_PROGRESS=1 ROLE=${ROLE:?trainer | teacher | rollout} CLUSTER=$RUNS/cluster-${LAUNCH:?a tag} NODES=3
NODE=$SMOKE/../opd/node.sh
[ "$ROLE" = trainer ] || exec bash "$NODE"

stage distill_split bash "$NODE" env TRAINER_NNODES=1 TEACHER_NNODES=1 ROLLOUT_NNODES=1 \
  TRAINER_GPUS=2 ROLLOUT_GPUS=1 TEACHER_GPUS=1 TEACHER_TP=1 STEPS=2 BATCH=2 \
  MODEL_FILE="$STUDENT_FILE" MODEL="$SFT_MODEL" TEACHER="$BASE" RUN="$RUNS/distill-split-$LAUNCH" \
  bash scripts/run.sh scripts/distill.sh chat_template_file="$TEMPLATE" "$UNITS" \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=false
ls "$RUNS/distill-split-$LAUNCH/ckpt"
