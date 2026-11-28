#!/usr/bin/env bash
# Step 3: on-policy distillation of a student from its SFT checkpoint, on trainer, rollout and teacher machines (or one
# machine whose GPUs TRAINER_GPUS, ROLLOUT_GPUS and TEACHER_GPUS split).
#   MODEL_FILE=configs/models/<student>.sh MODEL=<SFT checkpoint> TEACHER=<teacher dir> RUN=<run dir> \
#     [STEPS=200] [BATCH=32] bash scripts/run.sh scripts/distill.sh [overrides]
source "$(dirname "$0")/common.sh"

TRAINER_NNODES=${TRAINER_NNODES:?the Ray machines started as trainer}
TEACHER_NNODES=${TEACHER_NNODES:?the GPU machines of the teacher servers}
TEACHER=${TEACHER:?the teacher model directory}
TEACHER_TP=${TEACHER_TP:-2}  # a 27B replica: 54 GB of bf16 weights
STEPS=${STEPS:-200}
BATCH=${BATCH:-32}  # episodes per update; OPD scores every token, so small batches work (Lu et al., 2025)
# evaluated every trainer.test_freq updates on PEAR's LIBERO-Goal validation episodes; Step 4 evaluates in full
VAL_EPISODES=${VAL_EPISODES:-$PEAR/pear/envs/libero/configs/libero_goal_val.json}
TEACHER_HARDWARE_ARGS=()  # a hardware profile's settings for the teacher (configs/hardware)
pear_args

RUN_ARGS+=(
  "${MODEL_TRAINING_ARGS[@]}"
  data.train_batch_size="$BATCH"
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH"
  trainer.nnodes="$TRAINER_NNODES"
  trainer.n_gpus_per_node="${TRAINER_GPUS:-$GPUS_PER_NODE}"
  trainer.total_training_steps="$STEPS"
  trainer.default_local_dir="$RUN/ckpt"
  distillation.nnodes="$TEACHER_NNODES"
  distillation.n_gpus_per_node="${TEACHER_GPUS:-$GPUS_PER_NODE}"
  distillation.teacher_models.teacher_model.model_path="$TEACHER"
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP"
  # verl requires room for the student's prompt and full response plus one token; the model's own 262,144 leaves a
  # 48 GB GPU no KV cache for it
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_MODEL_LEN + 1))
  # each episode reaches the teacher once, so its prefix cache never helps; with it, vLLM's prefill of this hybrid
  # model stalls past about 100k tokens (27B on 2 L40S: a 159k-token episode never returned; without it, 61 s)
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=false
  "${TEACHER_HARDWARE_ARGS[@]}"
)
pear_exec distill "$@"
