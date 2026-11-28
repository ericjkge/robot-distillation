#!/bin/bash
# On-policy distillation of the 9B from its SFT checkpoint (stage 2's last, from the Hub), as separate Slurm jobs that
# form one Ray cluster (opd/node.sh): this trainer job (4 GPUs), a teacher job (2 GPUs, TP 2) and ROLLOUT_JOBS rollout
# jobs (2 GPUs each, one 9B replica per GPU, 8 episodes each at once). Settings: the note's OPD configuration table.
#   trainer: ROLE=trainer ROLLOUT_JOBS=<n> CODE=47200a9-mbfix SFT_SHA=bf9b274
#   others:  ROLE=teacher|rollout TRAINER_JOB=<trainer id> (same CODE, SFT_SHA, ROLLOUT_JOBS),
#            --dependency=after:<trainer id> --gres=gpu:2,vram:48G; teacher -c 6 --mem=120G, rollout -c 14 --mem=160G
# STEPS is a cap: the run is stopped by hand. PEAR checkpoints every update and verl resumes from the latest.
#SBATCH --partition=gpu_quad  # not preemptible: losing one job stops the whole cluster
#SBATCH --gres=gpu:4,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G  # the trainer's fp32 state on the CPU: the SFT probe's 9B step peaked at 212 GB on 2 GPUs
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/opd-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/${CODE:?the code directory under code/, e.g. 47200a9-mbfix}
SFT_SHA=${SFT_SHA:?the SFT run code, e.g. bf9b274}
HERE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/opd
RUN=$OUT/runs/opd-$SFT_SHA/Qwen3.5-9B
export WORK_DIR=$OUT/work VERL_CHANGE_IN_PROGRESS=1 LIBERO_SERVERS=8  # the verl fixes are commits on PEAR's checkout
export ROLE=${ROLE:?trainer | teacher | rollout} TRAINER_JOB=${TRAINER_JOB:-$SLURM_JOB_ID}
export CLUSTER=$RUN/cluster-$TRAINER_JOB NODES=$((2 + ${ROLLOUT_JOBS:?rollout jobs}))
cd "$CODE"
echo "$ROLE on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"
[ "$ROLE" = trainer ] || exec bash "$HERE/node.sh"

# the student: the one SFT checkpoint (sft.final, written when SFT ends), from its Hub repo once it holds it
FINAL=$(cat "$OUT/runs/sft-$SFT_SHA/Qwen3.5-9B/sft.final")
SFT=$("$WORK_DIR/envs/verl/bin/python" "$HERE/sft_from_hub.py" "$FINAL" eddykang06/qwen3.5-9b-pear-libero-sft "$OUT/models")
echo "student $SFT (SFT $FINAL)"
T=distillation.teacher_models.teacher_model.inference
bash "$HERE/node.sh" env TRAINER_NNODES=1 TEACHER_NNODES=1 ROLLOUT_NNODES="$ROLLOUT_JOBS" \
  TRAINER_GPUS=4 ROLLOUT_GPUS=2 TEACHER_GPUS=2 TEACHER_TP=2 SESSIONS_PER_REPLICA=8 STEPS="${STEPS:-1000}" BATCH=32 \
  MODEL_FILE=configs/models/qwen3_5_9b.sh MODEL="$SFT" TEACHER="$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only" RUN="$RUN" \
  bash scripts/run.sh scripts/distill.sh \
  +data.apply_chat_template_kwargs.reasoning_effort=xhigh transfer_queue.backend.SimpleStorage.num_data_storage_units=3 \
  trainer.v1.sampler.max_off_policy_threshold=1 trainer.val_before_train=false \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=false \
  $T.max_model_len=164097 $T.gpu_memory_utilization=0.9 $T.max_num_batched_tokens=2048  # as main 4e259bc
