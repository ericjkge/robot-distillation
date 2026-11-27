#!/bin/bash
# On-policy distillation of the 9B student from its SFT checkpoint (stage 2's last, from the Hub: sft_from_hub.py),
# with the 27B teacher at xhigh, on one 8-GPU machine: 4 trainer GPUs (FSDP2, the fp32 state on the CPU), 2 rollout
# GPUs (one student replica each, 8 episodes at once) and a TP2 teacher on 2. Algorithm 1: each update trains only on
# episodes of the current policy (max_off_policy_threshold=1), batch 32 (the human, Nov 27), validation (100 LIBERO-Goal episodes) after
# every 10th update and not before the first, as Algorithm 1 evaluates when (k+1) mod 10 = 0. STEPS is an upper
# bound: the run is stopped by hand. PEAR checkpoints every update and verl resumes from the latest, so a requeued job
# continues. CODE is the code to run: 47200a9-mbfix, main 47200a9 with the verl mini-batch fix (see the note).
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5; a requeued job resumes
#SBATCH --requeue
#SBATCH --gres=gpu:8,vram:48G  # A6000
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16 or too small (60796702-03 failed on RTX 8000)
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=20
# host memory: the 9B trainer's fp32 state on the CPU (SFT: about 330 GB) and the teacher's 75 GB
#SBATCH --mem=500G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/opd-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/${CODE:?the code directory under code/, e.g. 47200a9-mbfix}
SFT_SHA=${SFT_SHA:?the SFT run code, e.g. bf9b274}
HERE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/opd
export WORK_DIR=$OUT/work VERL_CHANGE_IN_PROGRESS=1  # the verl fix is a commit on top of PEAR's patched checkout
cd "$CODE"
echo "code $CODE on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"

# the student: the one SFT checkpoint (stage 2's last, named in sft.final when SFT ends), from its Hub repo once it holds it
FINAL=$(cat "$OUT/runs/sft-$SFT_SHA/Qwen3.5-9B/sft.final")
SFT=$("$WORK_DIR/envs/verl/bin/python" "$HERE/sft_from_hub.py" "$FINAL" eddykang06/qwen3.5-9b-pear-libero-sft "$OUT/models")
echo "student $SFT (SFT $FINAL)"
# 20 CPUs: Ray reserves 2 per rollout GPU and per teacher GPU, which leaves 12 for 8 simulators, 3 storage units and
# the queue's controller
TRAINER_GPUS=4 ROLLOUT_GPUS=2 TEACHER_GPUS=2 TEACHER_TP=2 SESSIONS_PER_REPLICA=8 LIBERO_SERVERS=8 STEPS=${STEPS:-1000} BATCH=32 \
  MODEL_FILE=configs/models/qwen3_5_9b.sh MODEL=$SFT TEACHER=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only \
  RUN=$OUT/runs/opd-$SFT_SHA/Qwen3.5-9B bash scripts/run.sh scripts/distill.sh \
  +data.apply_chat_template_kwargs.reasoning_effort=xhigh transfer_queue.backend.SimpleStorage.num_data_storage_units=3 \
  trainer.v1.sampler.max_off_policy_threshold=1 trainer.val_before_train=false \
  actor_rollout_ref.actor.fsdp_config.offload_policy=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=false \
  distillation.teacher_models.teacher_model.inference.max_model_len=164097 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.9 \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=2048  # as main 4e259bc: the teacher on 2 x 48 GB
