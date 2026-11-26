#!/bin/bash
# Evaluation of any model (a base model, an SFT or OPD checkpoint's huggingface/ directory, or the teacher) on a PEAR
# episode file, by PEAR's evaluation protocol (examples/eval/: one run per episode, 256,000 response tokens: main's
# scripts/eval.sh from a0aa10e), at xhigh
# effort. One hardware for every model, so that times per episode compare: 2 L40S, 4 episodes at once per replica (the
# teacher's TP2 replica holds about 4 long contexts; students run a replica per GPU). SHARD=i/n runs every n-th
# episode, so a set splits across jobs; a requeued job resumes (main's opd.episodes).
#   DESC=... NAME=sft-9b-step120 MODEL=<model dir> MODEL_FILE=configs/models/qwen3_5_9b.sh CODE_SHA=<sha> \
#     [BENCH=libero | libero_pro | libero_plus] [EPISODES=<episode file>] [SHARD=0/8] research/bin/submit \
#     on-policy-distillation research/experiments/on-policy-distillation/eval/run.sbatch
# The teacher: MODEL_FILE=configs/models/teacher.sh (TP2). Records land in runs/eval-<sha>/<BENCH>/<NAME>/, which
# main's opd.report reads.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5; a requeued job resumes
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
# host memory: the teacher's collection peaked at 75 GB
#SBATCH --mem=128G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/eval-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
NAME=${NAME:?a name for the evaluated model, e.g. sft-9b-step120}
MODEL=${MODEL:?the model directory}
MODEL_FILE=${MODEL_FILE:?configs/models/qwen3_5_9b.sh | qwen3_5_4b.sh | teacher.sh}
BENCH=${BENCH:-libero}
SHARD=${SHARD:-0/1}
CODE_SHA=${CODE_SHA:?the commit of the code to run}
CODE=$OUT/code/$CODE_SHA
export WORK_DIR=$OUT/work
exec 9> "$OUT/code/.lock"
flock 9  # the jobs may start together
if [ ! -d "$CODE" ]; then
  git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
  git -C "$CODE" checkout -q "$CODE_SHA"
  git -C "$CODE" submodule update -q --init
  bash "$CODE/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh"
fi
flock -u 9
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"
echo "model $MODEL"

if [ "$BENCH" = libero ]; then
  EPISODES=${EPISODES:-$CODE/third_party/pear/pear/envs/libero/configs/libero_all.json}
else
  EPISODES=${EPISODES:-$CODE/third_party/pear/pear/envs/$BENCH/configs/all.json}
fi
SHARD_DIR=$OUT/runs/eval-${CODE_SHA:0:7}/$BENCH/$NAME/shard-${SHARD/\//-of-}
RUN=$SHARD_DIR/attempt-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}
mkdir -p "$RUN"
PYTHONPATH=src "$WORK_DIR/envs/verl/bin/python" -m opd.episodes "$EPISODES" "$SHARD_DIR" --shard "$SHARD" > "$RUN/episodes.json"
if [ ! -s "$RUN/episodes.json" ]; then
  echo "eval $NAME $BENCH shard $SHARD: every episode is done"
  exit 0
fi
start=$SECONDS
# 12 CPUs: Ray reserves 2 per GPU for the replicas' placement groups, which leaves 8 for 4 simulators, 2 storage units
# and the queue's controller
SESSIONS_PER_REPLICA=4 LIBERO_SERVERS=4 BENCH=$BENCH EPISODES=$RUN/episodes.json \
  MODEL_FILE=$MODEL_FILE MODEL=$MODEL RUN=$RUN \
  bash scripts/run.sh scripts/eval.sh +data.apply_chat_template_kwargs.reasoning_effort=xhigh \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2
echo "eval $NAME $BENCH shard $SHARD: $((SECONDS - start)) s"
