#!/bin/bash
# Teacher-episode collection, one LIBERO suite per job (SUITE): the 27B teacher at xhigh effort over the suite's
# 500 episodes (10 tasks x 50 initial states), PEAR's protocol, keeping every successful episode's full multi-round
# context under $RUNS/<suite>/episodes, until each task has the 25 that SFT draws or has ended its 50 trials
# (enough.py). One TP2 replica on any two GPUs of 48 GB or more: L40S, A6000 or A100 (54 GB of weights in bf16); two
# GPUs start far sooner than four on gpu_requeue. A suite may be split across jobs by TASKS (e.g. 0-4 and 5-9).
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5; a requeued job resumes
#SBATCH --requeue
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=11  # the gpu partition's QOS allows 34 CPUs per user: three of these jobs
# host memory: libero_goal peaked at 75 GB (MaxRSS)
#SBATCH --mem=128G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/collect-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/collect
SUITE=${SUITE:?libero_goal | libero_spatial | libero_object | libero_10}
TASKS=${TASKS:-0-9}
CODE_SHA=${CODE_SHA:?the commit of the code to run}
CODE=$OUT/code/$CODE_SHA
export WORK_DIR=$OUT/work
exec 9> "$OUT/code/.lock"
flock 9  # the suites' jobs may start together
if [ ! -d "$CODE" ]; then
  git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
  git -C "$CODE" checkout -q "$CODE_SHA"
  git -C "$CODE" submodule update -q --init
  bash "$CODE/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh"
fi
flock -u 9
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"

# each attempt (a requeue starts another) writes its own directory and runs only the episodes of the tasks still short of
# 25 kept that no attempt has finished
SUITE_DIR=$OUT/runs/collect-${CODE_SHA:0:7}/$SUITE
RUN=$SUITE_DIR/attempt-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}
mkdir -p "$RUN"
python3 "$HERE/../baseline/remaining.py" "$HERE/$SUITE.json" 0-49 "$SUITE_DIR" --keep 25 --tasks "$TASKS" \
  > "$RUN/episodes.json"
if [ ! -s "$RUN/episodes.json" ]; then
  echo "collect $SUITE tasks $TASKS: every task is done"
  exit 0
fi
start=$SECONDS
# 11 CPUs: Ray reserves 4 for the replica's placement group (2 per GPU), which leaves 7 for 4 simulators (one per
# episode at once), 2 storage units and the queue's controller. 4 episodes at once: a TP4 replica on 4 L40S holds about 13 long contexts (at PEAR's
# 32 it evicted and recomputed them: libero_goal 60470355, 97% full, 208 preemptions), and TP2 has about a quarter of
# that KV cache
SESSIONS_PER_REPLICA=4 LIBERO_SERVERS=4 EPISODES=$RUN/episodes.json TP=2 MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only RUN=$RUN \
  setsid bash scripts/run.sh scripts/collect.sh transfer_queue.backend.SimpleStorage.num_data_storage_units=2 &
COLLECT=$!
until ! kill -0 "$COLLECT" 2>/dev/null || "$WORK_DIR/envs/verl/bin/python" "$HERE/enough.py" "$SUITE_DIR" --tasks "$TASKS"; do sleep 60; done
if kill -0 "$COLLECT" 2>/dev/null; then
  kill -- -"$COLLECT"  # every task has what SFT draws: the run's process group, Ray with it
  wait "$COLLECT" || true
else
  wait "$COLLECT"
fi
echo "collect $SUITE tasks $TASKS: $((SECONDS - start)) s, $(ls "$RUN/episodes" | wc -l) episodes kept"
