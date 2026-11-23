#!/bin/bash
# Untuned-student baseline, one LIBERO suite per job (SUITE) of one student (STUDENT=Qwen3.5-9B | Qwen3.5-4B): PEAR's
# protocol at xhigh effort over the suite's 500 episodes (10 tasks x 50 initial states, the episode files collection
# uses); the four suites together are LIBERO's 2,000. TRIALS (default 0-49) takes a range of the initial states, so a
# suite can be split across jobs. One replica on one GPU of 45 GB or more.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5; a preempted suite reruns whole
#SBATCH --requeue
#SBATCH --gres=gpu:1,vram:45G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=10
# host memory unmeasured: the 27B teacher's collection peaked at 75 GB; resize to 1.5x the first job's MaxRSS
#SBATCH --mem=64G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/baseline-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
EPISODES_DIR=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/collect
SUITE=${SUITE:?libero_goal | libero_spatial | libero_object | libero_10}
TRIALS=${TRIALS:-0-49}
STUDENT=${STUDENT:?Qwen3.5-9B | Qwen3.5-4B}
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

case $STUDENT in
  Qwen3.5-9B) MODEL_FILE=configs/models/qwen3_5_9b.sh ;;
  Qwen3.5-4B) MODEL_FILE=configs/models/qwen3_5_4b.sh ;;
  *) echo "unknown student $STUDENT" >&2; exit 1 ;;
esac
RUN=$OUT/runs/baseline-${CODE_SHA:0:7}/$STUDENT/$SUITE/trials-$TRIALS
rm -rf "$RUN"  # a requeued job reruns its episodes from the start
mkdir -p "$RUN"
sed "s/\"trial_id\": \"0-49\"/\"trial_id\": \"$TRIALS\"/" "$EPISODES_DIR/$SUITE.json" > "$RUN/episodes.json"
start=$SECONDS
# 10 CPUs: Ray reserves 2 for the replica's placement group, which leaves 8 for 4 simulators, 2 storage units and the
# queue's controller. 8 episodes at once, not PEAR's 32, so their long contexts stay in one GPU's KV cache (at 32 the
# teacher's collection evicted and recomputed them)
SESSIONS_PER_REPLICA=8 LIBERO_SERVERS=4 EPISODES=$RUN/episodes.json MODEL_FILE=$MODEL_FILE MODEL=$WORK_DIR/models/$STUDENT RUN=$RUN \
  bash scripts/run.sh scripts/eval.sh +data.apply_chat_template_kwargs.reasoning_effort=xhigh \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2
echo "baseline $STUDENT $SUITE trials $TRIALS: $((SECONDS - start)) s"
cat "$RUN"/val_dumps/eval_0_summary.json 2>/dev/null || true
