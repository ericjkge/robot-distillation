#!/bin/bash
# Teacher-episode collection, one LIBERO suite per job (SUITE): the 27B teacher at xhigh effort over the suite's
# 500 episodes (10 tasks x 50 initial states), PEAR's protocol, keeping every successful episode's full multi-round
# context under $RUNS/<suite>/episodes. One TP4 replica on the machine's four L40S (54 GB of weights in bf16).
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5; a preempted suite reruns whole
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=20  # gpu_requeue's cap
# memory unmeasured (SFT peaked at 78 GB): resize to 1.5x the first suite's MaxRSS
#SBATCH --mem=128G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/collect-%x-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=/n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/collect
SUITE=${SUITE:?libero_goal | libero_spatial | libero_object | libero_10}
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

RUN=$OUT/runs/collect-${CODE_SHA:0:7}/$SUITE
rm -rf "$RUN"  # a requeued job reruns the suite from the start
start=$SECONDS
LIBERO_SERVERS=8 EPISODES=$HERE/$SUITE.json TP=4 MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only RUN=$RUN \
  bash scripts/run.sh scripts/collect.sh transfer_queue.backend.SimpleStorage.num_data_storage_units=4  # 20 CPUs: 8 simulators, 4 storage units
echo "collect $SUITE: $((SECONDS - start)) s, $(ls "$RUN/episodes" | wc -l) episodes kept"
