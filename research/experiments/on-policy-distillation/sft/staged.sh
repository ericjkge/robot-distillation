#!/bin/bash
# SFT on the teacher's episodes (the human, Nov 27). The SFT job's code clone ($OUT/code/bf9b274) has scripts/sft.sh
# replaced by a call to this script (its original kept as scripts/sft-once.sh); sft/run.sbatch calls it with MODEL, RUN
# and NGPUS. Every episode is trained on 3 epochs, up to 25 successes per task (seed 0) over every source:
#   - every task complete on the Hub (collect/complete.py) within WAIT_HOURS (default 0: at the start): one stage, 3 epochs;
#   - otherwise two stages, so the GPUs do not wait longer: 2 epochs on the episodes there now ($RUN), then, once every
#     task is complete or after WAIT2_HOURS more (default 1), 1 epoch from that checkpoint ($RUN-final) on the final set
#     (sft/final_set.py: stage 1's episodes once, each task topped up to 25 with later ones three times).
# The last checkpoint is named in $RUN/sft.final for OPD. A requeued job resumes in the same mode ($RUN/mode) and stage
# (verl resumes from the run directory's latest checkpoint).
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OUT/work/envs/verl/bin/python}
STAGE=${STAGE:-$OUT/hub/qwen3.8-27b-pear-libero-rollouts}
COLLECT=${COLLECT:-$OUT/runs/collect-bf9b274}
SETS=${SETS:-$COLLECT}
ONCE=$PWD/scripts/sft-once.sh
mkdir -p "$RUN"
# the stage refreshed from the Hub and every task complete (a test may define its own)
declare -F ready > /dev/null || ready() {
  "$PY" "$HERE/collect/pull.py" "$STAGE" eddykang06/qwen3.8-27b-pear-libero-rollouts &&
    python3 "$HERE/collect/hub.py" "$STAGE" o2 "$COLLECT"/libero_*/attempt-* &&
    python3 "$HERE/collect/complete.py" "$STAGE/index.jsonl"
}
declare -F draw > /dev/null || draw() {  # draw <out>: up to 25 per task over the stage (main's opd.select_episodes)
  local code=$OUT/code/6005c4f
  [ -d "$code" ] || { git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$code"; git -C "$code" checkout -q 6005c4f; }
  PYTHONPATH=$code/src "$PY" -m opd.select_episodes --per-task 25 --seed 0 --out "$1" "$STAGE"
}
declare -F top_up > /dev/null || top_up() { python3 "$HERE/sft/final_set.py" "$STAGE" "$1" "$2"; }
wait_ready() {  # wait_ready <hours>: 0 once every task is complete, 1 if the hours run out first
  local deadline=$((SECONDS + $1 * 3600))
  until ready; do
    ((SECONDS < deadline)) || return 1
    sleep 600
  done
}
build() {  # build <set> <command...>: the set, built once (a requeue keeps it)
  [ -d "$1" ] && return
  rm -rf "$1.part"
  "${@:2}" "$1.part"
  mv "$1.part" "$1"
}
train() {  # train <model> <set> <run> <epochs>
  echo "== $(date +%H:%M) SFT on $(ls "$2"/*.pt | wc -l) episodes, $4 epochs, from $1"
  MODEL=$1 EPISODES_DIR=$2 RUN=$3 bash "$ONCE" "${EXTRA[@]}" trainer.total_epochs="$4"
}
last() { ls -d "$1"/ckpt/global_step_*/huggingface | sort -V | tail -1; }
EXTRA=("$@")

if [ ! -e "$RUN/mode" ]; then
  if wait_ready "${WAIT_HOURS:-0}"; then echo single > "$RUN/mode"; else echo two > "$RUN/mode"; fi
fi
if [ "$(cat "$RUN/mode")" = single ]; then
  build "$SETS/sft-set-final" draw
  train "$MODEL" "$SETS/sft-set-final" "$RUN" 3
  last "$RUN" > "$RUN/sft.final"
else
  echo "== $(date +%H:%M) tasks still short after ${WAIT_HOURS:-0} h: two stages"
  build "$SETS/sft-set-stage1" draw
  if [ ! -e "$RUN/stage1.done" ]; then  # an if, not ||: set -e then stops at a failed stage 1
    train "$MODEL" "$SETS/sft-set-stage1" "$RUN" 2
    touch "$RUN/stage1.done"
  fi
  wait_ready "${WAIT2_HOURS:-1}" || echo "== $(date +%H:%M) tasks still short: stage 2 on what is there"
  build "$SETS/sft-set-stage2" top_up "$SETS/sft-set-stage1"
  train "$(last "$RUN")" "$SETS/sft-set-stage2" "$RUN-final" 1
  last "$RUN-final" > "$RUN/sft.final"
  rm -rf "$RUN/ckpt"  # one SFT checkpoint: stage 2's
fi
echo "== $(date +%H:%M) SFT done: $(cat "$RUN/sft.final")"
