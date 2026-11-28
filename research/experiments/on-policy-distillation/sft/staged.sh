#!/bin/bash
# SFT in one stage on the final teacher set (the human, Nov 27). The SFT job's code clone ($OUT/code/bf9b274) has
# scripts/sft.sh replaced by a call to this script (its original kept as scripts/sft-once.sh); sft/run.sbatch calls it
# with MODEL, RUN and NGPUS.
#   1. waits, GPUs held, until every task is complete over every source on the Hub (collect/complete.py), for at most
#      WAIT_HOURS (default 3), then goes on with what is there;
#   2. draws up to 25 successful episodes per task (seed 0) over every source (main's opd.select_episodes);
#   3. trains 3 epochs on them, and names the last checkpoint in $RUN/sft.final for OPD (opd/run.sbatch).
# A requeued job resumes: the set is kept, and verl resumes from the run directory's latest checkpoint.
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OUT/work/envs/verl/bin/python}
STAGE=${STAGE:-$OUT/hub/qwen3.8-27b-pear-libero-rollouts}
COLLECT=${COLLECT:-$OUT/runs/collect-bf9b274}
SET=${FINAL_SET:-$COLLECT/sft-set-final}
ONCE=$PWD/scripts/sft-once.sh
# the stage refreshed from the Hub and every task complete (a test may define its own)
declare -F ready > /dev/null || ready() {
  "$PY" "$HERE/collect/pull.py" "$STAGE" eddykang06/qwen3.8-27b-pear-libero-rollouts &&
    python3 "$HERE/collect/hub.py" "$STAGE" o2 "$COLLECT"/libero_*/attempt-* &&
    python3 "$HERE/collect/complete.py" "$STAGE/index.jsonl"
}
declare -F draw > /dev/null || draw() {
  local code=$OUT/code/6005c4f  # main's opd.select_episodes, which reads the Hub layout
  [ -d "$code" ] || { git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$code"; git -C "$code" checkout -q 6005c4f; }
  PYTHONPATH=$code/src "$PY" -m opd.select_episodes --per-task 25 --seed 0 --out "$1" "$STAGE"
}

if [ ! -d "$SET" ]; then
  deadline=$((SECONDS + ${WAIT_HOURS:-3} * 3600))
  until ready; do
    ((SECONDS < deadline)) || { echo "== $(date +%H:%M) tasks still short after ${WAIT_HOURS:-3} h: training on what is there"; break; }
    sleep 600
  done
  rm -rf "$SET.part"
  draw "$SET.part"
  mv "$SET.part" "$SET"
fi
echo "== $(date +%H:%M) SFT: $(ls "$SET"/*.pt | wc -l) episodes, 3 epochs"
MODEL=$MODEL EPISODES_DIR=$SET RUN=$RUN bash "$ONCE" "$@" trainer.total_epochs=3
ls -d "$RUN"/ckpt/global_step_*/huggingface | sort -V | tail -1 > "$RUN/sft.final"
echo "== $(date +%H:%M) SFT done: $(cat "$RUN/sft.final")"
