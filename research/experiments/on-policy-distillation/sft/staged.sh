#!/bin/bash
# SFT in two stages so the rollouts still coming in are trained on last (the human, Nov 26 11:40), in one allocation.
# The running SFT job's code clone ($OUT/code/bf9b274) has scripts/sft.sh replaced by a call to this script, its
# original kept as scripts/sft-once.sh; sft/run.sbatch calls it from the clone with MODEL, EPISODES_DIR, RUN, NGPUS.
#   stage 1: the set drawn when SFT started (whatever was on the Hub), 2 epochs, in $RUN
#   stage 2: once the rollouts are complete (collect/complete.py) and not held ($STAGE/stage2.hold), 1 epoch from stage
#            1's last checkpoint, in $RUN-final, over sft/final_set.py's set: stage 1's episodes once and each task
#            topped up to 25 with the late episodes, each 3 times, so every episode gets 3 epochs and the late ones come
#            last.
# A requeued job resumes the stage it was in (verl resumes from the latest checkpoint of its run directory).
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OUT/work/envs/verl/bin/python}
STAGE=${STAGE:-$OUT/hub/qwen3.8-27b-pear-libero-rollouts}
ONCE=$PWD/scripts/sft-once.sh
COLLECT=${COLLECT:-$OUT/runs/collect-bf9b274}
rm -f "$STAGE/sft-stage1"  # the flag that let stage 1 start before the rollouts were complete (collect/complete.py)
# stage 2's start: every task complete over every cluster's episodes on the Hub, and no hold (a test may define its own)
declare -F ready > /dev/null || ready() {
  "$PY" "$HERE/collect/pull.py" "$STAGE" eddykang06/qwen3.8-27b-pear-libero-rollouts &&
    python3 "$HERE/collect/hub.py" "$STAGE" o2 "$COLLECT"/libero_*/attempt-* &&
    python3 "$HERE/collect/complete.py" "$STAGE/index.jsonl" && [ ! -e "$STAGE/stage2.hold" ]
}

if [ ! -e "$RUN/stage1.done" ]; then
  echo "== $(date +%H:%M) stage 1: $(ls "$EPISODES_DIR"/*.pt | wc -l) episodes, 2 epochs"
  bash "$ONCE" "$@" trainer.total_epochs=2
  touch "$RUN/stage1.done"
fi
FIRST=$(ls -d "$RUN"/ckpt/global_step_*/huggingface | sort -V | tail -1)
FINAL=$COLLECT/sft-set-final
if [ ! -d "$FINAL" ]; then
  until ready; do sleep 600; done
  rm -rf "$FINAL.part"
  python3 "$HERE/sft/final_set.py" "$STAGE" "$EPISODES_DIR" "$FINAL.part"
  mv "$FINAL.part" "$FINAL"
fi
EPOCHS=$(cat "$RUN/stage2.epochs" 2> /dev/null || echo 1)  # changeable while stage 1 runs (this script is then open)
echo "== $(date +%H:%M) stage 2: $(ls "$FINAL"/*.pt | wc -l) episodes, $EPOCHS epochs, from $FIRST"
MODEL=$FIRST EPISODES_DIR=$FINAL RUN=$RUN-final bash "$ONCE" "$@" trainer.total_epochs="$EPOCHS"
