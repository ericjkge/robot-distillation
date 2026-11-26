#!/bin/bash
# For SFT job 60783661 (it runs the earlier sft/staged.sh, which would draw stage 2's set unbalanced), called after each
# collect/sync.sh (so it never reads the index mid-write): once every task is complete on the Hub (sync.sh's stage) and
# the human has not held it ($OUT/hub/stage2.wait), builds stage 2's balanced set (sft/final_set.py) where the job looks
# for it, then lifts the job's hold. If stage 1 ends first, the job waits under the hold.
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=$(cd "$(dirname "$0")/.." && pwd)
COLLECT=$OUT/runs/collect-bf9b274
FINAL=$COLLECT/sft-set-final
[ -d "$FINAL" ] && exit 0
python3 "$HERE/collect/complete.py" "$OUT/hub/index.jsonl" > /dev/null 2>&1 || exit 0
[ -e "$OUT/hub/stage2.wait" ] && exit 0
rm -rf "$FINAL.building"
python3 "$HERE/sft/final_set.py" "$OUT/hub" "$COLLECT/sft-set" "$FINAL.building"
mv "$FINAL.building" "$FINAL"
rm -f "$OUT/hub/qwen3.8-27b-pear-libero-rollouts/stage2.hold"
echo "$(date +%H:%M) stage 2 set ready, hold lifted"
