#!/bin/bash
# The SFT set over every cluster's episodes, built by sft/run.sbatch when SFT starts (or by hand, inside a job), once
# the rollouts are complete (collect/complete.py; it waits until then): the other clusters' episodes pulled from the Hub dataset into the stage (collect/pull.py), the index rebuilt with ours
# (collect/hub.py), and up to 25 per task drawn from the stage by main's opd.select_episodes, which reads the Hub layout
# from 6005c4f on (bf9b274's, which SFT trains with, reads only collection runs). By hand:
#   srun --jobid=<job> --overlap -n1 --cpus-per-task=4 bash sft/build_set.sh
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=$OUT/work/envs/verl/bin/python
COLLECT=$OUT/runs/collect-bf9b274
STAGE=$OUT/hub/qwen3.8-27b-pear-libero-rollouts
CODE=$OUT/code/6005c4f

# wait, GPUs held, until every cluster's rollouts are complete (pull.py fails while an upload is half done)
until "$PY" "$HERE/collect/pull.py" "$STAGE" eddykang06/qwen3.8-27b-pear-libero-rollouts &&
  python3 "$HERE/collect/hub.py" "$STAGE" o2 "$COLLECT"/libero_*/attempt-* &&
  python3 "$HERE/collect/complete.py" "$STAGE/index.jsonl"; do
  sleep 600
done
[ -d "$CODE" ] || { git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"; git -C "$CODE" checkout -q 6005c4f; }
PYTHONPATH=$CODE/src "$PY" -m opd.select_episodes --per-task 25 --seed 0 --out "$COLLECT/sft-set" "$STAGE"
