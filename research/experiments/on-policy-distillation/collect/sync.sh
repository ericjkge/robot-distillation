#!/bin/bash
# Syncs the Hub dataset with every O2 collection run: integrates other collectors' PRs, pulls the other sources' records, lays out O2's (source o2),
# uploads what is new (not the SFT job's own stage, which sits inside this one) and prints the per-task tally of tasks not yet complete. Run each loop tick while O2 collects.
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
REPO=eddykang06/qwen3.8-27b-pear-libero-rollouts
S=$OUT/hub; BIN=$OUT/work/envs/verl/bin; HERE=$(dirname "$0")
# other collectors' open PRs, integrated in the dataset's format (or left open, with the reason, in $OUT/hub/prs.log)
"$BIN/python" "$HERE/prs.py" "$S" $REPO o2 "$OUT"/runs/collect-bf9b274/libero_*/attempt-* 2>&1 | grep "^PR #" | sed "s/^/$(date +%H:%M) /" | tee -a "$OUT/hub/prs.log" || true
"$BIN/python" "$HERE/pull.py" "$S" $REPO 2>&1 | tail -1
python3 "$HERE/hub.py" "$S" o2 "$OUT"/runs/collect-bf9b274/libero_*/attempt-* | tail -1
"$BIN/hf" upload-large-folder $REPO "$S" --repo-type dataset --exclude "qwen3.8-27b-pear-libero-rollouts/*" --num-workers 4 --no-bars >/dev/null 2>&1; echo upload-exit=$?
python3 "$HERE/complete.py" "$S/index.jsonl" || true
