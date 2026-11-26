#!/bin/bash
# Goal task 9's 24 untried trials (gcp/goal9.json) on one Colab GPU of 80 GB (H100 or A100), run as the O2 collection
# runs them (main's scripts/collect.sh: the teacher at xhigh, 160,000 response tokens), one TP1 replica with 3 episodes
# at once. Everything that must outlive the session (PEAR's environments as a tarball, the run's records and episodes)
# is on Google Drive, so rerunning after a disconnect resumes: only the trials without a record run again. SHARD=i/n
# splits the trials across notebooks. At the end the records and successful episodes go to the Hub dataset as source
# colab-<i> (collect/hub.py's layout), which O2's SFT set pulls (collect/pull.py).
#   HF_TOKEN=<write token> SHARD=0/1 bash goal9.sh
set -euo pipefail
: "${HF_TOKEN:?a Hugging Face write token for the dataset}"
SHARD=${SHARD:-0/1}
DRIVE=/content/drive/MyDrive/robot-distillation
CODE=/content/robot-distillation
export WORK_DIR=/content/work
RUN=$DRIVE/goal9/shard-${SHARD/\//-of-}
REPO=eddykang06/qwen3.8-27b-pear-libero-rollouts
BRANCH=claude/on-policy-distillation
EXP=research/experiments/on-policy-distillation
mkdir -p "$DRIVE" "$RUN"

# the code: main with opd.episodes, and this direction's episode file and Hub layout
if [ ! -d "$CODE" ]; then
  git clone -q https://github.com/ericjkge/robot-distillation "$CODE"
  git -C "$CODE" checkout -q 97059ba
  git -C "$CODE" fetch -q origin "$BRANCH"
  for f in gcp/goal9.json collect/hub.py; do git -C "$CODE" show "FETCH_HEAD:$EXP/$f" > "$CODE/$(basename "$f")"; done
fi
cd "$CODE"

# PEAR's environments and LIBERO: built once (1-2 h), kept on Drive as a tarball; the teacher from the Hub each session
if [ ! -d "$WORK_DIR/envs/verl" ]; then
  if [ -f "$DRIVE/envs.tar" ]; then
    mkdir -p "$WORK_DIR" && tar -xf "$DRIVE/envs.tar" -C "$WORK_DIR"
  else
    BENCHES=libero MODEL_FILES=/dev/null bash scripts/setup.sh 2>&1 | tee -a "$DRIVE/setup.log"
    tar -cf "$DRIVE/envs.tar" -C "$WORK_DIR" envs
  fi
fi
[ -d "$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only" ] ||
  MODEL_FILE=configs/models/teacher.sh bash third_party/pear/scripts/download_model.sh

# resume: the trials of this shard without a record on Drive
ATTEMPT=$RUN/attempt-$(date +%s)
mkdir -p "$ATTEMPT"
PYTHONPATH=src "$WORK_DIR/envs/verl/bin/python" -m opd.episodes goal9.json "$RUN" --shard "$SHARD" > "$ATTEMPT/episodes.json"
if [ -s "$ATTEMPT/episodes.json" ]; then
  TP=1 SESSIONS_PER_REPLICA=3 LIBERO_SERVERS=3 EPISODES=$ATTEMPT/episodes.json \
    MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only RUN=$ATTEMPT bash scripts/run.sh scripts/collect.sh \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
fi

STAGE=/content/stage
SOURCE=colab-${SHARD%/*}
python3 hub.py "$STAGE" "$SOURCE" "$RUN"/attempt-*
"$WORK_DIR/envs/verl/bin/python" - "$STAGE" "$REPO" "$SOURCE" <<'EOF'
import sys
from huggingface_hub import HfApi
stage, repo, source = sys.argv[1:]
HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=stage, commit_message=f"goal 9 from {source}",
                      allow_patterns=[f"libero_goal/records/{source}.jsonl", "libero_goal/episodes/*.pt"])
EOF
echo "done: $(grep -c . "$STAGE/libero_goal/records/$SOURCE.jsonl") episodes run, $(ls "$STAGE/libero_goal/episodes" | wc -l) kept"
