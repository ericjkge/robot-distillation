#!/bin/bash
# Goal task 9's 24 untried trials (goal9.json, from baseline/remaining.py on Nov 26) on one Google Cloud machine of
# 8 H100s, run as the O2 collection runs them (main's scripts/collect.sh: the teacher at xhigh, 160,000 response
# tokens), on PEAR's own H100 recipe (two TP4 replicas), all 24 at once; then the records and successful episodes go
# to the Hub dataset as source "gcp" (collect/hub.py's layout), which the SFT set pulls (collect/pull.py).
#   HF_TOKEN=<a write token for the dataset> bash goal9.sh     # in this directory, on the VM
set -euo pipefail
: "${HF_TOKEN:?a Hugging Face write token for the dataset}"
HERE=$(cd "$(dirname "$0")" && pwd)
export WORK_DIR=$HOME/work
CODE=$HOME/robot-distillation
RUN=$HOME/runs/goal9
REPO=eddykang06/qwen3.8-27b-pear-libero-rollouts

[ -d "$CODE" ] || git clone -q https://github.com/ericjkge/robot-distillation "$CODE"
git -C "$CODE" checkout -q 97059ba
cd "$CODE"
# PEAR's environment, LIBERO and the teacher; a rerun skips what is there
[ -d "$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only" ] ||
  BENCHES=libero MODEL_FILES=configs/models/teacher.sh bash scripts/setup.sh 2>&1 | tee -a "$HOME/setup.log"

GPU=h100 SESSIONS_PER_REPLICA=12 LIBERO_SERVERS=24 EPISODES=$HERE/goal9.json \
  MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only RUN=$RUN bash scripts/run.sh scripts/collect.sh

STAGE=$HOME/stage
python3 "$HERE/../collect/hub.py" "$STAGE" gcp "$RUN"
"$WORK_DIR/envs/verl/bin/python" - "$STAGE" "$REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
stage, repo = sys.argv[1:]
HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=stage, commit_message="goal 9 from gcp",
                      allow_patterns=["libero_goal/records/gcp.jsonl", "libero_goal/episodes/*.pt"])
EOF
echo "done: $(grep -c . "$STAGE/libero_goal/records/gcp.jsonl") episodes run, $(ls "$STAGE/libero_goal/episodes" | wc -l) kept"
