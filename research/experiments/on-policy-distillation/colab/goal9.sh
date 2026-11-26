#!/bin/bash
# Goal task 9's 24 untried trials (gcp/goal9.json) on one Colab GPU of 80 GB or more (G4 or H100), run as the O2
# collection runs them (main's scripts/collect.sh: the teacher at xhigh, 160,000 response tokens), one TP1 replica with
# 3 episodes at once (4 on G4's 96 GB, fewer if the machine has few CPUs). The run's records and episodes are on Google
# Drive, so rerunning after a disconnect resumes: only the trials without a record run again; PEAR's environments
# (~17 GB) are saved there too when the Drive has room, so that a later session restores them in minutes instead of
# rebuilding them (1-2 h). SHARD=i/n splits the trials across notebooks. At the end the records and successful episodes
# go to the Hub dataset as source colab-<i> (collect/hub.py's layout), which O2's SFT set pulls (collect/pull.py).
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
TEACHER=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only
mkdir -p "$DRIVE" "$RUN"
step() { echo "== $(date +%H:%M) $*"; }
# while a long step runs: every 2 minutes, the time and the last line of its log (the notebook shows it)
beat() {
  [ -z "${BEAT:-}" ] || kill "$BEAT" 2> /dev/null || true
  (while sleep 120; do echo ".. $(date +%H:%M) $(tail -n 1 "$1" 2> /dev/null | tr -d '\r' | cut -c1-160)"; done) &
  BEAT=$!
}
trap '[ -z "${BEAT:-}" ] || kill "$BEAT" 2> /dev/null || true' EXIT

step "code"
# main with opd.episodes, PEAR (a submodule) and its patched verl checkout, and this direction's episode file and Hub
# layout
if [ ! -f "$CODE/.ready" ]; then
  rm -rf "$CODE"
  git clone -q https://github.com/ericjkge/robot-distillation "$CODE"
  git -C "$CODE" checkout -q 97059ba
  git -C "$CODE" submodule update -q --init
  bash "$CODE/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh"
  git -C "$CODE" fetch -q origin "$BRANCH"
  for f in gcp/goal9.json collect/hub.py; do git -C "$CODE" show "FETCH_HEAD:$EXP/$f" > "$CODE/$(basename "$f")"; done
  touch "$CODE/.ready"
fi
cd "$CODE"

# PEAR's environments and LIBERO, and the teacher (52 GB, from the Hub)
if [ ! -f "$WORK_DIR/.ready" ]; then
  rm -rf "$WORK_DIR/envs"
  if [ -f "$DRIVE/envs.tar" ] && step "restoring PEAR's environments from Drive" && mkdir -p "$WORK_DIR" &&
    tar -xf "$DRIVE/envs.tar" -C "$WORK_DIR"; then
    step "teacher weights (52 GB)"
    beat "$DRIVE/download.log"
    MODEL_FILE=configs/models/teacher.sh bash third_party/pear/scripts/download_model.sh > "$DRIVE/download.log" 2>&1 ||
      { tail -n 30 "$DRIVE/download.log"; exit 1; }
  else
    step "building PEAR's environments and downloading the teacher (1-2 h; full log: $DRIVE/setup.log)"
    rm -rf "$WORK_DIR/envs"
    beat "$DRIVE/setup.log"
    BENCHES=libero MODEL_FILES=configs/models/teacher.sh bash scripts/setup.sh >> "$DRIVE/setup.log" 2>&1 ||
      { tail -n 30 "$DRIVE/setup.log"; exit 1; }
    step "saving the environments to Drive for later sessions"
    tar -cf "$DRIVE/envs.tar.part" -C "$WORK_DIR" envs && mv "$DRIVE/envs.tar.part" "$DRIVE/envs.tar" ||
      { rm -f "$DRIVE/envs.tar.part"; step "not saved (Drive full?): a later session rebuilds them"; }
  fi
  touch "$WORK_DIR/.ready"
fi

# resume: the trials of this shard without a record on Drive
ATTEMPT=$RUN/attempt-$(date +%s)
mkdir -p "$ATTEMPT"
PYTHONPATH=src "$WORK_DIR/envs/verl/bin/python" -m opd.episodes goal9.json "$RUN" --shard "$SHARD" > "$ATTEMPT/episodes.json"
# episodes at once: what the KV cache beside the 54 GB of weights holds (G4's 96 GB: 4; H100's 80 GB: 3), and what the
# CPUs hold: Ray reserves 2 for the replica, the queue 3, and each episode's simulator takes 1
SESSIONS=$(($(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) > 90000 ? 4 : 3))
SESSIONS=$((SESSIONS < $(nproc) - 5 ? SESSIONS : $(nproc) - 5))
[ "$SESSIONS" -ge 1 ] || { echo "Error: $(nproc) CPUs, too few for the simulators" >&2; exit 1; }
if [ -s "$ATTEMPT/episodes.json" ]; then
  beat "$ATTEMPT/train.log"
  step "running $(grep -o '"trial_id"' "$ATTEMPT/episodes.json" | wc -l) episodes, $SESSIONS at once (full log: $ATTEMPT/train.log)"
  TP=1 SESSIONS_PER_REPLICA=$SESSIONS LIBERO_SERVERS=$SESSIONS EPISODES=$ATTEMPT/episodes.json MODEL=$TEACHER \
    RUN=$ATTEMPT bash scripts/run.sh scripts/collect.sh transfer_queue.backend.SimpleStorage.num_data_storage_units=2
fi

step "uploading to the Hub"
beat /dev/null
STAGE=/content/stage
SOURCE=colab-${SHARD%/*}
rm -rf "$STAGE"
python3 hub.py "$STAGE" "$SOURCE" "$RUN"/attempt-*
"$WORK_DIR/envs/verl/bin/python" - "$STAGE" "$REPO" "$SOURCE" <<'EOF'
import sys
from huggingface_hub import HfApi
stage, repo, source = sys.argv[1:]
HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=stage, commit_message=f"goal 9 from {source}",
                      allow_patterns=[f"libero_goal/records/{source}.jsonl", "libero_goal/episodes/*.pt"])
EOF
echo "done: $(grep -c . "$STAGE/libero_goal/records/$SOURCE.jsonl") episodes run, $(ls "$STAGE/libero_goal/episodes" | wc -l) kept"
