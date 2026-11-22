#!/usr/bin/env bash
# Step 2: SFT of a student on the collected episodes, on the GPUs of this machine (no Ray).
#   MODEL=<student dir> EPISODES_DIR=<the set opd.select_episodes drew> RUN=<run dir> [NGPUS=...] bash scripts/sft.sh [overrides]
source "$(dirname "$0")/common.sh"

MODEL=${MODEL:?the student model directory}
EPISODES_DIR=${EPISODES_DIR:?a directory of episodes: the set opd.select_episodes drew}
RUN=$(readlink -m "${RUN:?the run directory}")
mkdir -p "$RUN"

NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}
# an episode at xhigh effort runs to 164k tokens: Ulysses sequence parallelism splits each across the GPUs
MAX_LENGTH=163840  # configs/step/sft.yaml's data.max_length

# shellcheck source=/dev/null
source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"
export PYTHONPATH=$PYTHON_PATH
ARGS=(
  +step=sft
  data.train_files="$EPISODES_DIR"
  model.path="$MODEL"
  trainer.default_local_dir="$RUN/ckpt"
  trainer.experiment_name="$(basename "$RUN")"
  hydra.run.dir="$RUN/hydra"
  engine.ulysses_sequence_parallel_size="$NGPUS"
  data.max_token_len_per_gpu=$(((MAX_LENGTH + NGPUS - 1) / NGPUS))
  "$@"
)
merge_args ARGS
torchrun --standalone --nproc_per_node="$NGPUS" -m opd.sft --config-dir "$ROOT/configs" "${ARGS[@]}" 2>&1 | tee -a "$RUN/train.log"
