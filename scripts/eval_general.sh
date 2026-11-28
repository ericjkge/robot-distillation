#!/usr/bin/env bash
# Bonus evaluation: general capability (math, science, coding, instruction following, vision), with PEAR's own recipe
# unchanged (third_party/pear/examples/eval/eval_general.sh and pear/configs/general.yaml): nine EvalScope benchmarks
# with PEAR's sampling, repeats and token budgets, each model served by vLLM with its own chat template. Only the model
# file is ours (configs/models: a student's replica takes one GPU). Install its environment and datasets once, on CPU:
#   WORK_DIR=<work dir> bash third_party/pear/scripts/install.sh evalscope
# Then, with MODEL_FILE=configs/models/<model>.sh MODEL=<model dir> NAME=<served name> RUN=<run dir>:
#   bash scripts/eval_general.sh serve                             # on every GPU machine, left running
#   bash scripts/eval_general.sh run <host>:8000,<host>:8001,...   # on one of them
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_FILE=$(readlink -f "${MODEL_FILE:?configs/models/<model>.sh}")
# PEAR's GPU profiles set only the tensor parallelism here: h100's on any GPU
export MODEL_FILE PEAR=$ROOT/third_party/pear GPU=${GPU:-h100}
exec bash "$PEAR/examples/eval/eval_general.sh" "$@"
