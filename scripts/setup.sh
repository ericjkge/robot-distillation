#!/usr/bin/env bash
# One-time install under WORK_DIR: PEAR's training environment (verl) and the simulators of BENCHES, then the models of
# MODEL_FILES at their pinned revisions. PEAR builds its CUDA extensions for this machine's GPU or, on a machine without
# one, for CUDA_ARCH, which it then needs (a single compute capability; MAX_JOBS bounds the compilers' memory).
# On a cluster that mixes GPU types, set CUDA_ARCH to the lowest one's: a build for an L40S (8.9) fails on an A6000 or
# A100 (cudaErrorNoKernelImageForDevice), while 8.0's kernels run on every 8.x GPU (A100, A6000, A40, L40S). An H100
# (9.0) needs its own build, which adds flash-attn 3.
# On a GPU machine it finishes with PEAR's checks of the training environment and of a LIBERO simulator.
#   WORK_DIR=<a directory every machine sees> [CUDA_ARCH=...] [BENCHES="libero libero_plus libero_pro"] \
#     [MODEL_FILES="configs/models/teacher.sh ..."] bash scripts/setup.sh
source "$(dirname "$0")/common.sh"

if ! nvidia-smi -L > /dev/null 2>&1; then
  : "${CUDA_ARCH:?no GPU here: set CUDA_ARCH to the compute capabilities to build for}"
fi
read -r -a BENCH_LIST <<< "${BENCHES:-libero libero_plus libero_pro}"
read -r -a MODEL_LIST <<< "${MODEL_FILES:-$(echo "$ROOT"/configs/models/*.sh)}"

git -C "$ROOT" submodule update --init
# PEAR's targets, each an environment of WORK_DIR/envs (the simulators bring Mesa, LIBERO-Plus ImageMagick)
for target in "${BENCH_LIST[@]}" numactl verl; do
  # PEAR's own check of its verl checkout sees this repository's patches as a change; scripts/verl.sh checks both
  VERL_CHANGE_IN_PROGRESS=1 bash "$PEAR/scripts/install.sh" "$target"
done
bash "$ROOT/scripts/verl.sh"  # this repository's verl patches (patches/verl) on PEAR's checkout
for model_file in "${MODEL_LIST[@]}"; do
  MODEL_FILE=$model_file bash "$PEAR/scripts/download_model.sh"
done

# PEAR's own checks (its libero_run target), on a GPU machine: the training environment on its GPU, then one LIBERO step
nvidia-smi -L > /dev/null 2>&1 || exit 0
(
  export LOCAL_DIR=$WORK_DIR/cache/environment_test
  # shellcheck source=/dev/null
  source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"
  cd "$PEAR"
  "$VERL_ENV/bin/python" -m pytest -q tests/test_training_environment_on_gpu.py
  LIBERO_PYTHON=$WORK_DIR/envs/libero/bin/python LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero/etc/libero \
    LIBERO_MESA_DIR=$WORK_DIR/envs/mesa "$VERL_ENV/bin/python" -m pear.sim.checker \
    --tool-config pear/envs/libero/configs/tool_config.yaml \
    --env-args '{"suite": "libero_spatial", "task_id": 0, "trial_id": 0}' \
    --action '{"arm": {"x": 2.0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": 0}}' \
    --log-dir "$WORK_DIR/runs/setup_check"
)
