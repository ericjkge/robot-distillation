#!/usr/bin/env bash
# One-time install, on a GPU machine: PEAR's builds take the machine's GPU architecture, or CUDA_ARCH's (e.g. "8.0;8.9"
# for a cluster of A100s and L40S; MAX_JOBS bounds the compilers' memory). Installs under WORK_DIR
# PEAR's training environment (verl) and the simulators of BENCHES, downloads the models of MODEL_FILES at their pinned
# revisions, then runs PEAR's checks of the training environment on the GPU and of a LIBERO simulator.
#   WORK_DIR=<a directory every machine sees> [BENCHES="libero libero_plus libero_pro"] \
#     [MODEL_FILES="configs/models/teacher.sh ..."] bash scripts/setup.sh
source "$(dirname "$0")/common.sh"

read -r -a BENCH_LIST <<< "${BENCHES:-libero libero_plus libero_pro}"
read -r -a MODEL_LIST <<< "${MODEL_FILES:-$(echo "$ROOT"/configs/models/*.sh)}"

git -C "$ROOT" submodule update --init
# PEAR's targets, each an environment of WORK_DIR/envs (the simulators bring Mesa, LIBERO-Plus ImageMagick)
for target in "${BENCH_LIST[@]}" numactl verl; do
  bash "$PEAR/scripts/install.sh" "$target"
done
for model_file in "${MODEL_LIST[@]}"; do
  MODEL_FILE=$model_file bash "$PEAR/scripts/download_model.sh"
done

# PEAR's own checks (its libero_run target): the training environment on this GPU, then one LIBERO step
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
