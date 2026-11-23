#!/usr/bin/env bash
# Starts this machine's Ray node and blocks until Ray stops.
#   bash scripts/ray_node.sh <role> [head address]     # no address: this machine is the head
# Roles: trainer (the actor), rollout (the policy's vLLM replicas and the LIBERO simulators), teacher (the distillation
# teacher's servers), or all: one machine for every role, its GPUs split by TRAINER_GPUS and ROLLOUT_GPUS with the rest
# left to the teacher.
source "$(dirname "$0")/common.sh"

ROLE=${1:?trainer | rollout | teacher | all}
HEAD_ADDRESS=${2:-}
GPUS=$(nvidia-smi -L | wc -l)
CPUS=${NUM_CPUS:-$(nproc)}  # before PEAR's variables: nproc counts OMP_NUM_THREADS, which they set
LIBERO_SERVERS=${LIBERO_SERVERS:-16}  # simulator processes per rollout machine, rendering on its CPUs

# PEAR places the actor on the "trainer" resource, the replicas on "rollout" and the simulators on "libero"; the
# teacher's pool takes the GPUs no role claims
case $ROLE in
  trainer) RESOURCES="{\"trainer\": $GPUS}" ;;
  rollout) RESOURCES="{\"rollout\": $GPUS, \"libero\": $LIBERO_SERVERS}" ;;
  teacher) RESOURCES="{}" ;;
  all) RESOURCES="{\"trainer\": ${TRAINER_GPUS:-0}, \"rollout\": ${ROLLOUT_GPUS:-$GPUS}, \"libero\": $LIBERO_SERVERS}" ;;
  *) echo "unknown role $ROLE" >&2; exit 1 ;;
esac

if [ -n "$HEAD_ADDRESS" ]; then
  NODE_ARGS=(--address="$HEAD_ADDRESS")
  PORT=${HEAD_ADDRESS##*:}
else
  PORT=${PORT:-6379}
  NODE_ARGS=(--head --port="$PORT")
fi

mkdir -p "$LOCAL_DIR/ray/$PORT"  # by the cluster's port: another cluster on this machine has its own
# shellcheck source=/dev/null
source "$PEAR/scripts/gpu_process_variables.sh" "$VERL_ENV"
# worker ports from the OS (0) rather than Ray's fixed 10002-19999, which another cluster on this machine also draws from
RAY_TMPDIR=$LOCAL_DIR/ray/$PORT PYTHONPATH=$PYTHON_PATH exec "$VERL_ENV/bin/ray" start "${NODE_ARGS[@]}" \
  --num-gpus="$GPUS" --num-cpus="$CPUS" --resources="$RESOURCES" --min-worker-port=0 --max-worker-port=0 --block
