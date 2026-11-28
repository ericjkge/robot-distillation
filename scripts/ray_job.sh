#!/bin/bash
# One machine of a Ray cluster spread over separate Slurm jobs, for clusters that allow one machine per job. All the
# jobs get one fresh $CLUSTER directory. The teacher and rollout jobs start first and wait; the trainer job, submitted
# to start after them, is the head: it writes its address to $CLUSTER/head, waits for all $NODES machines, runs the step
# ("$@") and writes $CLUSTER/done. The others join that address and leave when the step is done or the head has gone.
#   ROLE=trainer|teacher|rollout CLUSTER=<new dir> NODES=<machines> bash scripts/ray_job.sh [step...]
# Run from the repository with WORK_DIR set.
set -euo pipefail
: "${ROLE:?trainer | teacher | rollout}" "${CLUSTER:?}" "${NODES:?}"
# two jobs can share a machine without seeing each other's GPUs: one NCCL host per job, so that NCCL connects jobs over
# the network instead of GPU to GPU (which fails with "unhandled cuda error")
NCCL_HOSTID=$(hostname)-${SLURM_JOB_ID:?}
export NCCL_HOSTID
PY=$WORK_DIR/envs/verl/bin/python
mkdir -p "$CLUSTER"

if [ "$ROLE" = trainer ]; then
  PORT=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
  RAY_ADDRESS=$(getent ahostsv4 "$(hostname)" | awk '{print $1; exit}'):$PORT
  export PORT RAY_ADDRESS
  setsid bash scripts/ray_node.sh trainer &
  node=$!
  trap 'touch "$CLUSTER/done"; kill -- -$node 2> /dev/null' EXIT
  until "$WORK_DIR/envs/verl/bin/ray" status --address="$RAY_ADDRESS" > /dev/null 2>&1; do sleep 5; done
  echo "$RAY_ADDRESS" > "$CLUSTER/head"
  alive='import ray; ray.init(); print(sum(n["Alive"] for n in ray.nodes()))'
  until [ "$("$PY" -c "$alive" 2> /dev/null)" = "$NODES" ]; do sleep 30; done
  echo "$(date +%H:%M) all $NODES machines joined"
  "$@"
else
  until [ -e "$CLUSTER/head" ]; do sleep 30; done
  setsid bash scripts/ray_node.sh "$ROLE" "$(cat "$CLUSTER/head")" &
  node=$!
  trap 'kill -- -$node 2> /dev/null' EXIT
  # Ray's node exits on its own once the head is gone
  while [ ! -e "$CLUSTER/done" ] && kill -0 $node 2> /dev/null; do sleep 30; done
fi
