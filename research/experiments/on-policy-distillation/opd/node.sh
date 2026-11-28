#!/bin/bash
# One machine of a Ray cluster spread over separate Slurm jobs (O2 allows one node per job), all given one fresh
# $CLUSTER directory. The teacher and rollout jobs start first and wait; the trainer job, submitted to start after
# them, is the head: it writes its address to $CLUSTER/head, waits for all $NODES machines, runs the step ("$@") and
# writes $CLUSTER/done. The others join that address and leave when the step is done or the head has gone.
#   ROLE=trainer|teacher|rollout CLUSTER=<new dir> NODES=<machines> bash opd/node.sh [step...]
# Run from the code directory with WORK_DIR set. NCCL across O2's machines works over TCP only (checks/crossnode.py).
set -euo pipefail
: "${ROLE:?trainer | teacher | rollout}" "${CLUSTER:?}" "${NODES:?}"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=^lo,idrac
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
  until [ "$("$PY" -c 'import ray; ray.init(); print(sum(n["Alive"] for n in ray.nodes()))' 2> /dev/null)" = "$NODES" ]; do
    sleep 30
  done
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
