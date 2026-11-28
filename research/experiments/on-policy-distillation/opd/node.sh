#!/bin/bash
# One machine of a Ray cluster spread over separate Slurm jobs (O2 allows one node per job). The trainer job is the
# head: it writes its address to $CLUSTER/head, waits for all $NODES machines, runs the step ("$@") and writes
# $CLUSTER/done. The other jobs join that address and leave when the step is done or the trainer job has gone.
#   ROLE=trainer|teacher|rollout CLUSTER=<dir> NODES=<machines> TRAINER_JOB=<id> bash opd/node.sh [step...]
# Run from the code directory with WORK_DIR set. NCCL across O2's machines works over TCP only (checks/crossnode.py).
set -euo pipefail
: "${ROLE:?trainer | teacher | rollout}" "${CLUSTER:?}" "${NODES:?}" "${TRAINER_JOB:?}"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=^lo,idrac
PY=$WORK_DIR/envs/verl/bin/python

if [ "$ROLE" = trainer ]; then
  mkdir -p "$CLUSTER"  # one directory per trainer job: no earlier address to find
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
  while [ ! -e "$CLUSTER/done" ] && kill -0 $node 2> /dev/null && squeue -h -j "$TRAINER_JOB" | grep -q .; do sleep 30; done
fi
