#!/bin/bash
# Two copies of the SFT job wait in different partitions; the first to start runs. This cancels the other and points
# OPD's first jobs (afterok on SFT) at the one that runs, then prints it and exits.
#   bash sft/race.sh <SFT job> <other copy> <OPD job (afterok on SFT)>...
set -euo pipefail
first=$1 second=$2
shift 2
state() { squeue -h -j "$1" -o %T 2> /dev/null; }
while true; do
  if [ "$(state "$first")" = RUNNING ]; then
    scancel "$second" 2> /dev/null || true
    echo "$first runs; $second cancelled"
    exit 0
  fi
  if [ "$(state "$second")" = RUNNING ]; then
    scancel "$first"
    for job in "$@"; do scontrol update jobid="$job" Dependency="afterok:$second"; done
    echo "$second runs; $first cancelled; OPD jobs $* now after $second"
    exit 0
  fi
  [ -n "$(state "$first")$(state "$second")" ] || { echo "neither copy is queued"; exit 1; }
  sleep 10
done
