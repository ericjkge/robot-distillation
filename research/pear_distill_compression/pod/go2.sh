#!/bin/bash
# X2 after X1: same server, effort-frontier sampling.
cd /workspace; until grep -q GO_DONE logs/effort.log 2>/dev/null; do sleep 30; done
. /workspace/lib.sh; start_server || exit 1
( . /workspace/simenv.sh; sim/bin/python sample_effort.py --k 4 --per-ep 4 --threads 6 --efforts low,medium,xhigh > logs/frontier.log 2>&1 )
stop_server; echo GO2_DONE >> logs/frontier.log
