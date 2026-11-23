#!/bin/bash
# after upload of pod/ to /workspace and dflash2_main.tgz: setup, server, X1 scoring. Resumable (score_effort skips done rounds).
cd /workspace; [ -f logs/setup.done ] || { bash setup.sh > logs/setup.log 2>&1; grep -q DOWNLOAD_DONE logs/download.log && touch logs/setup.done; }
mkdir -p results; [ -d results/dflash2_main ] || tar xzf dflash2_main.tgz -C results --no-same-owner
grep -h "effort" models/pear/chat_template.jinja | head -5 > logs/effort_template.txt
. /workspace/lib.sh; start_server || exit 1
( . /workspace/simenv.sh; sim/bin/python score_effort.py --out results/effort.jsonl > logs/effort.log 2>&1 )
stop_server; echo GO_DONE >> logs/effort.log
