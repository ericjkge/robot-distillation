"""Whether the teacher rollouts are complete: every task of the four suites has 25 successful episodes or has run all
50 trials, over every cluster's episode runs in a Hub stage's index.jsonl (collect/hub.py). Exits 1 and names the
tasks still short otherwise.
    python complete.py <stage>/index.jsonl"""

import json
import sys
from collections import defaultdict

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

kept, trials = defaultdict(int), defaultdict(set)
for line in open(sys.argv[1]):
    row = json.loads(line)
    task = row["suite"], row["task_id"]
    kept[task] += row["success"]
    trials[task].add(row["trial_id"])  # a trial can appear twice (a requeued job)
short = [f"{s}/{t} {kept[s, t]}/{len(trials[s, t])}" for s in SUITES for t in range(10)
         if kept[s, t] < 25 and len(trials[s, t]) < 50]
if short:
    sys.exit(f"{len(short)} tasks short (kept/trials): {', '.join(short)}")
