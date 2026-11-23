"""Success rate of a baseline from the episode records its jobs have written so far (finished or not): per suite, its
episodes, successes, rate with its Wilson 95% interval, and the tasks covered; then LIBERO's mean over the suites.
    python tally.py <runs>/baseline-<sha>/<student> [...]
reads <suite>/trials-*/val_dumps/*.jsonl under each (the halves of a suite may run different commits). A suite's rate is the mean of its tasks' rates, so that tasks with
more finished episodes so far do not weigh more."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

scores = defaultdict(lambda: defaultdict(list))  # suite -> task -> successes (0/1)
for dump in sorted(d for root in sys.argv[1:] for d in Path(root).glob("*/trials-*/val_dumps/*.jsonl")):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        env = record["extra_fields"]["episode"]["env"]
        scores[env["suite"]][int(env["task_id"])].append(float(record["score"]) >= 1)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval: unlike rate +- 1.96 SE, it is not empty at 0 successes."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


rates = []
print(f"{'suite':16} {'episodes':>8} {'successes':>9} {'rate':>7} {'95% interval':>15}  tasks")
for suite in sorted(scores):
    tasks = scores[suite]
    task_rates = [sum(s) / len(s) for s in tasks.values()]
    rate = sum(task_rates) / len(task_rates)
    n = sum(len(s) for s in tasks.values())
    successes = sum(sum(s) for s in tasks.values())
    low, high = wilson(successes, n)  # over the suite's pooled episodes
    rates.append(rate)
    print(f"{suite:16} {n:8d} {successes:9d} {rate:7.1%} {low:6.1%} - {high:5.1%}  {len(tasks)}/10")
if rates:
    print(f"{'LIBERO (mean)':16} {'':8} {'':9} {sum(rates) / len(rates):7.1%}")
