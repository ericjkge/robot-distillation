"""Whether a suite's collection run has what SFT will draw from it: exits 0 once each of its TASKS tasks has KEEP
saved episodes or has ended all its TRIALS, 1 otherwise, and prints the tally.
    python enough.py <run> [--tasks 10] [--trials 50] [--keep 25]
An episode counts once its record is in val_dumps/, which opd.collect writes after the episode's .pt, so a run stopped
here leaves no partial episode that opd.select_episodes would take."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--tasks", type=int, default=10)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--keep", type=int, default=25)
args = parser.parse_args()

ended, kept = Counter(), Counter()
for dump in (args.run / "val_dumps").glob("*.jsonl"):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        task = int(record["extra_fields"]["episode"]["env"]["task_id"])
        ended[task] += 1
        kept[task] += (args.run / "episodes" / f"{record['uid']}.pt").exists()

done = all(kept[t] >= args.keep or ended[t] >= args.trials for t in range(args.tasks))
print(" ".join(f"{t}:{kept[t]}/{ended[t]}" for t in range(args.tasks)), "done" if done else "")
sys.exit(0 if done else 1)
