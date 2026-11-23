"""Whether a suite's collection has what SFT will draw from it: exits 0 once each of its TASKS (all ten, or a range such as 0-4) has KEEP
saved episodes or has ended all its TRIALS, 1 otherwise, and prints the tally.
    python enough.py <suite dir> [--tasks 0-9] [--trials 50] [--keep 25]
counting the records under the suite's directory, in each of its attempts' val_dumps/ (a requeue starts another).
An episode counts once its record is in val_dumps/, which opd.collect writes after the episode's .pt, so a run stopped
here leaves no partial episode that opd.select_episodes would take."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("suite_dir", type=Path)
parser.add_argument("--tasks", default="0-9")
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--keep", type=int, default=25)
args = parser.parse_args()
first, _, last = args.tasks.partition("-")
tasks = range(int(first), int(last or first) + 1)

ended, kept = Counter(), Counter()
for dump in args.suite_dir.glob("**/val_dumps/*.jsonl"):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        task = int(record["extra_fields"]["episode"]["env"]["task_id"])
        ended[task] += 1
        kept[task] += (dump.parent.parent / "episodes" / f"{record['uid']}.pt").exists()

done = all(kept[t] >= args.keep or ended[t] >= args.trials for t in tasks)
print(" ".join(f"{t}:{kept[t]}/{ended[t]}" for t in tasks), "done" if done else "")
sys.exit(0 if done else 1)
