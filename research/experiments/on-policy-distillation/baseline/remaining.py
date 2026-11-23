"""The episode file of a job's episodes that no earlier attempt of the job finished, so that a requeued job resumes
rather than reruns: one product per task, of the trials in TRIALS not yet in a record under the job's directory (in
its attempts' val_dumps/). With --keep, a task that already has that many episodes kept (a collection's
episodes/<uid>.pt) is left out.
    python remaining.py <suite episode file> <trials, e.g. 0-24> <job dir> [--keep 25] > episodes.json
Prints nothing when every episode is done."""

import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("suite_file", type=Path)
parser.add_argument("trials")
parser.add_argument("job_dir", type=Path)
parser.add_argument("--keep", type=int)
args = parser.parse_args()


def numbers(ranges: str) -> list[int]:
    """The integers of "a-b" ranges and numbers, comma-separated (PEAR's episode-file syntax)."""
    out = []
    for part in ranges.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


episode_file = json.loads(args.suite_file.read_text())
env = episode_file["env"]
done, kept = set(), Counter()
for dump in args.job_dir.glob("**/val_dumps/*.jsonl"):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by the preemption
            continue
        episode = record["extra_fields"]["episode"]["env"]
        done.add((int(episode["task_id"]), int(episode["trial_id"])))
        kept[int(episode["task_id"])] += (dump.parent.parent / "episodes" / f"{record['uid']}.pt").exists()

products = []
for task in numbers(env["task_id"]):
    if args.keep is not None and kept[task] >= args.keep:
        continue
    left = [trial for trial in numbers(args.trials) if (task, trial) not in done]
    if left:
        products.append({"suite": env["suite"], "task_id": [task], "trial_id": left})
if products:
    print(json.dumps({**episode_file, "env": products}, indent=1))
