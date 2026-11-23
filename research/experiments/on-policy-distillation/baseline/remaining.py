"""The episode file of a baseline job's episodes that no earlier attempt of the job finished, so that a requeued job
resumes rather than reruns: one product per task, of the trials in TRIALS not yet in a record under the run.
    python remaining.py <suite episode file> <trials, e.g. 0-24> <run> > episodes.json
Prints nothing when every episode is done."""

import json
import sys
from pathlib import Path

suite_file, trials, run = sys.argv[1], sys.argv[2], Path(sys.argv[3])


def numbers(ranges: str) -> list[int]:
    """The integers of "a-b" ranges and numbers, comma-separated (PEAR's episode-file syntax)."""
    out = []
    for part in ranges.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


episode_file = json.loads(Path(suite_file).read_text())
env = episode_file["env"]
done = set()
for dump in run.glob("*/val_dumps/*.jsonl"):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by the preemption
            continue
        episode = record["extra_fields"]["episode"]["env"]
        done.add((int(episode["task_id"]), int(episode["trial_id"])))

products = []
for task in numbers(env["task_id"]):
    left = [trial for trial in numbers(trials) if (task, trial) not in done]
    if left:
        products.append({"suite": env["suite"], "task_id": [task], "trial_id": left})
if products:
    print(json.dumps({**episode_file, "env": products}, indent=1))
