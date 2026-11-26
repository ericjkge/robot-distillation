"""Stage 2's SFT set (sft/staged.sh), balanced so every episode is trained on 3 epochs in all: stage 1's set kept
(its 2 epochs there, 1 here) and each task topped up to 25 with episodes that reached the Hub later, each linked 3 times
(3 epochs here). One epoch over the result. Seeded draw of the new episodes, as opd.select_episodes draws.
    python final_set.py <Hub stage, with index.jsonl> <stage 1 set> <out> [--per-task 25] [--epochs 3] [--seed 0]"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=Path)
parser.add_argument("first", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--per-task", type=int, default=25)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

tasks = defaultdict(dict)  # task -> {uid file name: path}
for line in (args.stage / "index.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["episode"]:
        path = args.stage / row["episode"]
        if not path.exists():
            raise FileNotFoundError(f"{path} is in the index but not in the stage")
        tasks[row["suite"], int(row["task_id"])].setdefault(path.name, path)
first = {p.name for p in args.first.glob("*.pt")}
rng = random.Random(args.seed)
args.out.mkdir(parents=True, exist_ok=False)
old = new = 0
for task in sorted(tasks):
    had = [n for n in tasks[task] if n in first]
    pool = sorted(n for n in tasks[task] if n not in first)
    added = rng.sample(pool, max(0, min(args.per_task - len(had), len(pool))))
    for name in had:
        (args.out / name).symlink_to((args.first / name).resolve())
    for name in added:
        for copy in range(args.epochs):
            (args.out / f"{name[:-3]}-{copy}.pt").symlink_to(tasks[task][name].resolve())
    old, new = old + len(had), new + len(added)
missing = first - {n for t in tasks.values() for n in t}
assert not missing, f"stage 1 episodes no longer on the Hub: {sorted(missing)[:5]}"
print(f"{old} stage 1 episodes once, {new} new episodes x{args.epochs}: {old + new * args.epochs} files in {args.out}")
