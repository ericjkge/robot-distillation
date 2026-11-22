"""The SFT set from one or more collection runs: up to --per-task successful episodes of every task, drawn at random
(all of a task's, when it has fewer), linked into one directory for scripts/sft.sh.
    python -m opd.select_episodes --per-task 25 --out <dir> <collect run> [<collect run> ...]
A collection run's val_dumps/*.jsonl name each episode's task (its uid's suite and task id); its episodes/<uid>.pt are
the episodes opd.collect kept."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def episodes_by_task(runs: list[Path]) -> dict[tuple[str, int], list[Path]]:
    """Every kept episode of the runs, under its task (suite, task id)."""
    tasks = defaultdict(list)
    for run in runs:
        for dump in sorted((run / "val_dumps").glob("*.jsonl")):
            for line in dump.read_text().splitlines():
                record = json.loads(line)
                path = run / "episodes" / f"{record['uid']}.pt"
                if path.exists():
                    env = record["extra_fields"]["episode"]["env"]
                    tasks[env["suite"], int(env["task_id"])].append(path)
    return tasks


def select(tasks: dict[tuple[str, int], list[Path]], per_task: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    chosen = []
    for task in sorted(tasks):
        paths = sorted(tasks[task])
        chosen += rng.sample(paths, min(per_task, len(paths)))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = episodes_by_task(args.runs)
    chosen = select(tasks, args.per_task, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    for path in chosen:
        (args.out / path.name).symlink_to(path.resolve())
    counts = {f"{suite}/{task}": len(paths) for (suite, task), paths in sorted(tasks.items())}
    (args.out / "selection.json").write_text(json.dumps({"per_task": args.per_task, "seed": args.seed, "kept": counts}))
    print(f"{len(chosen)} episodes of {sum(counts.values())} kept, over {len(counts)} tasks, in {args.out}")


if __name__ == "__main__":
    main()
