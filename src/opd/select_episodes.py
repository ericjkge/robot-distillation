"""The SFT set from the teacher's episodes: up to --per-task successful episodes of every task, drawn at random (all of
a task's, when it has fewer), linked into one directory for scripts/sft.sh.
    python -m opd.select_episodes --per-task 25 --out <dir> <source> [<source> ...]
A source is either
- a collection run (scripts/collect.sh): its val_dumps/*.jsonl name each episode's task (its uid's suite and task id),
  and its episodes/<uid>.pt are the episodes opd.collect kept; or
- a download of the episodes' Hub dataset: its index.jsonl has one line per episode run, with the suite, the task_id
  and the episode's path within the dataset (null for a failure).
An episode in several sources (by its uid) is counted once."""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

Task = tuple[str, int]  # (suite, task id)


def collection_run_episodes(run: Path) -> Iterator[tuple[Task, Path]]:
    """The kept episodes of a collection run, with their tasks."""
    for dump in sorted((run / "val_dumps").glob("*.jsonl")):
        for line in dump.read_text().splitlines():
            record = json.loads(line)
            path = run / "episodes" / f"{record['uid']}.pt"
            if path.exists():  # opd.collect saved it: the episode succeeded
                env = record["extra_fields"]["episode"]["env"]
                yield (env["suite"], int(env["task_id"])), path


def hub_dataset_episodes(dataset: Path) -> Iterator[tuple[Task, Path]]:
    """The successful episodes of a downloaded Hub dataset, with their tasks."""
    for line in (dataset / "index.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["episode"] is None:
            continue
        path = dataset / row["episode"]
        if not path.exists():  # a partial download would silently thin out the set
            raise FileNotFoundError(f"{path} is in {dataset}/index.jsonl but was not downloaded")
        yield (row["suite"], int(row["task_id"])), path


def episodes_by_task(sources: list[Path]) -> dict[Task, list[Path]]:
    """Every successful episode of the sources, under its task, each episode once."""
    tasks = defaultdict(dict)  # task -> {episode file name (its uid): path}
    for source in sources:
        episodes = hub_dataset_episodes if (source / "index.jsonl").exists() else collection_run_episodes
        for task, path in episodes(source):
            tasks[task].setdefault(path.name, path)
    return {task: list(paths.values()) for task, paths in tasks.items()}


def select(tasks: dict[Task, list[Path]], per_task: int, seed: int) -> list[Path]:
    """A seeded draw of up to per_task episodes of each task. It depends on the episodes (their file names, the uids),
    not on where they are, so a Hub download and the collection runs it came from give the same set."""
    rng = random.Random(seed)
    chosen = []
    for task in sorted(tasks):
        paths = sorted(tasks[task], key=lambda path: path.name)
        chosen += rng.sample(paths, min(per_task, len(paths)))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = episodes_by_task(args.sources)
    chosen = select(tasks, args.per_task, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    for path in chosen:
        (args.out / path.name).symlink_to(path.resolve())
    counts = {f"{suite}/{task}": len(paths) for (suite, task), paths in sorted(tasks.items())}
    (args.out / "selection.json").write_text(json.dumps({"per_task": args.per_task, "seed": args.seed, "kept": counts}))
    print(f"{len(chosen)} episodes of {sum(counts.values())} kept, over {len(counts)} tasks, in {args.out}")


if __name__ == "__main__":
    main()
