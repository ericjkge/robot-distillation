"""The episodes of a PEAR episode file (LIBERO, LIBERO-Pro, LIBERO-Plus) that an evaluation still has to run: one
shard of them (every n-th), less those already recorded under the run's directory (any val_dumps/*.jsonl below
it), so that a set splits across jobs and an interrupted job resumes rather than reruns. Prints an episode file of
one entry per episode, nothing when every episode is done.
    python -m opd.episodes <episode file> <run dir> [--shard i/n] > episodes.json"""

import argparse
import itertools
import json
from pathlib import Path


def values(value) -> list:
    """An env field's values: a list, a scalar, or PEAR's "a-b,c" ranges."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value[:1].isdigit():
        out = []
        for part in value.split(","):
            first, _, last = part.partition("-")
            out.extend(range(int(first), int(last or first) + 1))
        return out
    return [value]


def expand(env: dict | list) -> list[dict]:
    """Every episode of an episode file's env: the product of each entry's fields."""
    episodes = []
    for entry in env if isinstance(env, list) else [env]:
        episodes += [dict(zip(entry, combo, strict=True)) for combo in itertools.product(*map(values, entry.values()))]
    return episodes


def recorded(run_dir: Path) -> list[dict]:
    """The evaluation records under a run directory; a line cut off by a preemption is skipped."""
    records = []
    for dump in sorted(run_dir.glob("**/val_dumps/*.jsonl")):
        for line in dump.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def remaining(episode_file: dict, run_dir: Path, shard: str = "0/1") -> list[dict]:
    index, count = map(int, shard.split("/"))
    episodes = expand(episode_file["env"])[index::count]
    fields = list(episodes[0])
    done = {tuple(str(r["extra_fields"]["episode"]["env"][f]) for f in fields) for r in recorded(run_dir)}
    return [{f: [e[f]] for f in fields} for e in episodes if tuple(str(e[f]) for f in fields) not in done]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_file", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--shard", default="0/1", help="i/n: the episodes whose index is i modulo n")
    args = parser.parse_args()
    episode_file = json.loads(args.episode_file.read_text())
    if left := remaining(episode_file, args.run_dir, args.shard):
        print(json.dumps({**episode_file, "env": left}))
