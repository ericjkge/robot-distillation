"""Step 4's report: per-episode success, time and tokens of evaluated models, their success rates as PEAR reports them,
and the main plots.

    python -m opd.report summary <run dir> [--vs <run dir>] [--tokenizer <model dir>]
        writes <run dir>/episodes.jsonl (one row per episode) and prints one JSON line: the success rate (PEAR's mean
        over the episodes) with its binomial SE, per suite, perturbation or category, and the mean time and tokens per
        episode; with --vs, the difference in success rate from another run in SEs.
    python -m opd.report plot <out dir> <run dir> [<run dir> ...]
        success rate against mean time and against mean reasoning tokens per episode, one point per run, and each run's
        distributions of both, successes and failures apart (matplotlib).

A run directory is everything one model's evaluation wrote, in any layout below it (shards, requeued attempts): the
records (val_dumps/), train.log and sim/turns_*.jsonl. Time per episode runs from the model's first turn to the
episode's end (train.log's EPISODE_END), plus one mean turn for the first turn's generation, which no log marks the
start of. Tokens are the episode's generated text (every turn's reasoning and answer); reasoning tokens those inside
<think> ... </think>."""

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from opd.episodes import recorded

END = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}):EPISODE_END (\{.*\})")
THINK = re.compile(r"(?:^|<think>)(.*?)</think>", re.S)


def episodes(run_dir: Path, count_tokens) -> list[dict]:
    """One row per episode: its env, success, end reason, seconds, tokens and reasoning tokens."""
    ends, turns = {}, defaultdict(list)
    for log in run_dir.glob("**/train.log"):
        for match in END.finditer(log.read_text(errors="replace")):
            end = ast.literal_eval(match[2])
            ends[end["uid"]] = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").timestamp(), end["end_reason"]
    for log in run_dir.glob("**/sim/turns_*.jsonl"):
        for turn in map(json.loads, log.read_text().splitlines()):
            turns[turn["uid"]].append(turn["time"])
    rows = {}
    for record in recorded(run_dir):
        episode = record["extra_fields"]["episode"]
        end, end_reason = ends.get(record["uid"], (None, None))
        first = min(turns.get(episode["uid"], [math.inf]))
        n = len(turns.get(episode["uid"], []))
        rows[json.dumps(episode["env"], sort_keys=True)] = {  # an episode recorded twice counts once
            "env": episode["env"],
            "success": bool(episode["success"]),
            "end_reason": end_reason,
            "seconds": (end - first) * n / max(n - 1, 1) if end and n else None,
            "tokens": count_tokens(record["output"]),
            "reasoning_tokens": count_tokens("\n".join(THINK.findall(record["output"]))),
        }
    return list(rows.values())


def rate(rows: list[dict]) -> tuple[float, float]:
    """PEAR's success rate (pear/eval/embodied.py: the mean over the episodes) and its binomial SE."""
    p = sum(row["success"] for row in rows) / len(rows)
    return p, math.sqrt(p * (1 - p) / len(rows))


def mean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else None


def summary(name: str, rows: list[dict], reference: list[dict] | None = None) -> dict:
    groups = defaultdict(list)  # suite; LIBERO-Pro's perturbation, LIBERO-Plus's category
    for row in rows:
        for field, value in row["env"].items():
            if field not in ("task_id", "trial_id"):
                groups[f"{field}={value}"].append(row)
    p, se = rate(rows)
    out = {
        "run": name,
        "episodes": len(rows),
        "success_rate": round(p, 4),
        "se": round(se, 4),
        "by_group": {group: round(rate(members)[0], 4) for group, members in sorted(groups.items())},
        **{f"mean_{field}": mean(rows, field) for field in ("seconds", "tokens", "reasoning_tokens")},
        "end_reasons": dict(Counter(row["end_reason"] for row in rows)),
    }
    if reference:
        q, se_q = rate(reference)
        out["vs_reference_in_se"] = round((p - q) / math.hypot(se, se_q), 2) if se or se_q else None
    return out


def plot(out: Path, runs: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for field, label in (("seconds", "time per episode (s)"), ("reasoning_tokens", "reasoning tokens per episode")):
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, rows in runs.items():
            p, se = rate(rows)
            ax.errorbar(mean(rows, field), p, yerr=se, fmt="o", capsize=3)
            ax.annotate(name, (mean(rows, field), p), textcoords="offset points", xytext=(5, 5))
        ax.set(xlabel=f"mean {label}", ylabel="success rate")
        fig.tight_layout()
        fig.savefig(out / f"success_vs_{field}.png", dpi=150)

        fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(runs)), 4))
        for i, rows in enumerate(runs.values()):
            for offset, success, color in ((-0.2, True, "tab:green"), (0.2, False, "tab:red")):
                values = [r[field] for r in rows if r["success"] == success and r[field] is not None]
                if values:
                    for body in ax.violinplot(values, positions=[i + offset], widths=0.35, showmedians=True)["bodies"]:
                        body.set_facecolor(color)
        ax.set_xticks(range(len(runs)), list(runs))
        ax.set(ylabel=label, title="successes (green), failures (red)")
        fig.tight_layout()
        fig.savefig(out / f"distribution_{field}.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("run_dir", type=Path)
    summary_parser.add_argument("--vs", type=Path)
    # a model directory: the teacher and the students share a tokenizer
    summary_parser.add_argument("--tokenizer", required=True)
    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("out", type=Path)
    plot_parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    if args.command == "summary":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda text: len(tokenizer.encode(text))  # noqa: E731
        rows = episodes(args.run_dir, count)
        (args.run_dir / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reference = episodes(args.vs, count) if args.vs else None
        print(json.dumps(summary(args.run_dir.name, rows, reference)))
    else:
        runs = {d.name: list(map(json.loads, (d / "episodes.jsonl").read_text().splitlines())) for d in args.run_dirs}
        plot(args.out, runs)
