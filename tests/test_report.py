"""Step 4's bookkeeping: the episodes an evaluation still has to run (opd.episodes) and its report (opd.report)."""

import json
from datetime import datetime

from conftest import PEAR

from opd.episodes import expand, remaining
from opd.report import episodes, plot, summary

ENVS = PEAR / "pear" / "envs"


def test_expand_counts_pears_sets():
    sizes = {
        "libero/configs/libero_all.json": 2000,
        "libero_pro/configs/all.json": 10000,
        "libero_plus/configs/all.json": 10030,
    }
    for path, size in sizes.items():
        assert len(expand(json.loads((ENVS / path).read_text())["env"])) == size


def run_dir(path, results, times=(100.0, 110.0, 120.0)):
    """An evaluation run as PEAR leaves it: records, EPISODE_END lines and turn times, one episode per result."""
    (path / "val_dumps").mkdir(parents=True)
    (path / "sim").mkdir()
    records, ends, turns = [], [], []
    for trial, success in enumerate(results):
        env = {"suite": "libero_goal", "task_id": 0, "trial_id": trial}
        episode = {"env": env, "uid": f"e{trial}", "success": success}
        output = "plan a b</think>```json{}```<think>check c</think>```json{}```"
        records.append({"uid": f"r{trial}", "output": output, "extra_fields": {"episode": episode}})
        ends.append(f"WARNING:1970-01-01 00:02:10,000:EPISODE_END {{'uid': 'r{trial}', 'end_reason': 'env'}}")
        turns += [{"uid": f"e{trial}", "time": t} for t in times]
    (path / "val_dumps" / "eval_0.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n{cut off")
    (path / "train.log").write_text("\n".join(ends))
    (path / "sim" / "turns_1.jsonl").write_text("\n".join(map(json.dumps, turns)))
    return path


def test_remaining_shards_and_resumes(tmp_path):
    episode_file = {"env": {"suite": ["libero_goal"], "task_id": "0", "trial_id": "0-3"}}
    run = run_dir(tmp_path / "run", [True, False])  # trials 0 and 1 recorded
    assert [e["trial_id"] for e in remaining(episode_file, run)] == [[2], [3]]
    assert [e["trial_id"] for e in remaining(episode_file, run, "1/2")] == [[3]]


def test_report(tmp_path):
    rows = episodes(run_dir(tmp_path / "a", [True, False, True, True]), count_tokens=lambda text: len(text.split()))
    row = rows[0]
    # from the first turn (100 s) to the end (the log's local time), plus one mean turn for the first generation
    end = datetime(1970, 1, 1, 0, 2, 10).timestamp()
    assert row["seconds"] == (end - 100) * 3 / 2
    assert row["reasoning_tokens"] == 5 and row["end_reason"] == "env"
    result = summary("a", rows, reference=episodes(run_dir(tmp_path / "b", [False] * 4), lambda text: 1))
    assert result["success_rate"] == 0.75 and result["by_group"] == {"suite=libero_goal": 0.75}
    assert result["vs_reference_in_se"] > 2
    plot(tmp_path / "plots", {"a": rows})
    assert len(list((tmp_path / "plots").glob("*.png"))) == 4
