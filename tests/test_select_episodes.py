"""The SFT set opd.select_episodes draws: an even cap per task, over several collection runs."""

import json
import sys
from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf

from opd import select_episodes
from opd.select_episodes import episodes_by_task, select
from opd.sft_data import EpisodeDataset


def collection_run(path, suite, kept, failed=0):
    """A collection run as opd.collect leaves it: `kept[task]` saved episodes of each task, `failed` more per task in
    the evaluation records only."""
    (path / "episodes").mkdir(parents=True)
    (path / "val_dumps").mkdir()
    records = []
    for task, n in enumerate(kept):
        for trial in range(n + failed):
            uid = f"{suite}-{task}-{trial}"
            if trial < n:
                torch.save({"input_ids": torch.arange(4)}, path / "episodes" / f"{uid}.pt")
            env = {"suite": suite, "task_id": task, "trial_id": trial}
            records.append({"uid": uid, "extra_fields": {"episode": {"env": env}}})
    (path / "val_dumps" / "eval_0.jsonl").write_text("\n".join(map(json.dumps, records)))
    return path


def task_of(path):
    suite, task, _ = path.stem.rsplit("-", 2)
    return suite, int(task)


@pytest.fixture
def runs(tmp_path):
    return [
        collection_run(tmp_path / "goal", "libero_goal", kept=[40, 3], failed=2),
        collection_run(tmp_path / "long", "libero_10", kept=[10]),
    ]


def test_episodes_are_grouped_by_task_without_the_failures(runs):
    tasks = episodes_by_task(runs)
    assert {task: len(paths) for task, paths in tasks.items()} == {
        ("libero_goal", 0): 40,
        ("libero_goal", 1): 3,
        ("libero_10", 0): 10,
    }
    assert all(task_of(path) == task for task, paths in tasks.items() for path in paths)


def test_each_task_keeps_its_cap_or_all_it_has(runs):
    chosen = select(episodes_by_task(runs), per_task=25, seed=0)
    assert Counter(map(task_of, chosen)) == {("libero_goal", 0): 25, ("libero_goal", 1): 3, ("libero_10", 0): 10}
    assert len(set(chosen)) == len(chosen)


def test_the_draw_is_fixed_by_its_seed(runs):
    tasks = episodes_by_task(runs)
    assert select(tasks, 25, seed=0) == select(tasks, 25, seed=0)
    assert select(tasks, 25, seed=0) != select(tasks, 25, seed=1)


def test_the_set_is_a_directory_the_sft_dataset_reads(runs, tmp_path, monkeypatch):
    out = tmp_path / "set"
    monkeypatch.setattr(sys, "argv", ["select_episodes", "--per-task", "25", "--out", str(out), *map(str, runs)])
    select_episodes.main()
    assert len(EpisodeDataset(str(out), None, OmegaConf.create({"max_length": 8}))) == 25 + 3 + 10
    assert json.loads((out / "selection.json").read_text())["kept"]["libero_goal/0"] == 40
    with pytest.raises(FileExistsError):  # a set is never overwritten
        select_episodes.main()
