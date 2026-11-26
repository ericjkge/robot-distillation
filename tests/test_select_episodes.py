"""The SFT set opd.select_episodes draws: an even cap per task, over collection runs and Hub dataset downloads."""

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


def hub_dataset(path, suite, kept, failed=0, first_trial=0, cluster="o2"):
    """A download of the Hub dataset as snapshot_download leaves it: index.jsonl, and each successful episode at its
    `episode` path, a link into the download's blob store (a file named by its hash). Trials start at `first_trial`."""
    (path / suite / "episodes").mkdir(parents=True)
    (path / "blobs").mkdir()
    rows = []
    for task, n in enumerate(kept):
        for trial in range(first_trial, first_trial + n + failed):
            uid = f"{suite}-{task}-{trial}"
            episode = f"{suite}/episodes/{uid}.pt" if trial < first_trial + n else None
            if episode:
                blob = path / "blobs" / f"{cluster}{task:02d}{trial:02d}"
                torch.save({"input_ids": torch.arange(4)}, blob)
                (path / episode).symlink_to(blob)
            rows.append(
                {
                    "suite": suite,
                    "task_id": task,
                    "trial_id": trial,
                    "success": episode is not None,
                    "uid": uid,
                    "source": cluster,
                    "episode": episode,
                }
            )
    (path / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def counts(tasks):
    return {task: len(paths) for task, paths in tasks.items()}


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
    assert counts(tasks) == {("libero_goal", 0): 40, ("libero_goal", 1): 3, ("libero_10", 0): 10}
    assert all(task_of(path) == task for task, paths in tasks.items() for path in paths)


def test_each_task_keeps_its_cap_or_all_it_has(runs):
    chosen = select(episodes_by_task(runs), per_task=25, seed=0)
    assert Counter(map(task_of, chosen)) == {("libero_goal", 0): 25, ("libero_goal", 1): 3, ("libero_10", 0): 10}
    assert len(set(chosen)) == len(chosen)


def test_the_draw_is_fixed_by_its_seed(runs):
    tasks = episodes_by_task(runs)
    assert select(tasks, 25, seed=0) == select(tasks, 25, seed=0)
    assert select(tasks, 25, seed=0) != select(tasks, 25, seed=1)


def test_a_hub_download_gives_its_successes_by_task(tmp_path):
    hub = hub_dataset(tmp_path / "hub", "libero_goal", kept=[30, 0, 2], failed=3)
    tasks = episodes_by_task([hub])
    assert counts(tasks) == {("libero_goal", 0): 30, ("libero_goal", 2): 2}
    assert all(
        task_of(path) == task and path.parent == hub / "libero_goal" / "episodes"
        for task, paths in tasks.items()
        for path in paths
    )


def test_sources_of_both_kinds_pool_a_tasks_episodes_before_the_cap(runs, tmp_path):
    """Another cluster's episodes of the same task add to ours, and the cap applies to the pool."""
    hub = hub_dataset(tmp_path / "hub", "libero_10", kept=[10, 4], first_trial=10, cluster="other")
    tasks = episodes_by_task([*runs, hub])
    assert counts(tasks) == {("libero_goal", 0): 40, ("libero_goal", 1): 3, ("libero_10", 0): 20, ("libero_10", 1): 4}
    chosen = select(tasks, per_task=15, seed=0)
    assert len([path for path in chosen if path in tasks["libero_10", 0]]) == 15
    assert len(chosen) == 15 + 3 + 15 + 4


def test_an_episode_in_two_sources_counts_once(runs, tmp_path):
    """The same episodes, as our collection run and as our upload of it to the Hub."""
    hub = hub_dataset(tmp_path / "hub", "libero_10", kept=[10])
    tasks = episodes_by_task([runs[1], hub])
    assert counts(tasks) == {("libero_10", 0): 10}
    assert all(path.parent == runs[1] / "episodes" for path in tasks["libero_10", 0])  # the first source's copy


def test_the_draw_depends_on_the_episodes_not_their_location(tmp_path):
    """The same 40 episodes, in one download or split across two (as across a requeued job's attempts)."""
    whole = hub_dataset(tmp_path / "whole", "libero_goal", kept=[40])
    second_half = hub_dataset(tmp_path / "a", "libero_goal", kept=[20], first_trial=20)
    first_half = hub_dataset(tmp_path / "b", "libero_goal", kept=[20])
    names = [
        [path.name for path in select(episodes_by_task(sources), 25, seed=0)]
        for sources in ([whole], [first_half, second_half])
    ]
    assert names[0] == names[1]


def test_an_incomplete_download_is_an_error(tmp_path):
    hub = hub_dataset(tmp_path / "hub", "libero_goal", kept=[3])
    (hub / "libero_goal" / "episodes" / "libero_goal-0-1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="libero_goal-0-1.pt"):
        episodes_by_task([hub])


def test_the_set_is_a_directory_the_sft_dataset_reads(runs, tmp_path, monkeypatch):
    hub = hub_dataset(tmp_path / "hub", "libero_object", kept=[30, 5], failed=1)
    out = tmp_path / "set"
    monkeypatch.setattr(sys, "argv", ["select_episodes", "--out", str(out), *map(str, runs), str(hub)])
    select_episodes.main()
    assert len(EpisodeDataset(str(out), None, OmegaConf.create({"max_length": 8}))) == 25 + 3 + 10 + 25 + 5
    assert all(
        link.resolve().parent in (runs[0] / "episodes", runs[1] / "episodes", hub / "blobs")
        for link in out.glob("*.pt")
    )
    kept = json.loads((out / "selection.json").read_text())["kept"]
    assert kept["libero_goal/0"] == 40 and kept["libero_object/1"] == 5
    with pytest.raises(FileExistsError):  # a set is never overwritten
        select_episodes.main()
