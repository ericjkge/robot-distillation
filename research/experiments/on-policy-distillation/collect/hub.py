"""Lay out collection runs for the Hugging Face Hub dataset, in a staging directory:
    <stage>/<suite>/records/<source>.jsonl   one record per episode run (success or not), from that source's runs
    <stage>/<suite>/episodes/<uid>.pt        each successful episode's full multi-round context (a symlink)
    <stage>/index.jsonl                      one line per episode run, over every source in <stage>
    <stage>/README.md                        the dataset card, with the success/run tally per task
    python hub.py <stage> <source, e.g. o2> <collection run> [...]   (a run: runs/collect-<sha>/<suite>/attempt-*)
Rerunning with a source's runs replaces its records; another source's records (another cluster's) stay."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

stage, source, runs = Path(sys.argv[1]), sys.argv[2], map(Path, sys.argv[3:])
records = defaultdict(list)
for run in runs:
    for dump in sorted(run.glob("val_dumps/*.jsonl")):
        for line in dump.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # a line cut off when its run was stopped, or still being written
                continue
            suite = record["extra_fields"]["episode"]["env"]["suite"]
            records[suite].append(line + "\n")
            episode = run / "episodes" / f"{record['uid']}.pt"
            link = stage / suite / "episodes" / episode.name
            if episode.exists() and not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(episode.resolve())
for suite, lines in records.items():
    (stage / suite / "records").mkdir(parents=True, exist_ok=True)
    (stage / suite / "records" / f"{source}.jsonl").write_text("".join(lines))

index, kept, ended, trials = [], Counter(), Counter(), defaultdict(set)
for path in sorted(stage.glob("*/records/*.jsonl")):
    for line in path.read_text().splitlines():
        record = json.loads(line)
        env = record["extra_fields"]["episode"]["env"]
        episode = f"{env['suite']}/episodes/{record['uid']}.pt"
        success = record["score"] >= 1  # opd.collect saved its episode
        ended[env["suite"], int(env["task_id"])] += 1
        kept[env["suite"], int(env["task_id"])] += success
        trials[env["suite"], int(env["task_id"])].add(int(env["trial_id"]))
        index.append({"suite": env["suite"], "task_id": int(env["task_id"]), "trial_id": int(env["trial_id"]),
                      "success": success, "num_turns": record["num_turns"], "uid": record["uid"],
                      "source": path.stem, "episode": episode if success else None})
(stage / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))

complete = sum(kept[suite, t] >= 25 or len(trials[suite, t]) >= 50 for suite in SUITES for t in range(10))
status = "all 40 complete" if complete == 40 else f"{complete} of 40 complete"
names = [f"`{name}`" for name in sorted({row["source"] for row in index})]
sources = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
rows = []
for suite in SUITES:
    cells = [f"{kept[suite, t]}/{ended[suite, t]}" if ended[suite, t] else "–" for t in range(10)]
    rows.append(f"| {suite} | " + " | ".join(cells) + f" | {sum(kept[suite, t] for t in range(10))} |")
(stage / "README.md").write_text(f"""---
pretty_name: Qwen3.8-27B-PEAR teacher rollouts on LIBERO
tags: [robotics, libero, distillation]
---

# Qwen3.8-27B-PEAR teacher rollouts on LIBERO

Episodes of [Qwen3.8-27B-PEAR-LIBERO_only](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only) on
LIBERO's four suites at xhigh reasoning effort, run with [PEAR](https://github.com/mit-han-lab/pear)'s protocol. They
are the SFT data of [robot-distillation](https://github.com/ericjkge/robot-distillation).

Each task runs until it has 25 successes or has tried all 50 initial states. Tasks: {status}.

## Files

| Path | Contents |
| --- | --- |
| `index.jsonl` | One row per episode run: `suite`, `task_id`, `trial_id` (initial state), `success`, `num_turns`, `uid`, `source` (who collected it: {sources}), `episode` (path, if successful) |
| `<suite>/records/<source>.jsonl` | The full record of each run: prompts, outputs, actions, score |
| `<suite>/episodes/<uid>.pt` | Each successful episode, as the student sees it at inference |

An episode is a `torch.save` dict: `input_ids` (L), `position_ids` (4, L), `loss_mask` (L; 1 on the teacher's
generated tokens) and `multi_modal_inputs` (`pixel_values`, `image_grid_thw`, `images_seqlens`). A trial can appear
twice when a job was requeued mid-episode.

## Successes / runs per task

| Suite | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Loading

```python
import json, torch
from huggingface_hub import snapshot_download

root = snapshot_download("eddykang06/qwen3.8-27b-pear-libero-rollouts", repo_type="dataset")
rows = [json.loads(line) for line in open(f"{{root}}/index.jsonl")]
episode = torch.load(f"{{root}}/{{next(r for r in rows if r['success'])['episode']}}", weights_only=False)
```
""")
print(f"{len(index)} episode runs, {sum(kept.values())} successful, in {stage}")
