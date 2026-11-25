"""Lay out finished collection runs for the Hugging Face Hub dataset, in a staging directory of symlinks:
    <stage>/<suite>/<run>/val_dumps/*.jsonl   one record per episode run (success or not)
    <stage>/<suite>/<run>/episodes/<uid>.pt   each successful episode's full multi-round context
    <stage>/index.jsonl                       one line per episode run, over every run in <stage>
    <stage>/README.md                         the dataset card, with the kept/run tally per task
    python hub.py <stage> <collection run> [...]   (a run: runs/collect-<sha>/<suite>/attempt-<job>-<n>[-...])
A run is named o2-<job>-<n>[-...] after its attempt; runs already in <stage> (another cluster's too) stay."""

import json
import sys
from collections import Counter
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

stage = Path(sys.argv[1])
for attempt in map(Path, sys.argv[2:]):
    run = stage / attempt.parent.name / ("o2-" + attempt.name.removeprefix("attempt-"))
    for sub, pattern in [("val_dumps", "*.jsonl"), ("episodes", "*.pt")]:
        (run / sub).mkdir(parents=True, exist_ok=True)
        for f in (attempt / sub).glob(pattern):
            if not (run / sub / f.name).exists():
                (run / sub / f.name).symlink_to(f.resolve())

index, kept, ended = [], Counter(), Counter()
for dump in sorted(stage.glob("*/*/val_dumps/*.jsonl")):
    run = dump.parent.parent
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line cut off when its run was stopped
            continue
        env = record["extra_fields"]["episode"]["env"]
        episode = run / "episodes" / f"{record['uid']}.pt"
        key = env["suite"], int(env["task_id"])
        ended[key] += 1
        kept[key] += episode.exists()
        index.append({
            "suite": env["suite"], "task_id": int(env["task_id"]), "trial_id": int(env["trial_id"]),
            "success": record["score"] >= 1, "num_turns": record["num_turns"], "uid": record["uid"],
            "run": str(run.relative_to(stage)),
            "episode": str(episode.relative_to(stage)) if episode.exists() else None,
        })
(stage / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))

rows = []
for suite in SUITES:
    cells = [f"{kept[suite, t]}/{ended[suite, t]}" if ended[suite, t] else "–" for t in range(10)]
    rows.append(f"| {suite} | " + " | ".join(cells) + f" | {sum(kept[suite, t] for t in range(10))} |")
(stage / "README.md").write_text(f"""---
pretty_name: Qwen3.8-27B-PEAR teacher rollouts on LIBERO
tags: [robotics, libero, distillation, vision-language-action]
---

# Qwen3.8-27B-PEAR teacher rollouts on LIBERO

Episodes of the teacher [Qwen3.8-27B-PEAR-LIBERO_only](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only),
at xhigh reasoning effort, on LIBERO's four suites with [PEAR](https://github.com/mit-han-lab/pear)'s protocol (one
episode per task and initial state, or trial). They are the SFT data of an on-policy distillation project
([robot-distillation](https://github.com/ericjkge/robot-distillation), `scripts/collect.sh`).

**Rule.** Each task ran until it had 25 successful episodes or had run all 50 trials, so tasks the teacher solves less
often have more failed runs and fewer successes. SFT draws up to 25 successful episodes per task.

## Successful episodes / episodes run, per task

| suite | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Layout

```
<suite>/<run>/val_dumps/*.jsonl   one record per episode run: task, trial, success, actions, and the prompt and output text
<suite>/<run>/episodes/<uid>.pt   each successful episode, keyed by its record's uid
index.jsonl                       one line per episode run: suite, task_id, trial_id, success, num_turns, uid, run, episode
```

A run is one collection job: `o2-<job>-<attempt>[-tasks<t>-trials<a-b>]` on Harvard O2; runs from another cluster are
named after it. A trial may appear in two runs (a requeued or relayed job); use one.

Each episode file is a `torch.save` dict of the episode's full multi-round context, as the student sees it at inference:
`input_ids` (L,), `position_ids` (4, L) (Qwen's multimodal RoPE), `loss_mask` (L,) (1 on the teacher's generated tokens:
reasoning and actions), and `multi_modal_inputs` (`pixel_values`, `image_grid_thw`, `images_seqlens`: the camera images
of every round).

```python
import json, torch
from huggingface_hub import snapshot_download
root = snapshot_download("<this repo>", repo_type="dataset")
rows = [json.loads(l) for l in open(f"{{root}}/index.jsonl")]
episode = torch.load(f"{{root}}/{{rows[0]['episode']}}", weights_only=False)
```

The SFT set: `python -m opd.select_episodes --per-task 25 --seed 0 --out <dir> <root>/*/*` (robot-distillation).
""")
print(f"{len(index)} episode runs, {sum(kept.values())} kept, in {stage}")
