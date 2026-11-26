"""Publishes every SFT checkpoint to the Hub while SFT trains, the latest always on the Hub: a checkpoint's
huggingface/ directory, once it stops changing, is hard-linked aside (the trainer keeps only its latest 2; the link is removed once uploaded) and uploaded
over the model repo's main through publish.py (card, collection): the repo holds only the latest checkpoint. Exits once the SFT job has
left the queue and every checkpoint is published.
    python live.py <ckpt dir>[,<ckpt dir> ...] <repo> <collection slug> <sft set dir> <job id>"""

import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ckpts, repo, collection, sft_set, job = [Path(c) for c in sys.argv[1].split(",")], sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
api = HfApi()


def snapshot(d: Path) -> tuple:
    return tuple(sorted((f.name, f.stat().st_size, f.stat().st_mtime) for f in d.iterdir()))


def running() -> bool:
    return bool(subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True).stdout.strip())


while True:
    alive = running()
    for ckpt, hf in [(c, h) for c in ckpts for h in sorted(c.glob("global_step_*/huggingface"), key=lambda p: int(p.parent.name.split("_")[-1]))]:
        aside = ckpt.parent / "published"
        step = hf.parent.name
        if (aside / step / ".done").exists() or not (hf / "config.json").exists():
            continue
        before = snapshot(hf)
        time.sleep(int(os.environ.get("SETTLE", 180)))
        if not hf.exists() or snapshot(hf) != before:
            continue  # still being written (or already removed)
        (aside / step).mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-al", f"{hf}/.", str(aside / step / "huggingface")], check=True)
        episodes = str(len(list(sft_set.glob("*.pt"))))
        subprocess.run([sys.executable, str(Path(__file__).parent / "publish.py"), str(aside / step / "huggingface"),
                        repo, collection, episodes], check=True)
        api.super_squash_history(repo_id=repo, commit_message=f"SFT checkpoint {step}")  # drop earlier checkpoints' history
        (aside / step / ".done").touch()
        subprocess.run(["rm", "-rf", str(aside / step / "huggingface")], check=True)  # on the Hub now; frees scratch
        print(f"{time.strftime('%H:%M')} {step} -> {repo}", flush=True)
    if not alive:
        break
    time.sleep(int(os.environ.get("POLL", 300)))
