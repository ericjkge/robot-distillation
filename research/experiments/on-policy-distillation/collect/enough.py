"""Whether a suite's collection has what SFT will draw from it: exits 0 once each of its TASKS (all ten, or a range such as 0-4) has KEEP
saved episodes or has ended all its TRIALS, 1 otherwise, and prints the tally.
    python enough.py <suite dir> [--tasks 0-9] [--trials 50] [--keep 25]
counting the records under the suite's directory, in each of its attempts' val_dumps/ (a requeue starts another).
An episode counts once its record is in val_dumps/, which opd.collect writes after the episode's .pt, so a run stopped
here leaves no partial episode that opd.select_episodes would take.

A job named on the first line of <suite dir>/../relay that finds its suite done hands its GPUs over rather than ending:
it stops its collection run (the per-suite job's process group, as the job itself would) and works through the units
on the file's second line (collect.sbatch's UNITS) before it exits."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("suite_dir", type=Path)
parser.add_argument("--tasks", default="0-9")
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--keep", type=int, default=25)
args = parser.parse_args()
first, _, last = args.tasks.partition("-")
tasks = range(int(first), int(last or first) + 1)

ended, kept = Counter(), Counter()
for dump in args.suite_dir.glob("**/val_dumps/*.jsonl"):
    for line in dump.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        task = int(record["extra_fields"]["episode"]["env"]["task_id"])
        ended[task] += 1
        kept[task] += (dump.parent.parent / "episodes" / f"{record['uid']}.pt").exists()

done = all(kept[t] >= args.keep or ended[t] >= args.trials for t in tasks)
print(" ".join(f"{t}:{kept[t]}/{ended[t]}" for t in tasks), "done" if done else "")


def processes() -> dict[int, tuple[str, int, int]]:
    """Each process's state, parent and session, from /proc."""
    out = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, parent, _, session = stat.read_text().rsplit(") ", 1)[1].split()[:4]
        except (OSError, ValueError):  # ended while read
            continue
        out[int(stat.parent.name)] = state, int(parent), int(session)
    return out


def relay(units: str) -> None:
    """Stop the calling job's collection run, the setsid group the job's shell started (as the job itself would stop
    it), then run the units in the job."""
    leaders = [pid for pid, (_, parent, session) in processes().items() if parent == os.getppid() and session == pid]
    for leader in leaders:
        os.killpg(leader, signal.SIGTERM)
    # until every process of the runs has ended: a zombie, which the job's shell reaps, holds no GPU
    while any(session in leaders and state != "Z" for state, _, session in processes().values()):
        time.sleep(5)
    for _ in range(60):  # and the GPUs' memory is free (Ray's workers may outlive their session)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
        if not apps.stdout.strip():
            break
        time.sleep(10)
    print("relay: units", units, flush=True)
    subprocess.run(["bash", str(Path(__file__).with_name("collect.sbatch"))], env={**os.environ, "UNITS": units, "RELAYED": "1"})


relay_file = args.suite_dir.parent / "relay"
if done and "RELAYED" not in os.environ and relay_file.exists():
    jobs, units = relay_file.read_text().splitlines()[:2]
    if os.environ.get("SLURM_JOB_ID") in jobs.split():
        relay(units)
sys.exit(0 if done else 1)
