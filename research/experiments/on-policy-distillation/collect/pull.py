"""Download other clusters' episodes from the Hub dataset into the staging directory (collect/hub.py), so that it holds
every source: each source's <suite>/records/<source>.jsonl, then its successful episodes. Rerun hub.py afterwards to
rebuild index.jsonl over all of them.
    python pull.py <stage> <repo> [--skip o2]   (--skip: the sources already staged here, from our own runs)"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=Path)
parser.add_argument("repo")
parser.add_argument("--skip", nargs="*", default=["o2"])
args = parser.parse_args()

files = HfApi().list_repo_files(args.repo, repo_type="dataset")
records = [f for f in files if f.split("/")[1:2] == ["records"] and Path(f).stem not in args.skip]
snapshot_download(args.repo, repo_type="dataset", local_dir=args.stage, allow_patterns=records)
episodes = []
for path in records:
    for line in (args.stage / path).read_text().splitlines():
        record = json.loads(line)
        if record["score"] >= 1:  # as hub.py: a success, whose episode was uploaded
            episodes.append(f"{path.split('/')[0]}/episodes/{record['uid']}.pt")
missing = sorted(set(episodes) - set(files))
if missing:
    raise SystemExit(f"{len(missing)} successful episodes are not on the Hub yet, e.g. {missing[0]}")
snapshot_download(args.repo, repo_type="dataset", local_dir=args.stage, allow_patterns=episodes, max_workers=8)
print(f"{len(records)} record files, {len(episodes)} episodes, from {sorted({Path(r).stem for r in records})}")
