"""Integrates other collectors' open pull requests to the Hub dataset in its own format: each PR's records
(<suite>/records/<source>.jsonl) and new episodes (<suite>/episodes/<uid>.pt) are checked, copied into the staging
directory, and committed to main with index.jsonl and README.md rebuilt by hub.py; the PR is then closed. A PR's own
index.jsonl and README.md are ignored (they conflict with main whenever main moves). A PR that touches anything else,
drops earlier records, changes the record schema or lacks a success's episode stays open, with the reason printed.
    python prs.py <stage> <repo> <hub.py arguments after the stage...>"""

import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

stage, repo, hub_args = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
api = HfApi()
REBUILT = {"index.jsonl", "README.md"}


def files(revision: str) -> dict:
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 if s.lfs else s.blob_id for s in info.siblings}


def problems(pr_files: dict, main: dict, revision: str) -> list[str]:
    changed = [f for f, sha in pr_files.items() if main.get(f) != sha and f not in REBUILT and f != ".gitattributes"]
    records = [f for f in changed if f.split("/")[1:2] == ["records"]]
    episodes = [f for f in changed if f.split("/")[1:2] == ["episodes"] and f.endswith(".pt")]
    out = [f"unexpected file {f}" for f in changed if f not in records + episodes]
    out += [f"{f} replaces an existing episode" for f in episodes if f in main]
    out += [f"{f} is our own source" for f in records if Path(f).stem == "o2"]
    reference = sorted(json.loads(next((stage / "libero_10" / "records" / "o2.jsonl").open())))
    for path in records:
        new = [
            json.loads(line)
            for line in Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision)).open()
        ]
        old = stage / path
        if old.exists() and {json.loads(line)["uid"] for line in old.open()} - {r["uid"] for r in new}:
            out.append(f"{path} drops earlier records")
        out += [f"{path}: record {r['uid']} has another schema" for r in new if sorted(r) != reference]
        suite = path.split("/")[0]
        out += [
            f"{path}: success {r['uid']} has no episode"
            for r in new
            if r["score"] >= 1 and f"{suite}/episodes/{r['uid']}.pt" not in pr_files
        ]
    return out, records + episodes


main = files("main")
for pr in api.get_repo_discussions(repo, repo_type="dataset"):
    if not pr.is_pull_request or pr.status != "open":
        continue
    revision = f"refs/pr/{pr.num}"
    found, wanted = problems(files(revision), main, revision)
    if found:
        print(f"PR #{pr.num} ({pr.author}) left open: " + "; ".join(found[:3]))
        continue
    for path in wanted:
        hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=stage)
    subprocess.run([sys.executable, str(Path(__file__).with_name("hub.py")), str(stage), *hub_args], check=True)
    operations = [CommitOperationAdd(f, stage / f) for f in [*wanted, *REBUILT]]
    api.create_commit(
        repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"{pr.title} (PR #{pr.num}): index and card rebuilt",
    )
    api.change_discussion_status(
        repo,
        pr.num,
        "closed",
        repo_type="dataset",
        comment=(
            f"Integrated in main: {len(wanted)} files, with index.jsonl and README.md rebuilt in the dataset's format. "
            "A PR's own index and README are not used, as they conflict whenever main moves."
        ),
    )
    print(f"PR #{pr.num} ({pr.author}) integrated: {len(wanted)} files")
    main = files("main")
