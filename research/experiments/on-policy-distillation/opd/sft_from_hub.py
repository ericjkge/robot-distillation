"""The SFT checkpoint OPD starts from, from its Hub repo: waits until the repo's weights are SFT's final checkpoint
(the same SHA-256 for every safetensors file; sft/live.py uploads it minutes after SFT ends), downloads that revision
under <models dir> and prints its path.
    python sft_from_hub.py <final checkpoint>/huggingface <repo> <models dir>"""

import hashlib
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def hub_revision(local: Path, repo: str, timeout: float = 3 * 3600, poll: float = 120) -> str:
    """The repo's revision whose safetensors files are exactly local's, once there is one."""
    want = {p.name: sha256(p) for p in local.glob("*.safetensors")}
    assert want, f"no safetensors in {local}"
    api, deadline = HfApi(), time.time() + timeout
    while True:
        info = api.model_info(repo, files_metadata=True)
        have = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.rfilename.endswith(".safetensors") and s.lfs}
        if have == want:
            return info.sha
        if time.time() > deadline:
            sys.exit(f"{repo} does not hold {local} after {timeout / 3600:.0f} h")
        print(f"{time.strftime('%H:%M')} waiting for {repo} to hold {local}", file=sys.stderr, flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    local, repo, models = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    revision = hub_revision(local, repo)
    path = snapshot_download(repo, revision=revision, local_dir=models / f"{repo.split('/')[-1]}-{revision[:7]}")
    print(path)
