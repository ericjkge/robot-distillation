"""The teacher's remaining LIBERO episodes on Modal, as O2 runs them (main's scripts/collect.sh: the teacher at xhigh,
160,000 response tokens), on PEAR's own H100 recipe: two TP4 replicas on 8 H100s, every episode at once.
    modal run modal/app.py::setup      # once: PEAR's environments and the teacher into the volume (1 H100, ~1-1.5 h)
    modal run modal/app.py::collect    # gcp/goal5-9.json's 45 trials, then the upload to the Hub as source modal
The volume keeps the environments, the teacher and the runs; a rerun of collect resumes (opd.episodes: only trials
without a record run). The Hub token is the Modal secret "hf" (HF_TOKEN)."""

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
WORK = "/work"
CODE = "/root/robot-distillation"
COMMIT = "97059ba"  # main: scripts/collect.sh, opd.episodes
REPO = "eddykang06/qwen3.8-27b-pear-libero-rollouts"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "wget", "bzip2", "ca-certificates", "build-essential", "numactl", "libegl1", "libgl1")
    .run_commands(
        f"git clone -q https://github.com/ericjkge/robot-distillation {CODE}",
        f"git -C {CODE} checkout -q {COMMIT}",
        f"git -C {CODE} submodule update -q --init",
        f"bash {CODE}/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh",
    )
    .add_local_file(HERE.parent / "gcp" / "goal5-9.json", f"{CODE}/goal5-9.json")
    .add_local_file(HERE.parent / "collect" / "hub.py", f"{CODE}/hub.py")
)
volume = modal.Volume.from_name("robot-distillation-work", create_if_missing=True)
app = modal.App("robot-distillation", image=image)


def sh(command: str) -> None:
    """Runs a bash command in the code directory with WORK_DIR set, its output streamed to the Modal logs."""
    subprocess.run(["bash", "-c", f"set -euo pipefail; export WORK_DIR={WORK}; cd {CODE}; {command}"], check=True)


@app.function(gpu="H100", cpu=16, memory=65536, timeout=4 * 3600, volumes={WORK: volume})
def setup():
    sh("BENCHES=libero MODEL_FILES=configs/models/teacher.sh bash scripts/setup.sh")
    volume.commit()


@app.function(gpu="H100:8", cpu=64, memory=262144, timeout=24 * 3600, volumes={WORK: volume},
              secrets=[modal.Secret.from_name("hf")])
def collect():
    run = f"{WORK}/runs/goal5-9"
    sh(f"""
    attempt={run}/attempt-$(date +%s); mkdir -p $attempt
    PYTHONPATH=src {WORK}/envs/verl/bin/python -m opd.episodes goal5-9.json {run} > $attempt/episodes.json
    n=$(grep -o '"trial_id"' $attempt/episodes.json | wc -l || true)
    echo "== $(date +%H:%M) running $n episodes"
    if [ "$n" -gt 0 ]; then
      # every episode at once: two TP4 replicas with half each, a simulator per episode
      per_replica=$(((n + 1) / 2))
      GPU=h100 SESSIONS_PER_REPLICA=$per_replica LIBERO_SERVERS=$n EPISODES=$attempt/episodes.json \\
        MODEL={WORK}/models/Qwen3.8-27B-PEAR-LIBERO_only RUN=$attempt bash scripts/run.sh scripts/collect.sh
    fi
    rm -rf /root/stage
    python3 hub.py /root/stage modal {run}/attempt-*
    {WORK}/envs/verl/bin/python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='{REPO}', repo_type='dataset', folder_path='/root/stage', commit_message='goal 5 and 9 from modal',
                      allow_patterns=['libero_goal/records/modal.jsonl', 'libero_goal/episodes/*.pt'])"
    echo "done: $(grep -c . /root/stage/libero_goal/records/modal.jsonl) episodes run, $(ls /root/stage/libero_goal/episodes | wc -l) kept"
    """)
    volume.commit()
