"""scripts/ray_node.sh: the Ray resources each role's machine advertises."""

import os
import stat
import subprocess

import pytest
from conftest import ROOT


@pytest.mark.parametrize(
    "role,resources",
    [
        ("trainer", '{"trainer": 2}'),
        ("rollout", '{"rollout": 2, "libero": 4}'),
        # verl requires trainer.node_label of every training pool, the teacher's included: without it, a teacher
        # machine of its own never gets the teacher's placement group
        ("teacher", '{"trainer": 2}'),
    ],
)
def test_each_role_advertises_its_resources(verl_env, tmp_path, role, resources):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in [("nvidia-smi", "echo GPU 0; echo GPU 1"), ("ray", 'printf "%s\\n" "$@" > "$RAY_ARGS"')]:
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{text}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    for tool in (verl_env / "bin").iterdir():
        (env_dir / "bin" / tool.name).symlink_to(tool)
    (env_dir / "bin" / "ray").symlink_to(bin_dir / "ray")
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_DIR": str(tmp_path / "work"),
        "LOCAL_DIR": str(tmp_path / "local"),
        "VERL_ENV": str(env_dir),
        "LIBERO_SERVERS": "4",
        "RAY_ARGS": str(tmp_path / "args"),
    }
    script = ROOT / "scripts" / "ray_node.sh"
    result = subprocess.run(["bash", str(script), role], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args").read_text().splitlines()
    assert f"--resources={resources}" in args and "--num-gpus=2" in args
