"""scripts/eval_general.sh: the bonus general-capability evaluation is PEAR's own recipe
(examples/eval/eval_general.sh), unchanged apart from the model file: the same vLLM servers, the same benchmark run."""

import os
import stat
import subprocess

import pytest
import yaml
from conftest import PEAR, ROOT

MODELS = ROOT / "configs" / "models"
STUBS = {
    "nvidia-smi": "for i in 0 1 2 3; do echo GPU $i; done",
    "vllm": 'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@" > "$CAPTURE/vllm.$CUDA_VISIBLE_DEVICES"',
    "curl": "exit 0",
    "eval_python": '[ "$3" = run ] && printf "%s\\n" "$@" > "$CAPTURE/client"; exit 0',
    "sleep": "exit 0",
}


@pytest.fixture
def launch(verl_env, tmp_path):
    """launch(script, command, model_file) -> {name: captured lines}: what the recipe starts, with stand-ins."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{text}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def run(script, command, model_file):
        capture = tmp_path / f"capture-{script.name}-{command}-{model_file.stem}"
        capture.mkdir()
        env = {
            "PATH": f"{bin_dir}:{verl_env}/bin:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "WORK_DIR": "/work",
            "LOCAL_DIR": str(tmp_path / "local"),
            "VERL_ENV": str(verl_env),
            "EVAL_PYTHON": str(bin_dir / "eval_python"),
            "MODEL_FILE": str(model_file),
            "MODEL": "/models/m",
            "NAME": "m",
            "RUN": str(tmp_path / "run"),
            "GPU": "h100",
            "PEAR": str(PEAR),
            "CAPTURE": str(capture),
        }
        args = ["bash", str(script), command] + (["h:8000"] if command == "run" else [])
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return {p.name: p.read_text().splitlines() for p in capture.iterdir()}

    return run


PEARS = PEAR / "examples" / "eval" / "eval_general.sh"
OURS = ROOT / "scripts" / "eval_general.sh"


@pytest.mark.parametrize("command", ["serve", "run"])
def test_the_teacher_is_evaluated_exactly_as_pears_recipe(launch, command):
    pears = launch(PEARS, command, PEAR / "examples" / "models" / "qwen3_8_27b_pear" / "qwen3_8_27b.sh")
    assert pears  # the recipe started its servers or its client
    assert launch(OURS, command, MODELS / "teacher.sh") == pears


def flags(lines: list[str]) -> dict[str, str]:
    """A vllm serve call's flags, its GPUs (CUDA_VISIBLE_DEVICES) and model as entries, the port left out: each
    replica has its own."""
    gpus, _, model, *rest = lines
    out, key = {"gpus": gpus, "model": model}, None
    for item in rest:
        if item.startswith("--"):
            key = item
            out[key] = ""
        else:
            out[key] = item
    out.pop("--port")
    return out


def test_a_student_differs_only_in_its_tensor_parallelism(launch):
    (teacher,) = [flags(v) for v in launch(OURS, "serve", MODELS / "teacher.sh").values()]
    students = [flags(v) for v in launch(OURS, "serve", MODELS / "qwen3_5_9b.sh").values()]
    assert teacher["gpus"] == "0,1,2,3" and sorted(s["gpus"] for s in students) == ["0", "1", "2", "3"]
    for student in students:
        assert student["--tensor-parallel-size"] == "1"
        assert {k: v for k, v in student.items() if k not in ("gpus", "--tensor-parallel-size")} == {
            k: v for k, v in teacher.items() if k not in ("gpus", "--tensor-parallel-size")
        }
    assert "--chat-template" not in teacher  # each model its own template, as PEAR serves every model


def test_the_protocol_is_pears():
    config = yaml.safe_load((PEAR / "pear" / "configs" / "general.yaml").read_text())
    assert config["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
    assert sorted(config["benchmarks"]) == sorted(
        [
            "aime26",
            "gpqa_diamond",
            "ifbench",
            "live_code_bench",
            "real_world_qa",
            "erqa",
            "mmmu",
            "mmmu_pro_standard10",
            "mmmu_pro_vision",
        ]
    )
