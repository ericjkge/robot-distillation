"""Our evaluation is PEAR's LIBERO evaluation: scripts/eval.sh composes, key for key, the config of PEAR's own recipe
(examples/eval/eval_libero.sh, the protocol of the paper's Table 2: LIBERO's 2,000 episodes, one rollout each, xhigh
effort, a 256k-token budget). The only differences allowed are the hardware's (configs/hardware/generic.sh: kernels and
caches, not the protocol), the absolute paths of the same files, and this repository on the workers' path."""

import json
import os
import stat
import subprocess

import pytest
from conftest import PEAR
from omegaconf import OmegaConf
from test_configs import compose_step
from test_scripts import CLUSTER, MODELS

TEACHER = {"MODEL_FILE": MODELS / "teacher.sh", "MODEL": "/models/teacher"}
STUDENT = {"MODEL_FILE": MODELS / "qwen3_5_9b.sh", "MODEL": "/models/student"}
# the stand-in interpreter for PEAR's recipe: it records the evaluation's launch only (the recipe also starts its
# progress panel with python -m)
STUB_PYTHON = """#!/usr/bin/env bash
if [ "$1" = -m ] && [ "$2" = pear.eval.embodied ]; then shift 2; printf '%s\\0' "$@" > "$OPD_CAPTURE"; exit 0; fi
if [ "$1" = -m ]; then exit 0; fi
exec {python} "$@"
"""
# what the hardware profile replaces (configs/hardware/generic.sh), and run-specific values
HARDWARE = {
    "actor_rollout_ref.actor.fsdp_config.offload_policy",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload_step",
    "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
    "actor_rollout_ref.actor.fsdp_config.fsdp_size",
    "actor_rollout_ref.actor.fsdp_config.use_gpu_backend",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.model.override_config.attn_implementation",
    "actor_rollout_ref.rollout.kernels",
    "actor_rollout_ref.rollout.custom.lmcache",
    "actor_rollout_ref.rollout.n_gpus_per_node",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
}
RUN_SPECIFIC = {"chat_template_file"}
# what decides the episodes, the prompts, the sampling and the score: identical, whatever the hardware
PROTOCOL = [
    "data.train_files",
    "data.val_files",
    "data.max_prompt_length",
    "data.max_response_length",
    "data.train_batch_size",
    "data.apply_chat_template_kwargs",
    "data.image_patch_size",
    "data.pixel_dtype",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.max_model_len",
    "actor_rollout_ref.rollout.limit_images",
    "actor_rollout_ref.rollout.dtype",
    "actor_rollout_ref.rollout.fp32_modules",
    "actor_rollout_ref.rollout.engine_kwargs",
    "actor_rollout_ref.rollout.multi_turn",
    "actor_rollout_ref.rollout.agent",
    "trainer.total_training_steps",
    "trainer.total_epochs",
    "trainer.eval_step",
    "run_sizes",
]


def flatten(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out or {prefix[:-1]: {}}
    return {prefix[:-1]: node}


def absolute(value):
    """PEAR's recipe names its dataset and agent-loop config relative to its root, where it starts."""
    if isinstance(value, str) and value.startswith("pear/") and (PEAR / value).exists():
        return str(PEAR / value)
    return value


def comparable(config) -> dict:
    flat = flatten(OmegaConf.to_container(config, resolve=False))
    flat = {k: absolute(v) for k, v in flat.items()}
    flat.pop("ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH")  # ours adds src/ (opd)
    flat.pop("ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH")  # named by the launch's time
    return flat


def under(key: str, prefixes) -> bool:
    return any(key == p or key.startswith(p + ".") for p in prefixes)


@pytest.fixture
def pear_recipe(verl_env, tmp_path):
    """pear_recipe(**variables) -> the Hydra overrides PEAR's examples/eval/eval_libero.sh passes its evaluation."""
    stub = tmp_path / "pear_env"
    (stub / "bin").mkdir(parents=True)
    real_python = subprocess.run(["bash", "-c", "command -v python3"], capture_output=True, text=True).stdout.strip()
    python = stub / "bin" / "python"
    python.write_text(STUB_PYTHON.format(python=real_python))
    python.chmod(python.stat().st_mode | stat.S_IEXEC)

    def run(**variables) -> list[str]:
        capture = tmp_path / "pear_capture"
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "WORK_DIR": str(tmp_path / "work"),
            "VERL_ENV": str(stub),
            "LOCAL_DIR": str(tmp_path / "local"),
            "RUN": str(tmp_path / "run"),
            "OPD_CAPTURE": str(capture),
            # PEAR's check knows its own patches only; ours (patches/verl) are scripts/verl.sh's to check
            "VERL_CHANGE_IN_PROGRESS": "1",
            **{k: str(v) for k, v in variables.items()},
        }
        script = PEAR / "examples" / "eval" / "eval_libero.sh"
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, cwd=PEAR)
        assert result.returncode == 0, result.stderr + result.stdout
        return capture.read_bytes().decode().split("\0")[:-1]

    return run


def pear_config(pear_recipe, model: str):
    args = pear_recipe(GPU="h100", MODEL=model, ROLLOUT_NNODES=1, EVAL_STEP=0)
    return compose_step("eval.sh", [a for a in args if not a.startswith("hydra.")])


@pytest.mark.parametrize("gpu", ["h100", "generic"])
def test_teacher_eval_is_pears_libero_recipe(run_step, pear_recipe, gpu):
    _, ours = run_step("eval.sh", GPU=gpu, **TEACHER, **CLUSTER)
    ours, theirs = comparable(compose_step("eval.sh", ours)), comparable(pear_config(pear_recipe, "/models/teacher"))
    allowed = RUN_SPECIFIC | (HARDWARE if gpu == "generic" else {"actor_rollout_ref.rollout.n_gpus_per_node"})
    differ = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k) and not under(k, allowed))
    assert not differ, json.dumps({k: [ours.get(k), theirs.get(k)] for k in differ}, indent=1, default=str)
    for key in PROTOCOL:
        assert {k: v for k, v in ours.items() if under(k, [key])} == {
            k: v for k, v in theirs.items() if under(k, [key])
        }, key


def test_the_protocol_is_the_papers(run_step):
    _, args = run_step("eval.sh", GPU="generic", **TEACHER, **CLUSTER)
    config = compose_step("eval.sh", args)
    episodes = json.loads((PEAR / "pear" / "envs" / "libero" / "configs" / "libero_all.json").read_text())
    assert list(config.data.train_files) == [str(PEAR / "pear" / "envs" / "libero" / "configs" / "libero_all.json")]
    assert episodes["env"] == {
        "suite": ["libero_goal", "libero_spatial", "libero_object", "libero_10"],
        "task_id": "0-9",
        "trial_id": "0-49",
    }  # 4 suites x 10 tasks x 50 initial states: the paper's 2,000 episodes
    assert episodes["data_source"] == "suite"  # the summary's per-suite scores, the columns of Table 2
    rollout = config.actor_rollout_ref.rollout
    assert rollout.n == 1 and config.trainer.total_training_steps == 1  # one rollout of each episode, one pass
    assert (rollout.temperature, rollout.top_p, rollout.top_k) == (1.0, 0.95, -1)
    assert config.data.max_response_length == 256000 and rollout.max_model_len == 4096 + 256000  # 256k (Table 1)
    assert config.data.apply_chat_template_kwargs.enable_thinking is True
    # no effort given: the teacher's chat template defaults to xhigh, the paper's setting unless stated otherwise
    assert "reasoning_effort" not in config.data.apply_chat_template_kwargs
    assert list(rollout.fp32_modules) == ["visual", "A_log"]
    assert rollout.engine_kwargs.vllm.hf_overrides.head_dtype == "float32"


@pytest.mark.parametrize("gpu", ["h100", "generic"])
def test_student_eval_differs_from_the_teachers_only_in_model(run_step, gpu):
    """A student is evaluated as the teacher is: the same protocol, with the teacher's chat template (so the same
    system prompt and effort) and its own tensor parallelism."""
    _, teacher = run_step("eval.sh", GPU=gpu, **TEACHER, **CLUSTER)
    _, student = run_step("eval.sh", GPU=gpu, **STUDENT, **CLUSTER)
    teacher, student = comparable(compose_step("eval.sh", teacher)), comparable(compose_step("eval.sh", student))
    differ = {k for k in set(teacher) | set(student) if teacher.get(k) != student.get(k)}
    assert differ == {
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "chat_template_file",
    }, sorted(differ)
    assert student["chat_template_file"].endswith("Qwen3.8-27B-PEAR-LIBERO_only/chat_template.jinja")
