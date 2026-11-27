"""OPD's teacher, started as verl starts it, scores one real episode: the check that its prefill fits its GPUs.

    python teacher_prefill.py <composed OPD config (.yaml)> <episode .pt> [<episode .pt> ...]

The config is opd/run.sbatch's distill call composed on CPU with the real teacher (a variant of the teacher's memory
settings). The teacher's servers come up through verl's own MultiTeacherModelManager on a Ray pool of
distillation.n_gpus_per_node GPUs (PEAR's vLLM server: fp32 vision tower, A_log and head), and verl's AsyncTeacherLLMServerManager
asks for the episode's prompt log-probs exactly as OPD does (compute_teacher_logprobs_single: the token ids and the
episode's images). The episode is the set's longest (158,599 tokens, 52 images of 512 x 512); its images are stand-ins
of the same size, which give the same tokens and memory. Prints the teacher's log-probs' summary and the time; a
failure (startup or out of memory) raises."""

import asyncio
import sys
import time

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image

import pear.rollout.server  # noqa: F401  (registers PEAR's vLLM replica, as the trainer's imports do)
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.single_controller.ray import RayResourcePool


def load(episode_path: str):
    episode = torch.load(episode_path, weights_only=False, mmap=True)
    ids = episode["input_ids"].tolist()
    grids = episode["multi_modal_inputs"]["image_grid_thw"].tolist()
    assert all(g == [1, 32, 32] for g in grids), grids  # 512 x 512 frames
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)) for _ in grids]
    return episode, ids, images


def main(config_path: str, episode_paths: list[str]) -> None:
    config = OmegaConf.load(config_path)
    teacher = config.distillation.teacher_models.teacher_model.inference
    print(
        f"{len(episode_paths)} episodes; teacher TP {teacher.tensor_model_parallel_size}, "
        f"gpu_memory_utilization {teacher.gpu_memory_utilization}, max_num_batched_tokens "
        f"{teacher.max_num_batched_tokens}",
        flush=True,
    )

    ray.init(runtime_env=OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env, resolve=True))
    gpus = config.distillation.n_gpus_per_node * config.distillation.nnodes
    pool = RayResourcePool(process_on_nodes=[gpus], use_gpu=True, max_colocate_count=1, name_prefix="teacher_check")
    start = time.time()
    teachers = MultiTeacherModelManager(config=config, resource_pool=pool)
    print(f"teacher up in {time.time() - start:.0f} s", flush=True)
    manager = AsyncTeacherLLMServerManager(config=config, teacher_client=teachers.get_client())

    loop = asyncio.new_event_loop()  # one loop for every request, as the agent loop keeps
    for path in episode_paths:
        episode, ids, images = load(path)

        async def score():
            # as the agent loop calls it (_compute_teacher_logprobs): raw images, the data's processor kwargs
            return await manager.compute_teacher_logprobs_single(
                sequence_ids=ids,
                multi_modal_data={"images": images},
                mm_processor_kwargs=OmegaConf.to_container(config.data.get("mm_processor_kwargs", {})),
            )

        start = time.time()
        teacher_ids, logprobs = loop.run_until_complete(score())
        took = time.time() - start
        lp = logprobs.float().flatten()
        print(
            f"scored {len(ids)} tokens ({len(images)} images) in {took:.0f} s ({len(ids) / took:.0f} tokens/s): "
            f"finite {bool(torch.isfinite(lp[1:]).all())}, mean over generated tokens "
            f"{lp[episode['loss_mask'].bool()].mean():.3f}, ids match {teacher_ids.tolist() == ids}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
