---
slug: on-policy-distillation
status: running
hypothesis: On-policy reverse-KL distillation from the PEAR 27B teacher makes a Qwen3.5-9B student succeed more often on LIBERO than SFT alone, and more often than the teacher at low effort, while taking less time per episode than the teacher at xhigh.
kill_criterion: After on-policy distillation, the 9B student's LIBERO success rate (PEAR protocol, 2,000 episodes) is not more than 2 SE above its SFT-only checkpoint, or not above 21.9%, or its mean time per episode is not below the teacher's at xhigh.
best_metric:
started: 2026-11-19
---

# on-policy-distillation

## Hypothesis

On-policy reverse-KL distillation from Qwen3.8-27B-PEAR-LIBERO_only gives a Qwen3.5-9B student a higher LIBERO success rate than its SFT-only checkpoint and than the teacher at low effort (21.9%), at lower mean time per episode than the teacher at xhigh.

## Kill criterion

The direction is dead if the on-policy-distilled 9B student, evaluated with PEAR's protocol (2,000 episodes: 40 tasks × 50 initial states), fails any of these:
1. Its success rate is more than 2 SE above its own SFT-only checkpoint.
2. Its success rate is above 21.9%, the teacher's paper number at low effort.
3. Its mean wall-clock time per episode is below the teacher's at xhigh, measured on the same hardware.

The 4B student is reported alongside the 9B student but does not decide the direction.

## Current state

*Updated 2026-11-22 23:00. The full history of fixes, submissions and earlier notes is in git, up to commit e1643d7.*

**Now running or queued**

| Step | Jobs | State |
| --- | --- | --- |
| Teacher episodes, libero_goal | 60470355 | running since 20:44; stopped automatically once each task has 25 |
| Teacher episodes, spatial / object / libero_10 | 60478188, 60478189, 60478190 | queued |
| Base baselines, 9B (goal, spatial, object, libero_10) | 60478193, 60478194, 60478195, 60478196 | queued |
| Base baselines, 4B (same order) | 60478197, 60478198, 60478199, 60478200 | queued |
| Smoke part 3 (distill, export, eval) | 60471092 | queued; earlier attempt ran out of host memory at 96G, now 192G |
| SFT 9B / 4B | 60478203, 60478204 | waits for collection |
| OPD | none | not yet, at the human's direction |

**Human decisions**
- Students use the teacher's chat template (reasoning effort in the system prompt; every round's reasoning kept).
- SFT: 25 successful episodes per task, drawn at random (all of a task's if it has fewer); lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs.
- Base baselines: full LIBERO (2,000 episodes, PEAR's protocol), xhigh, LIBERO only. The teacher's numbers are the paper's (74.4% xhigh, 21.9% low; brief cff8cb4).
- Collection stops per suite once each task has 25 successes or has ended its 50 trials (`collect/enough.py`).
- Compute: any limit, as long as it costs no priority. GPU jobs go on `gpu_requeue` only, and may use any GPU with enough memory (`vram:48G`, or `vram:45G` for one student GPU).

**Verified on GPU**
- Collect, SFT (sequence parallelism and CPU offload) and eval work end to end on the smoke test.
- The 27B teacher runs as one TP4 replica on 4 L40S: 75 GB of host memory; about 50 episodes in the first 100 minutes, slowing as long and failing episodes fill the 32 slots.
- Distill reaches the student's update; export and eval of the distilled model are not yet confirmed (part 3).

**Findings to keep**
- Tokenizers: the teacher and students share vocabulary and merges. The differences (7 audio tokens, a combining-mark rule) never occur in LIBERO text, so they are compatible (jobs 60171018 and 60171577).
- Algorithm 1 versus `main`'s distill step: the advantage, loss, batch and evaluation interval all match. Open: PEAR trains on episodes up to 4 updates old, while Algorithm 1 uses only the current policy (see Blocked on).
- Ray's CPUs: verl reserves 2 per rollout GPU, and each simulator and storage unit takes 1. A shortfall hangs the run silently; the README on `main` (7b31af8) now says so.

**Submission headers** (all `gpu_requeue`, `--requeue`)
- Collection, code e00b38c (`collect/collect.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 128G.
- Baselines, code 35a76df (`baseline/eval.sbatch`): gpu:1,vram:45G, 3 days, 10 CPUs, 64G (memory unmeasured).
- SFT, code 3b30094 (`sft/run.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 450G (4B: 224G); afterany on 60470355, afterok on the other three suites.
- Smoke part 3, code 35a76df (`smoke/part3.sbatch`): gpu:l40s:4, 1h15, 16 CPUs, 192G.

## Results

None yet.

## Next step

Record each baseline suite's success rate and time per episode as its job finishes, and resize the baseline memory from the first job's MaxRSS.

## Blocked on

- **Before OPD (not needed now, at the human's direction):** `max_off_policy_threshold`, 5 (PEAR) or 1 (Algorithm 1; recommended), and the OPD run settings (K, B, lr, validation set).

## Dead ends
