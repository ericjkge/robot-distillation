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

*Updated 2026-11-23 01:25. The full history of fixes, submissions and earlier notes is in git, up to commit e1643d7.*

**Now running or queued**

| Step | Jobs | State |
| --- | --- | --- |
| Teacher episodes (goal, spatial, object, libero_10) | 60483184, 60483185, 60483186, 60483187 | goal running since 01:16; each stops once every task has 25 |
| Base baseline 9B, each suite split in two (initial states 0–24 / 25–49) | goal 60486046 / 60486047, spatial 60486048 / 60486049, object 60486050 / 60486072, libero_10 60486073 / 60486085 | queued; for the presentation today at 17:00 |
| Base baseline 4B | none | cancelled to make room; resubmit after the presentation |
| Smoke part 3 (distill, export, eval) | 60471092 | queued; earlier attempt ran out of host memory at 96G, now 192G |
| SFT 9B / 4B | 60483188, 60483189 | waits for all four suites |
| OPD | none | not yet, at the human's direction |

All of these except part 3 run `main` bf9b274.

**Human decisions**
- Students use the teacher's chat template (reasoning effort in the system prompt; every round's reasoning kept).
- SFT: 25 successful episodes per task, drawn at random (all of a task's if it has fewer); lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs.
- Base baselines: full LIBERO (2,000 episodes, PEAR's protocol), xhigh, LIBERO only.
- Presentation today (Nov 23) at 17:00: a complete baseline for one student. The 9B was chosen, with each suite split across two jobs so that all four finish around 09:00–11:00. The teacher's numbers are the paper's (74.4% xhigh, 21.9% low; brief cff8cb4).
- Collection stops per suite once each task has 25 successes or has ended its 50 trials (`collect/enough.py`).
- Compute: any limit, as long as it costs no priority. GPU jobs go on `gpu_requeue` only, and may use any GPU with enough memory (`vram:48G`, or `vram:45G` for one student GPU).

**Latest fixes (Nov 22 23:30 to Nov 23 00:50)**
- **Teacher stall:** at PEAR's 32 episodes at once, the teacher's KV cache was full (97%, 208 preemptions, no prefix hits). Now 12 at once for the teacher and 8 per student GPU.
- **Jobs sharing a node killed each other** (baselines 60479702 and 60479703). Every Ray cluster used port 6379 and one temporary directory, and `ray stop --force` stopped all of the user's clusters on the machine. Fixed on `main` (bf9b274): each run gets a free port and its own temporary directory, and stops only its own processes. The head address is now resolved as IPv4 (off Slurm, `getent hosts` returned IPv6 first and the run waited forever), and a cluster that fails to start fails its run within 10 minutes.
- **Tests on bf9b274:** the CPU suite passes 75 of 75. `repo_tests/two_clusters.sbatch` runs two clusters on one node, under Slurm (60482850) and off it (60482851). It passes both times: separate clusters, the short run's exit leaves the long one's task to complete, no processes are left over, and a node that cannot start fails its run in 44 s. On the old code it fails (60482296).

**Verified on GPU**
- Base 9B baselines run, two of them sharing one A6000 node. After about 40 minutes: 38 episodes, 0 successes; most end at once because the reply is not a valid action ("parse"), and the rest run out of steps. About 10 episodes per 25 minutes per job, so about 20 hours per suite. Host memory is 31–35 GB of 64G.
- Collect, SFT (sequence parallelism and CPU offload) and eval work end to end on the smoke test.
- The 27B teacher runs as one TP4 replica on 4 L40S, using 75 GB of host memory.
- Distill reaches the student's update; export and eval of the distilled model are not yet confirmed (part 3).

**Findings to keep**
- Tokenizers: the teacher and students share vocabulary and merges. The differences (7 audio tokens, a combining-mark rule) never occur in LIBERO text, so they are compatible (jobs 60171018 and 60171577).
- Algorithm 1 versus `main`'s distill step: the advantage, loss, batch and evaluation interval all match. Open: PEAR trains on episodes up to 4 updates old, while Algorithm 1 uses only the current policy (see Blocked on).
- Ray's CPUs: verl reserves 2 per rollout GPU, and each simulator and storage unit takes 1. A shortfall hangs the run silently; the README on `main` (7b31af8) now says so.

**Submission headers** (all `gpu_requeue`, `--requeue`)
- Collection, code bf9b274 (`collect/collect.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 128G; 12 episodes at once.
- Baselines, code bf9b274 (`baseline/eval.sbatch`, `TRIALS` selects the initial states): gpu:1,vram:45G, 3 days, 10 CPUs, 64G (35 GB measured); 8 episodes at once.
- SFT, code bf9b274 (`sft/run.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 450G (4B: 224G); afterok on all four suites.
- Smoke part 3, code 35a76df (`smoke/part3.sbatch`): gpu:l40s:4, 1h15, 16 CPUs, 192G.

## Results

None yet.

## Next step

When libero_goal collection starts (around 03:00), check the teacher's KV cache stays below full with prefix-cache hits. Record each baseline suite's success rate and time per episode as its job finishes.

## Blocked on

- **Before OPD (not needed now, at the human's direction):** `max_off_policy_threshold`, 5 (PEAR) or 1 (Algorithm 1; recommended), and the OPD run settings (K, B, lr, validation set).

## Dead ends
