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

The 4B student is not run (human, Nov 23).

## Current state

*Updated 2026-11-23 17:20. The full history of fixes, submissions and earlier notes is in git, up to commit e1643d7.*

**Now running or queued**

| Step | Jobs | State |
| --- | --- | --- |
| Teacher episodes | spatial 60512542 (TP4, last task); goal, object, libero_10 each split in two by task (0–4 / 5–9) on 2 GPUs: 60554675 / 60554677, 60554678 / 60554679, 60554680 / 60554681 | spatial: task 9 has 17 of 47 ended, so it ends short of 25; the six TP2 jobs queued (17:15) |
| Base baseline 9B, each suite split in two (initial states 0–24 / 25–49; tally: `baseline/tally.py`) | goal 60512338 / 60512340, spatial 60512341 / 60486049, object 60512117 / 60486072, libero_10 60512116 / 60486085 | spatial done (500 episodes, 0 successes); 6 running; at 16:58 2/1,604 (goal 2/417, object 0/410, libero_10 0/277); five resumed after preemptions (see Latest fixes); for the presentation today at 17:00 |
| Smoke part 3 (distill, export, eval) | 60517692 (any 4 GPUs of 48 GB, was L40S only) | candidate verl fix with a probe (see Blocked on) |
| SFT 9B | 60554683 | afterok on the seven collection jobs (60517720 was cancelled with the old collection jobs) |
| OPD 9B, from the SFT checkpoint | none | after SFT and the verl fix |

The resumed halves run `main` 47200a9 (adds the worker-port fix); spatial 25–49, object 25–49, libero_10 25–49, collection and SFT run bf9b274.

**Human decisions**
- No 4B runs: the 4B baseline and SFT are dropped (17:10).
- Students use the teacher's chat template (reasoning effort in the system prompt; every round's reasoning kept).
- SFT: 25 successful episodes per task, drawn at random (all of a task's if it has fewer); lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs.
- Base baselines: full LIBERO (2,000 episodes, PEAR's protocol), xhigh, LIBERO only.
- As fast as possible (17:15): one finished 9B run, SFT then OPD; no OPD from the untuned student. So collection moved to 2 GPUs (TP2, 4 episodes at once), which start far sooner than 4, each suite split in two by task; the 4-GPU jobs (60512543, 60512544, 60512585; Slurm estimated Nov 26–27) were cancelled. OPD will use `max_off_policy_threshold=1` (Algorithm 1).
- Presentation today (Nov 23) at 17:00: a complete baseline for one student. The 9B was chosen, with each suite split across two jobs. At the measured rates (0.14–0.37 episodes a minute per job, the student KV caches near full), about 1,300 of the 2,000 episodes will be done by 15:00, covering every task; libero_10 will be partial. The human chose to keep the jobs as they are rather than re-split. The teacher's numbers are the paper's (74.4% xhigh, 21.9% low; brief cff8cb4).
- Collection stops per suite once each task has 25 successes or has ended its 50 trials (`collect/enough.py`).
- Compute: any limit, as long as it costs no priority. GPU jobs go on `gpu_requeue` only, and may use any GPU with enough memory (`vram:48G`, or `vram:45G` for one student GPU).

**Latest fixes (Nov 22 23:30 to Nov 23 08:00)**
- **Preemption (07:35):** libero_10 0–24 (60486073) was preempted and object 0–24 (60487512) requeued. The job script deleted its run on restart (`rm -rf`), so both would have lost their finished episodes (39 and 47). Both were held and their episodes kept. Now each attempt writes its own `attempt-<job>-<restart>/` and runs only the episodes no attempt has finished (`baseline/remaining.py`); `tally.py` counts an episode once. Resubmitted as 60512116 and 60512117. At 07:41 goal 0–24, goal 25–49 and spatial 0–24 (60486046–48) were preempted too and moved the same way (60512338, 60512340, 60512341). The three halves and the collection jobs still on the old script are watched: one that is preempted is cancelled before it restarts, for the same move. Collection has the same `rm -rf`. At 07:42 spatial collection (60483185) was preempted; collection now resumes the same way, leaving out the tasks that already have 25 kept (`remaining.py --keep 25`), and SFT draws from every attempt's run (tested: 278 episodes over 18 tasks). Object and libero_10 collection were resubmitted with the new script before starting. Goal collection (60483184) followed at 07:45 and was resubmitted as 60512585. The resumed baselines load only their remaining episodes (object 0–24: 203, libero_10 0–24: 211, as `remaining.py` computed).
- **Teacher stall:** at PEAR's 32 episodes at once, the teacher's KV cache was full (97%, 208 preemptions, no prefix hits). Now 12 at once for the teacher and 8 per student GPU.
- **Ray worker ports (main 47200a9):** two clusters starting on one node at once drew the same worker port from Ray's fixed range (the object 0–24 half, 60486050, crashed). Workers now take OS-assigned ports. The CPU suite passes 75 of 75, and the two-cluster test, now with 8 actors per cluster, passes (60486976).
- **Jobs sharing a node killed each other** (baselines 60479702 and 60479703). Every Ray cluster used port 6379 and one temporary directory, and `ray stop --force` stopped all of the user's clusters on the machine. Fixed on `main` (bf9b274): each run gets a free port and its own temporary directory, and stops only its own processes. The head address is now resolved as IPv4 (off Slurm, `getent hosts` returned IPv6 first and the run waited forever), and a cluster that fails to start fails its run within 10 minutes.
- **Tests on bf9b274:** the CPU suite passes 75 of 75. `repo_tests/two_clusters.sbatch` runs two clusters on one node, under Slurm (60482850) and off it (60482851). It passes both times: separate clusters, the short run's exit leaves the long one's task to complete, no processes are left over, and a node that cannot start fails its run in 44 s. On the old code it fails (60482296).

**Verified on GPU**
- Base 9B baselines run, two of them sharing one A6000 node. After about 40 minutes: 38 episodes, 0 successes; most end at once because the reply is not a valid action ("parse"), and the rest run out of steps. About 10 episodes per 25 minutes per job, so about 20 hours per suite. Host memory is 31–35 GB of 64G.
- Collect, SFT (sequence parallelism and CPU offload) and eval work end to end on the smoke test.
- The 27B teacher runs as one TP4 replica on 4 L40S, using 75 GB of host memory. At 12 episodes at once its KV cache is 27% full, with no preemptions and an 82% prefix-cache hit rate, generating 16k tokens a minute (11k at 32). Its first 29 episodes on goal and spatial all succeeded.
- Distill runs validation, rollout, teacher prefill and the student's backward pass. Part 3 (60471092, 192G) then ran out of GPU memory in Adam's step on 2 trainer L40S: PEAR's `optimizer_offload` brings the 4B's fp32 state onto the GPUs, as in the SFT pilot. Resubmitted with `actor_rollout_ref.actor.fsdp_config.offload_policy=true` (FSDP2 CPU offload, as SFT uses). With the state on the CPU (60494741), the first update completed. Fetching the next mini-batch failed (`split_with_sizes ... 5129 ... [4, 4]`): pickling through the transfer queue resets the nested 3D `position_ids`' ragged dimension from 2 to 1. verl's `train_mini_batch` repairs it only on each selected mini-batch (`maybe_fix_3d_position_ids`), not on the batch it selects from. Reproduced on CPU, and the repair applied to the whole batch first fixes it with the values intact. Export and eval of the distilled model are not yet confirmed.

**Findings to keep**
- Tokenizers: the teacher and students share vocabulary and merges. The differences (7 audio tokens, a combining-mark rule) never occur in LIBERO text, so they are compatible (jobs 60171018 and 60171577).
- Algorithm 1 versus `main`'s distill step: the advantage, loss, batch and evaluation interval all match. Open: PEAR trains on episodes up to 4 updates old, while Algorithm 1 uses only the current policy (see Blocked on).
- Ray's CPUs: verl reserves 2 per rollout GPU, and each simulator and storage unit takes 1. A shortfall hangs the run silently; the README on `main` (7b31af8) now says so.

**Submission headers** (all `gpu_requeue`, `--requeue`)
- Collection, code bf9b274 (`collect/collect.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 128G; 12 episodes at once. Spatial, object and libero_10 resubmitted (60512542–44), and goal (60512585), with the same header and code, so all suites land in `runs/collect-bf9b274`.
- Collection TP2, code bf9b274 (`collect/collect.sbatch` at eb1a500 and after, `TASKS` selects the tasks): gpu:2,vram:48G, 2 days, 16 CPUs, 128G; 4 episodes at once. Same `runs/collect-bf9b274`.
- Baselines, code bf9b274 (`baseline/eval.sbatch`, `TRIALS` selects the initial states): gpu:1,vram:45G, 3 days, 10 CPUs, 64G (35 GB measured); 8 episodes at once. The resumed halves (60512116, 60512117, 60512338, 60512340, 60512341) run code 47200a9 with the same header.
- SFT, code bf9b274 (`sft/run.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 450G (4B: 224G); afterok on collection (now 60554683: 60512542 and 60554675–81).
- Smoke part 3, code 47200a9 (`smoke/part3.sbatch`): gpu:l40s:4, 1h15, 16 CPUs, 192G.

## Results

- **Base Qwen3.5-9B, LIBERO, xhigh, PEAR's protocol (for the presentation, 14:58; partial):** 1 success in 1,417 of 2,000 episodes, covering all 40 tasks; LIBERO mean 0.1%. Compare the teacher's 74.4% (paper, xhigh).

| Suite | Episodes | Successes | Rate | Wilson 95% |
| --- | --- | --- | --- | --- |
| spatial | 481 | 0 | 0.0% | 0–0.8% |
| goal | 349 | 1 | 0.3% | 0.1–1.6% |
| object | 351 | 0 | 0.0% | 0–1.1% |
| libero_10 | 236 | 0 | 0.0% | 0–1.6% |

How the 1,417 episodes end: 60% at the first turn with no executable action, 23% out of steps, 17% before the step limit later in the episode, 0.1% success (median 2 turns). The sweep ends about 05:30 Nov 24, with libero_10 0–24 last.

## Next step

After the presentation, tally the finished 9B baseline (`python3 baseline/tally.py runs/baseline-bf9b274/Qwen3.5-9B runs/baseline-47200a9/Qwen3.5-9B`) for the presentation (`baseline/tally.py`), flagging libero_10 and object 0–24 as partial.

## Blocked on

- **OPD, the verl mini-batch bug:** the human chose a verl patch shipped from this repo, plus an upstream report to PEAR (09:20). The earlier diagnosis was incomplete: `train_mini_batch` already repairs the whole batch before selecting, and on CPU that repair holds, even through a consolidated pickle. So something in the GPU path undoes it. The candidate fix repairs the batch before every selection, inside verl's iterator. Smoke part 3 runs it from a patched copy of 47200a9 (`code/47200a9-mbfix`, `VERL_CHANGE_IN_PROGRESS=1`), with a probe that logs `_ragged_idx` before each selection (60517692). Next: ship the patch to `main` and open the PEAR report.
- **Before OPD:** the run settings (K, B, lr, validation set); `max_off_policy_threshold=1`.

## Dead ends
