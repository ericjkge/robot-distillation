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

*Updated 2026-11-23 23:45. The full history of fixes, submissions and earlier notes is in git, up to commit e1643d7.*

**Now running or queued**

| Step | Jobs | State |
| --- | --- | --- |
| Teacher episodes | spatial and object done (300, 250 kept). Running (old per-suite script): goal 5–9 60576751, object 0–4 60576883, object 5–9 60576884. The rest (goal 0–4, libero_10) as 30 units (task × trials 0–24 / 25–49) shared by 10 workers, 60623284–60623300 and 60630347 (added when the baseline freed a job slot), each a TP2 job on `gpu,gpu_quad,gpu_requeue` excluding the 11 RTX 8000 nodes | at 12:56: object 219 of 250 kept (tasks 4, 6, 9 done, the rest at 17–23; done about 14:00), goal 5–9 38 of 125 (tasks 5 and 9 at 2 and 3 kept; they run to 50 trials, about 2 more days). A goal or libero_10 episode takes about an hour at xhigh, 4 at a time per job, and the queue, not the episodes, was the bottleneck: per-task jobs were estimated to start as late as Nov 29. So at 10:40 every queued job became a worker that claims the next unit not yet claimed by a running job and keeps its GPUs until no unit is left (`collect/collect.sbatch`, UNITS). At Nov 25 11:17 the first workers are estimated Nov 29 13:40: every node of the three partitions is allocated or held for higher-priority jobs. Shorter time limits (4, 8, 12 h) did not move the estimates, and no account with more fairshare exists. So from 13:30 the three running jobs relay: each, once its suite is done, stops its run and works through the same 30 units in its own allocation until its limit (Nov 26 04:09–07:59), rather than ending (`enough.py`, the list in `runs/collect-bf9b274/relay`). Object 5–9 (60576884) relayed at 14:14: libero_10 task 0 trials 0–24, task 0 done (25 kept) by 17:39; 91 kept by Nov 25 14:39 (tasks 1–4: 25, 20, 0 of 7, 21 of 25; task 3 episodes take about 3 h). Object done (250) by 17:39; 60576883 relayed too (libero_10 task 1). Their run then exits 143, so SFT waits on them with afterany, the workers still with afterok |
| Base baseline 9B, each suite split in two (initial states 0–24 / 25–49; tally: `baseline/tally.py`) | goal 60512338 / 60512340, spatial 60512341 / 60486049, object 60512117 / 60486072, libero_10 60512116 / 60486085 | done Nov 24 11:47: 2 successes in 2,000 (see Results) |
| Smoke part 3 (distill, export, eval) | 60517692 (any 4 GPUs of 48 GB, was L40S only; estimated start Nov 24 04:36) | candidate verl fix with a probe (see Blocked on) |
| Teacher episodes, goal 5, 7, 9 (ours to finish; human, Nov 25 18:40) | 60576751 (until Nov 26 04:09; at 19:15 relayed off its per-suite run, which spent GPUs on done tasks 6 and 8, to units goal 7 → 5 → 9: `relay` line 3, `enough.py`); workers 60796702, 60796703 (`gpu_quad,gpu_requeue`, units goal 7 → 5 → 9, all trials not yet run) | 60576751 running goal 5 (20:38: goal 5 3/25 kept/trials, 7 done 28/28, 9 9/26); workers queued |
| SFT 9B | 60783661 (`gpu_quad,gpu_requeue`) | no dependency, so it gains age priority; at start it waits, polling every 10 min, until every task is complete on the Hub (`collect/complete.py`), then draws the set (`sft/build_set.sh`); checkpoints every 10 steps |
| Publish SFT to the Hub (`sft/publish.sbatch`: private `eddykang06/qwen3.5-9b-pear-libero-sft`, in the robot-distillation collection with the rollouts; human, Nov 25) | 60789893 | afterok:60783661 |
| OPD 9B, from the SFT checkpoint | 60576893 | afterok:60783661; runs until stopped by hand |

The resumed halves run `main` 47200a9 (adds the worker-port fix); spatial 25–49, object 25–49, libero_10 25–49, collection and SFT run bf9b274.

- Teacher rollouts first (Nov 24 10:30): every suite's rollouts must finish as soon as possible, because a teammate is waiting on them for a separate SFT run.
- No other account (Nov 24 13:10): do everything to finish the teacher rollouts and SFT as soon as possible.
- Split with a teammate (Nov 25 12:40): our jobs finish libero_10; the teammate collects libero_goal tasks 0–4 (episode list `runs/collect-bf9b274/handoff/libero_goal-tasks-0-4.json`, the trials no run has ended), under the same rule (25 kept or 50 trials). Our jobs skip those units: their claims name 60162454 (vla-agent-host), so delete the claim files to take them back. A trials 0–24 budget (~775 SFT episodes, ~Nov 26) was considered and declined.
- No early stop (Nov 25 13:00): the human declined giving up on a task after 15 trials without a success; every task runs to 25 kept or 50 trials.
- Teammate also takes libero_10 tasks 5–9 (Nov 25 12:35; claims name 60162454). Our libero_10 task 3 run was stopped by mistake at 12:29 and took relayed job 60576883 down with it: killing the run's process group from an `srun --overlap` step ended the job step that holds the relay. 60576883 was requeued (pending). Task 3 (0 kept of 7) stays ours: 60576884 reaches trials 25–49 after task 4; trials 0–24 wait for 60576883 or a worker. Lesson: never signal a relayed run from outside; let `enough.py` end it.
- Teammate also takes libero_10 task 3 (Nov 25 12:40). Ours: goal 5–9 (60576751), libero_10 4 and 1–2 trials 25–49 (60576884). The ten queued workers and requeued 60576883 were cancelled; SFT 60576892 depends afterany on 60576751 and 60576884 and is held until the teammate's episodes (goal 0–4, libero_10 3 and 5–9) are merged.
- Teacher rollouts go to the Hub dataset `eddykang06/qwen3.8-27b-pear-libero-rollouts` (public since Nov 25, human), shared with the teammate's cluster: `<suite>/records/<source>.jsonl` (every episode run), `<suite>/episodes/<uid>.pt` (successes), `index.jsonl` and the card, staged by `collect/hub.py` under `$OUT/hub/`. Synced hourly from the running jobs (`collect/pull.py` first, so the index covers every cluster); at Nov 25 16:45: 923 episode runs, 792 successful; libero_10 0–2 and 4 done (100 episodes); at 18:40: 937 runs, 803 successful. `opd.select_episodes` reads a Hub download as a source (main 6005c4f, Nov 25, at the human's request).
**Human decisions**
- Brief revised (origin/eddy cff8cb4, seen Nov 25 20:38): Algorithm 1 reformatted and typos fixed; the method is unchanged, so no plan change.
- No 4B runs: the 4B baseline and SFT are dropped (17:10).
- Students use the teacher's chat template (reasoning effort in the system prompt; every round's reasoning kept).
- SFT keeps its place in the queue but trains only on complete rollouts (Nov 25): a job that starts early waits for every cluster's episodes.
- SFT: 25 successful episodes per task, drawn at random (all of a task's if it has fewer); lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs.
- Base baselines: full LIBERO (2,000 episodes, PEAR's protocol), xhigh, LIBERO only.
- As fast as possible (17:15): one finished 9B run, SFT then OPD; no OPD from the untuned student. So collection moved to 2 GPUs (TP2, 4 episodes at once), which start far sooner than 4, each suite split in two by task; the 4-GPU jobs (60512543, 60512544, 60512585; Slurm estimated Nov 26–27) were cancelled. OPD will use `max_off_policy_threshold=1` (Algorithm 1).
- Deadline (17:55): OPD finished before Nov 29; the final eval is not needed by then. SFT waits for every suite's collection. No partitions that cost priority yet (18:05). Then (22:10): collection may use `gpu` (costs priority), after the estimated cost (about 200 billing an hour for about 15 h; usage halves every 6 h); SFT and OPD stay on `gpu_requeue`. `research/bin/submit` allows only `gpu_requeue`, so the jobs were submitted there and then given `gpu` too with `scontrol update`, at the human's instruction. OPD runs open-ended; the human decides when to stop it. Keep re-optimizing for the earliest finish as complications arise (18:00).
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
- Collection TP2, code bf9b274 (`collect/collect.sbatch` from fae6d88 on; `TASKS` selects the tasks): gpu:2,vram:48G, 2 days, 11 CPUs (the `gpu` QOS allows 34 CPUs and 420G per user: three jobs at once), 128G; 4 episodes at once, 4 simulators; partition then updated to `gpu,gpu_requeue`. Same `runs/collect-bf9b274`.
- Baselines, code bf9b274 (`baseline/eval.sbatch`, `TRIALS` selects the initial states): gpu:1,vram:45G, 3 days, 10 CPUs, 64G (35 GB measured); 8 episodes at once. The resumed halves (60512116, 60512117, 60512338, 60512340, 60512341) run code 47200a9 with the same header.
- Collection workers, code bf9b274 (`collect/collect.sbatch`, `UNITS=libero_goal:7:0-49 libero_goal:5:0-49 libero_goal:9:0-49`): gpu:2,vram:48G, 2 days, 11 CPUs, 128G, `--requeue`; partition then `gpu_quad,gpu_requeue`; 60796702, 60796703.
- SFT, code bf9b274 (`sft/run.sbatch`): gpu:4,vram:48G, 2 days, 20 CPUs, 450G, `--requeue`; partition then `gpu_quad,gpu_requeue`; no dependency; the set drawn at start by `sft/build_set.sh` (main 6005c4f); `trainer.save_freq=10`; now 60783661.
- Smoke part 3, code 47200a9 (`smoke/part3.sbatch`): gpu:l40s:4, 1h15, 16 CPUs, 192G.
- Publish SFT (`sft/publish.sbatch`): short, 4 h, 4 CPUs, 16G, afterok:60783661; 60789893.
- OPD, code `code/47200a9-mbfix` (`opd/run.sbatch`): gpu:8,vram:48G (A6000), 5 days, 20 CPUs, 500G; 4 trainer, 2 rollout (8 episodes each), 2 teacher (TP2) GPUs; batch 64, `max_off_policy_threshold=1`, validation every 10 updates on PEAR's LIBERO-Goal validation episodes; checkpoints every update; now 60576893, afterok:60576892.

## Results

- **Base Qwen3.5-9B, LIBERO, xhigh, PEAR's protocol (final, Nov 24 11:47):** 2 successes in 2,000 episodes (both libero_goal); LIBERO mean 0.1% (spatial, object, libero_10 0.0%, each 0.0–0.8% at 95%; goal 0.4%, 0.1–1.4%). Teacher: 74.4% (paper, xhigh).
- **Base Qwen3.5-9B, LIBERO, xhigh, PEAR's protocol (for the presentation, 14:58; partial):** 1 success in 1,417 of 2,000 episodes, covering all 40 tasks; LIBERO mean 0.1%. Compare the teacher's 74.4% (paper, xhigh).

| Suite | Episodes | Successes | Rate | Wilson 95% |
| --- | --- | --- | --- | --- |
| spatial | 481 | 0 | 0.0% | 0–0.8% |
| goal | 349 | 1 | 0.3% | 0.1–1.6% |
| object | 351 | 0 | 0.0% | 0–1.1% |
| libero_10 | 236 | 0 | 0.0% | 0–1.6% |

How the 1,417 episodes end: 60% at the first turn with no executable action, 23% out of steps, 17% before the step limit later in the episode, 0.1% success (median 2 turns). The sweep ends about 05:30 Nov 24, with libero_10 0–24 last.

## Next step

Watch goal 5, 7 and 9 to completion (60576751, then workers 60796702–03); SFT 60783661 waits until every task is complete on the Hub, the teammate's (`<suite>/records/<cluster>.jsonl` and episodes, not `index.jsonl` or the card) included.

## Blocked on

- **SFT and OPD start late (Nov 25 17:00):** Slurm estimates, from our priority, put SFT (4×48 GB) at Nov 29–30 and OPD (8 GPUs) at Nov 30, past the Nov 29 deadline. `gpu` is full until January and caps us at 420G; smaller shapes estimate the same. The human allowed other partitions at a modest priority cost (Nov 25): SFT was resubmitted on `gpu_quad,gpu_requeue` without a dependency to gain age priority (60783661; 60576892 cancelled); OPD's 5 days are refused on `gpu_quad`. More priority is the human's lever.

- **Nov 29 (asked 09:17 Nov 24; the human answered 10:30 that the teacher rollouts come first, see Human decisions):** at the current pace OPD cannot finish before Nov 29. Goal and libero_10 teacher episodes take about an hour each at xhigh, and many end in failure, so tasks run to the 50-trial cap: goal 5–9 alone needs about 130 more trials, about 54 h at 2.4 an hour. Libero_10's jobs are estimated to start Nov 27 08:50 – Nov 28 19:40 (10:17). Object will be done today. The levers are the human's:
  - SFT starts at a cutoff with whatever has been kept, instead of waiting for every suite.
  - Each task stops after fewer trials (e.g. 30 instead of 50), which cuts most of the failing tasks' time.
- **OPD, the verl mini-batch bug:** the human chose a verl patch shipped from this repo, plus an upstream report to PEAR (09:20). The earlier diagnosis was incomplete: `train_mini_batch` already repairs the whole batch before selecting, and on CPU that repair holds, even through a consolidated pickle. So something in the GPU path undoes it. The candidate fix repairs the batch before every selection, inside verl's iterator. Smoke part 3 runs it from a patched copy of 47200a9 (`code/47200a9-mbfix`, `VERL_CHANGE_IN_PROGRESS=1`), with a probe that logs `_ragged_idx` before each selection (60517692). Next: ship the patch to `main` and open the PEAR report.
- **OPD's start:** the verl fix is unconfirmed on GPU (smoke 60517692). If OPD hits the bug, it fails at its second update and is resubmitted with the fix that smoke settles.

## Dead ends
