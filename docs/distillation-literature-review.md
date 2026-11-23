# Literature review: distilling a robot-use reasoning model (2026-11-22)

**Project.** On-policy distillation (OPD) of PEAR's 27B robot-use reasoning model into Qwen3.5-9B and 4B students, to
get a small, fast robot-use agent. The pipeline is on `main`. This review asks four questions:
1. What is the closest prior work?
2. Which of our three novelty claims survive?
3. What does the literature say will go wrong?
4. What should change in the pipeline?

**How sources were checked.** Each entry is marked:
- **[P]**: read in the paper text (HTML).
- **[A]**: read on the arXiv abstract page only.
- **[S]**: a search-result snippet or secondary summary only.

Nothing marked [S] should be cited without re-checking. Finding no prior work in a search is not proof that none
exists, and this field moves weekly: about 300 OPD papers in 2026 alone, per the curated list in §8.

---

## 0. Bottom line

| Claim in our pitch | Verdict | Closest prior work |
|---|---|---|
| **1. First distilled robot-use LLM** | **Holds, narrowly, for now.** No distillation of PEAR or of any other native robot-use LLM was found: one that reasons in language and outputs an end-effector pose with no harness. | EmbodyAgent (19 Nov 2026): a frontier teacher distilled by SFT into a 9B VLM that controls a 7-DoF arm on LIBERO-Spatial, through code. Also AgenticRoboRL (9B planner with a harness) and OPD-AVMP (an LLM motion planner for driving). |
| **2. First OPD over multi-round robot episodes** | **Does not hold as worded.** | VLA-OPD (Mar 2026) already runs reverse-KL OPD on LIBERO with closed-loop, multi-step student episodes. Multi-teacher flow-VLA OPD and REPAIR are on-policy too. Multi-turn agent OPD is a crowded area (§3), and PIVOT applies it to embodied VLM agents. What remains new is OPD over **long reasoning episodes**: about 20 rounds and about 94k tokens, where every earlier round's reasoning stays in context. |
| **3. Fewer reasoning tokens per action** | **The general mechanism is published. The robot-episode setting is open, but our pipeline does not do this yet.** | COPD (Qwen team, Jul 2026) is OPD whose per-token advantage is the teacher's log-prob gap between a light-thinking and a heavy-thinking instruction. It cuts length by 56% while raising accuracy, on multimodal benchmarks, and it is exactly the "cross-effort distillation" I proposed earlier. CRISP/OPSDC does the same as self-distillation. COPD also shows **vanilla OPD does not shorten reasoning** (2.12k to 2.23k tokens). |

**Recommended framing.** Our method should be the first to distill a robot-use reasoning model, with reasoning
compression that exploits the multi-round structure. Make it the method contribution, not plain OPD. Robot episodes
differ from the single-turn settings of COPD and CRISP in three ways we can measure:
- **Shorter reasoning compounds.** Every round's reasoning stays in context, so cutting tokens per round also shrinks
  every later prefill and the KV cache. Total cost grows roughly quadratically with tokens per round, not linearly.
- **Context exhaustion is a failure mode.** 19% (Goal) and 33% (Long) of the LIBERO-only teacher's episodes hit the
  256k context budget (PEAR App. B [P]). A shorter-reasoning student could **succeed more** than its teacher, not only
  run faster.
- **The supervision signal is physical.** Each round ends in an executed action and a new observation, so turn-level
  credit can come from simulator outcomes (cf. OG-OPD, OVCSD in §3), not only from teacher log-probs.

---

## 1. PEAR facts that affect the project (all [P], arXiv 2611.18562 HTML)

- **The 74.4% is the LIBERO-only checkpoint** (step 225, `Qwen3.8-27B-PEAR-LIBERO_only`), which is what the repo
  uses. The final `Qwen3.8-27B-PEAR` was trained further on RoboDojo, and its LIBERO score is not reported. Our pitch
  says "Qwen3.8-27B-PEAR (74.4%)", which should be corrected to the LIBERO-only checkpoint.
- **Effort and success:** xhigh 74.4% (the training effort), low 21.9% (mean of 2 runs). Medium is not reported for
  PEAR.
- **Thinking per round:** 3.23k tokens at xhigh, 0.86k at low.
- **Latency.** The paper says "a round of xhigh reasoning takes about 4.3 s" on a GB300 (747 tok/s), and 1.1 s at
  low effort.
  - It gives no per-round time for the RTX 5090, only 221 tok/s. That implies about 14.6 s per round at 3.23k tokens.
  - **Our pitch's 2.8 s (GB300) and 9.5 s (5090) don't match the paper.** They correspond to about 2.1k tokens per
    round, which matches our own 14 recorded episodes on the full `-PEAR` checkpoint, not the paper's figure.
  - Either cite the paper (4.3 s and about 14.6 s) or label ours as measured.
- **Episodes:** RL episodes average 19.7 rounds and about 94k tokens.
- **Context budget:** 256k tokens on LIBERO (Table 1), 160k on RoboDojo; evaluation uses 260,096.
  - **Check:** the collection job log says "PEAR's protocol, 160k-token responses", and `configs/step/sft.yaml` uses
    `max_length: 163840` with `truncation: error`.
  - If LIBERO is evaluated at 256k, collecting at 160k will lower the teacher's success below 74.4%. Episodes longer
    than about 164k would also fail SFT.
- **LIBERO-Pro:** 57.4% is the *Task* perturbation (a new instruction in a training scene). The mean over the four
  perturbations is 62.1%, against π0.5's 0.75% on Task.
- **PEAR's own future work:**
  - "Shortening the reasoning without giving up success" is left open, possibly by rewarding shorter chains.
  - It does not mention distillation or smaller models.
  - It calls the 27B base "relatively small" and notes that a drafter tuned to PEAR would decode faster.
  - Claim 3 is the gap the authors themselves name. Expect them, or others, to attack it soon.

## 2. On-policy distillation: foundations

| Work | What matters for us |
|---|---|
| **MiniLLM** (Gu et al., 2023, [2306.08543](https://arxiv.org/abs/2306.08543)) [S] | Reverse KL on student samples for generative LMs; named the field. |
| **GKD** (Agarwal et al., 2023, [2306.13649](https://arxiv.org/abs/2306.13649)) [S] | On-/off-policy mixtures and general divergences. The standard baseline name. |
| **Thinking Machines, "On-Policy Distillation"** (Lu, Oct 2025, [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)) [P] | **Exactly our step 3:** per-token reverse KL as advantage, zero discount, no task reward, importance-sampled RL loss. Qwen3-8B-Base from an SFT checkpoint goes from 60% to 70% AIME'24 in about 150 steps, against 68% for RL. It estimates a 9–30× compute saving over more SFT. **It does not evaluate multi-turn or agents.** |
| **Qwen3 technical report** ([2505.09388](https://arxiv.org/abs/2505.09388)) [S] | Strong-to-weak OPD in production for small Qwen3 models: 74.4% AIME at about 1.8k GPU-hours, against 17.9k for RL (as cited by the Thinking Machines post). |
| **ExOPD** ([2602.12125](https://arxiv.org/abs/2602.12125)) [S] | OPD as dense KL-constrained RL, extrapolating the reward past the teacher. COPD uses it as its strongest baseline. |
| **Train4Merge** ([2609.32303](https://arxiv.org/abs/2609.32303)) [S] | RL-trained teachers guide OPD students better than equally strong SFT teachers, especially on agentic tasks. Good news for us, since PEAR is RL-trained. |
| **A Survey of OPD for LLMs** ([2604.00626](https://arxiv.org/abs/2604.00626)) [S] | Survey; cite it for the taxonomy. |
| **verl async OPD** ([docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)) [S] | Our trainer. It trades strict on-policy guarantees for throughput, which is the `max_off_policy_threshold` gap already noted in the experiment log. |

## 3. Multi-turn and agentic OPD (closest on method)

Multi-turn OPD is now a crowded subfield. Almost all of it uses text or GUI environments: ALFWorld, WebShop,
ScienceWorld, search and GUI agents. **None of it uses robot control with native pose actions.** A reviewer will expect
the strongest of these as baselines or related work.

| Work | Relevance |
|---|---|
| **TCOD** ([2604.24005](https://arxiv.org/abs/2604.24005), Apr 2026) [A] | **Key risk.** Trajectory-level KL grows with turn depth: compounding errors push the student off the teacher's support, so deep-turn supervision becomes unreliable. A short-to-long depth curriculum gives up to +18 points over vanilla OPD and can beat the teacher. Our episodes are about 20 rounds, so expect this. |
| **Guided-OPD** ([2606.15912](https://arxiv.org/abs/2606.15912)) [S] | Mixes teacher and student turns, with reverse KL on student turns and forward KL on teacher turns, under a decaying intervention schedule. |
| **Multi-Turn OPD with Prefix Replay / ReOPD** ([2607.04763](https://arxiv.org/abs/2607.04763)) [S] | Replays teacher prefixes to avoid fresh environment rollouts. Relevant because our simulator rollouts are expensive. |
| **OG-OPD, "Teacher–Student Gaps Are Not Enough"** ([2609.35319](https://arxiv.org/abs/2609.35319)) [S] | Reweights each turn's supervision by task outcomes from paired student continuations. The outcome-guided variant we would compare against. |
| **Act First, Reason Later** ([2609.36608](https://arxiv.org/abs/2609.36608), Sep 2026) [A] | Speeds up multi-turn OPD by acting via reference-conditioned inverse dynamics and generating the reasoning asynchronously; 1.8–4.9× faster training with Qwen3 0.6–4B students. A rollout-cost idea, not inference speed. |
| **Physics of Multi-Turn Long-Horizon Planning** ([2607.24720](https://arxiv.org/abs/2607.24720)) [S] | OPD has a broader effective region than GRPO for long-horizon planning. It also separates policy-gradient-style OPD (ours) from GKD-style OPD. |
| **SOD** ([2605.07725](https://arxiv.org/abs/2605.07725)), **TurnOPD** ([2607.05804](https://arxiv.org/abs/2607.05804)), **DASH-OPD** ([2607.29078](https://arxiv.org/abs/2607.29078)), **Revisiting DAgger** ([2605.12913](https://arxiv.org/abs/2605.12913)) [S] | Step- or turn-level reweighting and rollout mixing for small agent students. |
| **PIVOT** ([2609.35303](https://arxiv.org/abs/2609.35303), Sep 2026) [A] | Pivot-aware OPSD plus GRPO for **multi-turn VLM agents on embodied and navigation tasks**. Students are Qwen2.5-VL-3B and Qwen3-VL-2B. The closest multi-turn multimodal agent work, but it is self-distillation, not robot control. |
| **OVCSD** ([2607.27937](https://arxiv.org/abs/2607.27937)), **Look Ahead Before You Distill** ([2608.01953](https://arxiv.org/abs/2608.01953)) [S] | Distill only teacher signals that environment-verified continuations support. |

## 4. Reasoning compression by distillation (closest to claim 3)

| Work | Relevance |
|---|---|
| **COPD, Contrastive On-Policy Distillation** ([2607.19046](https://arxiv.org/abs/2607.19046), Jul 2026; authors include Shuai Bai and Dayiheng Liu, Qwen) [P, main results] | **The closest method.** A frozen teacher scores the student's rollout under a low-thinking and a high-thinking instruction, and the advantage is clip(log π(y_t \| x^LT) − log π(y_t \| x^HT), ±10). There is no task reward, KL penalty or length regularizer. The teacher is external and larger (Qwen3-VL-8B/32B, Qwen3.5-4B, Qwen3.5-35A3B), the student is 2B, and there is a 9B ablation. Qwen3-VL-8B → 2B: **COPD 57.0% at 0.91k tokens, against vanilla OPD 53.5% at 2.23k and ExOPD 55.2% at 2.08k.** It costs 60 GPU-hours against 132 for OPD. A self-distillation variant (COPSD) also works. Single-turn multimodal QA, not agents. |
| **CRISP / OPSDC** ([2603.05433](https://arxiv.org/abs/2603.05433), Mar 2026) [A] | Self-distillation: the same model with a "be concise" instruction is the teacher, with reverse KL on the student's own rollouts. Qwen3-14B: −56% length on MATH-500 with accuracy up to +3.3, and −38% on AIME'24 with accuracy within 1 point. It compresses easy problems more than hard ones. |
| **OPSD Compresses What RLVR Teaches** ([2605.06188](https://arxiv.org/abs/2605.06188)) [S] | SFT → RLVR → OPSD as a post-RL compaction stage. It maps onto PEAR, which is an RL model with long reasoning. |
| **BIRD** ([2607.15736](https://arxiv.org/abs/2607.15736)), **Reasoning Compression with Mixed-Policy Distillation** ([2605.08776](https://arxiv.org/abs/2605.08776)), **ThinkOPD** ([2609.37044](https://arxiv.org/abs/2609.37044)), **Masked Distillation** ([2607.22629](https://arxiv.org/abs/2607.22629)) [S] | Other compression variants: bootstrapped SFT, teacher rewrites of student traces, think-trace-conditioned teachers, and answer-only rollouts. |
| **Rethinking OPSD for Thinking Models** ([2607.05184](https://arxiv.org/abs/2607.05184)); **Why Does Self-Distillation (Sometimes) Degrade Reasoning?** ([2603.24472](https://arxiv.org/abs/2603.24472)) [S] | **Cautions.** Privileged self-distillation can degrade long reasoning, by suppressing epistemic verbalization. Relevant if compression removes the spatial self-checks PEAR's success relies on. |
| **ECoT-Lite** (Chen et al., CoRL 2025, [2505.08243](https://arxiv.org/abs/2505.08243)) [S] | The VLA precedent for cheaper embodied reasoning. Reasoning dropout trains with reasoning but acts without it: about 3× faster, LIBERO-90 about 76% against 77% for full ECoT. Reasoning pretraining helps more on Bridge. A baseline family for "fewer tokens". |
| **Fast ECoT** ([2506.07639](https://arxiv.org/abs/2506.07639)) [S] | Reuses reasoning across timesteps. Note that our own cross-round reuse oracle gave at most +1.5% (`experiments/pear_specdec`). |

**What is not yet done (as far as found):** reasoning compression inside **multi-round closed-loop episodes**, where
the compressed reasoning of earlier rounds also forms the context of later rounds, and where success is measured in a
simulator rather than as answer accuracy.

## 5. Distillation in robotics and VLAs

| Work | Relevance |
|---|---|
| **VLA-OPD** ([2603.26666](https://arxiv.org/abs/2603.26666), Mar 2026) [A] | Reverse-KL OPD for VLAs on **LIBERO** and RoboTwin2.0, with an expert teacher giving dense token-level supervision on the student's own trajectories. It argues forward KL causes entropy explosion and hard CE causes collapse. **This defeats claim 2 as worded.** It distills action tokens, not language reasoning. |
| **Multi-Teacher OPD for Flow-Based VLAs** ([2610.21522](https://arxiv.org/abs/2610.21522), Oct 2026) [A] | Specialist VLAs supervise one student along its own action-flow trajectories. LIBERO-40 +22.6 points over the base, and +6.4–9.5 points over action-loss OPD and teacher-rollout SFT. Evidence that OPD beats teacher-rollout SFT in robotics, which supports our SFT-vs-OPD hypothesis. No language reasoning. |
| **REPAIR, Beyond Policy Patches** ([2611.02777](https://arxiv.org/abs/2611.02777), Nov 2026) [A] | Value-weighted distillation of RL specialists into a flow VLA: offline initialization, then on-policy refinement. 6 real tasks at 100%. |
| **FastOPD** ([2610.02832](https://arxiv.org/abs/2610.02832)), **DriftOPD** ([2610.00317](https://arxiv.org/abs/2610.00317)), **RPD** ([2503.05833](https://arxiv.org/abs/2503.05833)), **ROAD-VLA** ([2606.25800](https://arxiv.org/abs/2606.25800)) [S] | Flow- and one-step VLA compression and RL-guided distillation. Action-only. |
| **OPD-AVMP** ([2604.07944](https://arxiv.org/abs/2604.07944), Apr 2026) [A] | **The closest "LLM controls a vehicle" analog.** GKD distillation of a GPT-Driver-style CoT motion-planning LLM into a 5× smaller student on nuScenes, close to the teacher. Open-loop and single-shot, with unnamed models. |
| **EmbodyAgent** ([2611.26196](https://arxiv.org/abs/2611.26196), 19 Nov 2026) [A] | **The closest to claim 1, and posted six days after PEAR.** A black-box frontier teacher (think-then-code), success-filtered SFT into an open 9B VLM, then GRPO. LIBERO-Spatial held-out 0 → 14 (SFT) → 84/100 (RL), on 100 test states rather than the full protocol. Differences from us: code-as-policy through a tool, off-policy SFT plus RL rather than OPD, no teacher logits, one suite. |
| **AgenticRoboRL** ([2610.28589](https://arxiv.org/abs/2610.28589), Oct 2026) [A] | A 9B robot planner trained by SFT plus multi-turn RL, using VLA skills and code harnesses. LIBERO-Pro 18.1% → 71.0%, against Codex (GPT-5.5) at 72.1%. Not distillation of a robot-use LLM. |
| **CoTinyVLA** ([2607.25487](https://arxiv.org/abs/2607.25487), Jul 2026) [A] | CoT distilled from a 35B teacher into a 0.9B VLA (Qwen3.5-0.8B backbone). LIBERO-Plus 80.7–90.8%. A VLA with an action head, not a reasoning agent; whether the distillation is offline is not stated. |
| **LeAct** ([2607.21856](https://arxiv.org/abs/2607.21856)) [S] | Selects student reasoning by how well it recovers expert actions. A possible action-grounded signal for compression. |
| **"From Reasoning to Action"** ([2610.30051](https://arxiv.org/abs/2610.30051), Oct 2026) [A] | Benchmarks frontier models as robot-use agents on LIBERO-10: GPT-6 Astra at most 44%, Gemini-4 Argon 43%, with higher effort generally better. Context for "small open model vs frontier". |

## 6. Known OPD failure modes we should expect

1. **Length inflation and termination mismatch.**
   - "When EOS Tokens Disagree" ([2609.20511](https://arxiv.org/abs/2609.20511), [A]): base students and post-trained
     teachers put their stopping mass on different EOS tokens. OPD then suppresses the student's stop without
     transferring the teacher's, and responses run to the budget. The fix treats equivalent EOS tokens as one stopping
     action.
   - Also "Demystifying OPD — Length Inflation" ([2604.08527](https://arxiv.org/abs/2604.08527)) and "Demystifying OPD
     — Roles, Pathologies" ([2607.13399](https://arxiv.org/abs/2607.13399)) [S].
   - **Our smoke test already shows this pattern:** the base Qwen3.5-4B "reasons until the 4,096-token cap without
     acting". In PEAR the stop points that matter are `</think>`, the action fence, and end of turn.
   - Check that the teacher and the Qwen3.5 student agree on all three before running OPD.
2. **KL growing with turn depth** (TCOD). Expect deeper rounds to get less reliable supervision. A depth curriculum or
   turn-level mixing is a ready ablation.
3. **Vanilla OPD doesn't shorten reasoning** (COPD's tables). If we want claim 3, length needs an explicit signal.
4. **Compression can remove the reasoning that matters** ([2607.05184](https://arxiv.org/abs/2607.05184),
   [2603.24472](https://arxiv.org/abs/2603.24472)). Our earlier anytime-act study found that naive truncation of
   PEAR's reasoning failed.

## 7. Implications for the pipeline

1. **Fix the pitch:**
   - The teacher is `-LIBERO_only`.
   - Use the paper's latency figures or our measured ones, labeled as such.
   - Narrow claim 2 to OPD of a **reasoning** robot-use agent over long multi-round episodes.
2. **Check the context budget:** 256k on LIBERO against the 160k used in collection and SFT (§1).
3. **Make compression the method.**
   - A COPD-style contrastive advantage across PEAR's own effort levels: the teacher scores the student under xhigh
     and low instructions, and PEAR's chat template already supports both.
   - Extend it to multi-round episodes with turn-level, simulator-grounded credit. That is where the novelty is.
   - Plain OPD stays as the baseline it already is.
4. **Baselines a reviewer will ask for:**
   - teacher at low and at xhigh;
   - base student;
   - SFT;
   - vanilla OPD (Thinking Machines/verl);
   - COPD-style compression applied naively per round;
   - ECoT-Lite-style reasoning dropout;
   - optionally TCOD or Guided-OPD for multi-turn stability.
5. **Report tokens per round, rounds per episode and wall-clock per episode** for every model. The main plot is
   success against time per episode.
6. **Move fast.** PEAR is 9 days old, EmbodyAgent appeared 6 days after it, and PEAR names reasoning shortening as its
   own future work.

## 8. Index used

- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation): read in full on
  2026-11-22. Its robotics/embodied, multimodal, agent and compression sections are the source of most [S] entries
  above.
- [PEAR paper](https://arxiv.org/html/2611.18562), [repo](https://github.com/mit-han-lab/pear) and
  [blog](https://mit-han-lab.github.io/pear-blog/).
- Earlier reviews in this repo: `docs/literature-review.md` covers VLA quantization only, and
  `experiments/pear_specdec/LITERATURE_REVIEW.md` (branch `itsjenniepan/experiments`) covers PEAR inference and
  speculative decoding.
