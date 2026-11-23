# Reasoning compression for distilled robot-use agents (2026-11-23)

**Owner:** Jennie. **Context:** Eddy runs the teacher rollouts → SFT pipeline; Sophie runs vanilla OPD (untuned and SFT
students). This direction is the method contribution: a student that reasons in fewer tokens per action than the
teacher at matched success. Literature: `../../docs/distillation-literature-review.md`.

## What we already know (prior PEAR studies in this repo)
- Truncating a round's reasoning hurts closed-loop success: PEAR's action converges late (~15% of tokens after the
  provisional action is within 1 cm), and imprecise moves are paid back in correction rounds (`../pear_anytime_act`).
- Dropping earlier rounds' thinking shifts actions (+0.016 nats/token). Hand-made selectors (geometry, grounded claims)
  do not beat random sentences at the same budget (`../pear_context_exhaustion`, `../pear_anytime_act` §9).
- So the redundancy, if any, is not findable by heuristics. The candidate learned signal is COPD's: the teacher's own
  log-prob gap between a light-thinking (effort=low) and a heavy-thinking (effort=xhigh) instruction.

## X1 (offline, teacher-forced): does the effort contrast point at reasoning the action does not need?
Pre-registered 2026-11-23, before any data. Script `pod/score_effort.py`, analysis `scripts/analyze_effort.py`.

- **Teacher:** Qwen3.8-27B-PEAR-LIBERO_only (the distillation teacher), BF16, SGLang, 1× H200.
- **Data:** 14 recorded PEAR episodes (`../pear_specdec/results/dflash2_main`), every 3rd round.
- **Signal:** gap_t = log p_low(y_t) − log p_xhigh(y_t) on round r's thinking tokens (COPD's advantage). Round r's
  thinking is cut into K ≈ equal sentence chunks (K = clamp(chars/300, 4, 16)); chunk score = mean gap.
- **Outcome (lesson from earlier studies: score actions, not thinking):** action-token NLL under xhigh after deleting
  chunks covering ~30% (and ~60%) of the round's thinking characters, chosen by
  `copd` (most negative gap first), `anti` (most positive), `rand0/rand1`, `oracle` (smallest single-deletion action
  dNLL), `tail` (drop the end: early-exit analogue). dNLL is relative to the undeleted round on the same cached path.
- **Statistic:** per-round mean action dNLL, averaged over rounds; 95% CI by episode bootstrap (2,000 resamples);
  paired differences per round.

**GO (effort contrast localizes removable reasoning) — all of:**
1. at 30%: dNLL(copd) ≤ 0.5 × dNLL(rand), with the paired rand − copd CI above 0;
2. dNLL(copd) < dNLL(anti), paired CI above 0;
3. reported: how close copd is to oracle, i.e. (rand − copd)/(rand − oracle).

**NO-GO:** rand − copd CI includes 0. Then COPD in the robot setting acts as generic length pressure; the method needs
an action-grounded signal (the oracle map's direction), not the effort contrast.

**Also recorded:** action NLL under low vs xhigh with the same reasoning (does the effort tag change actions by itself);
Spearman correlation between chunk gap and single-deletion necessity.

**Caveats:** teacher forcing on edited reasoning is off-distribution; it ranks selectors, it is not closed-loop success.
The recorded episodes come from the sibling full PEAR checkpoint, not from LIBERO_only.

## X2 (offline, sampled): the teacher's per-round effort frontier
Pre-registered 2026-11-23, before any data. `pod/sample_effort.py`, `scripts/analyze_frontier.py`.
- Same teacher and recorded episodes; 4 rounds per episode; 4 samples per round at each effort the template supports (low, medium,
  xhigh), T=1, top-p 0.95, up to 32k new tokens.
- Per effort: mean completion tokens; action error = translation distance from each sample to the medoid of the xhigh
  samples (for xhigh itself: leave-one-out medoid of the other 3); gripper-command agreement; parse failures.
- **Reading, fixed now:** an effort e < xhigh with ≤ 0.6× xhigh's tokens and action error within xhigh's own
  sample-to-sample error (paired CI on rounds includes 0 or is below it) means the cheap baseline "distill the teacher
  at effort e" already compresses, and our method must beat it. If every cheaper effort's error is clearly above
  xhigh's, the paper's 74.4 → 21.9 drop is per-round precision, and compression has to be selective (X1).
- Caveat: action error is a proxy; PEAR's closed loop corrects small errors (and pays in rounds).

**Budget:** one H200, cap $25 for X1 + X2.

## X1 result (2026-11-23, 61 rounds from 14 episodes; `results/effort.jsonl`, `results/effort_summary.txt`)

Run: one Secure H200 (pod `btf6569cfqy8yk`), about 1 h 15 min including setup. 57/61 rounds have exact token-to-chunk
alignment; excluding the other 4 does not change any conclusion (copd/rand 0.54 at 30%).

| Delete ~f of round's thinking, chosen by | dNLL action @30% [95% CI] | @60% |
|---|---|---|
| oracle (smallest single-deletion effect) | −0.003 [−0.008, +0.003] | −0.000 [−0.004, +0.004] |
| **copd (most negative low−xhigh gap)** | **+0.019 [+0.007, +0.032]** | +0.042 [+0.024, +0.060] |
| random (2 seeds) | +0.030 / +0.027 | +0.045 / +0.058 |
| anti (most positive gap) | +0.035 [+0.022, +0.050] | +0.054 |
| tail (drop the end) | +0.080 [+0.058, +0.100] | +0.091 |

Paired: rand − copd +0.0096 [+0.0019, +0.0185] at 30%, +0.0092 [+0.0001, +0.0189] at 60%; anti − copd +0.016
[+0.005, +0.033] at 30%.

**Verdict against the pre-registration: not GO, not NO-GO.**
- Criterion 1 fails: copd/rand = 0.66 at 30% (bar ≤ 0.5); 0.82 at 60%.
- Criterion 1's CI part and criterion 2 hold: the effort contrast is a real but weak pointer to removable reasoning. It
  captures 31% of the oracle headroom at 30% and 18% at 60%.

**What the data say:**
1. **The effort tag does not change the action given the same reasoning** (−0.002 [−0.006, +0.002] nats/token). A COPD
   advantage is ≈ 0 on action tokens; all of its pressure is on the thinking.
2. **COPD's signal is the opening of each round.** Mean gap in the first fifth of a round's chunks is −0.052 vs
   −0.003 to −0.007 elsewhere; copd@30% picks 55% of first-fifth chunks vs 32% for random. The light-effort teacher
   mostly declines to write the opening scene read-out. Averaged over all thinking tokens the gap is small (−0.013
   nats/token; 39% of tokens negative), so as a training signal it is sparse.
3. **The action hangs on the round's conclusion.** The last chunk's single-deletion dNLL is +0.053 vs +0.0003 for every
   other chunk, and it is the most needed chunk in 46/61 rounds. Deleting the tail is by far the worst.
4. **The oracle result is not compressibility.** Teacher forcing keeps the recorded conclusion; it shows the action is
   read off the final plan, not that the earlier reasoning was unnecessary to *reach* that plan. Earlier closed-loop
   evidence (`../pear_anytime_act` §4: action converges late; truncation costs correction rounds) says it was needed.
   Teacher-forced deletion is the wrong necessity oracle for designing a compression reward; a regeneration test is.

**Implication for the method.** Plain COPD would mainly teach the student to skip the scene read-out at the start of a
round. Whether that is safe is a perception question teacher forcing cannot answer (the conclusion still names the
target). The decisive check is closed loop or regeneration: does reasoning that skips the opening still reach the same
plan?

## X2 result (2026-11-23, 55 rounds × 3 efforts × 4 samples; `results/frontier.jsonl`, `results/frontier_summary.txt`)

Rounds have prompts ≤ 60k characters (4 per episode), so they skew to earlier, shorter rounds than the paper's averages.
Action error = distance (state units) from each sample's translation to the medoid of the xhigh samples.

| Effort | Mean tokens (×xhigh) | Median tokens | p90 | Action error, mean [95% CI] | Paired vs xhigh |
|---|---|---|---|---|---|
| low | 1,049 (0.68) | 451 | 1,533 | 4.38 [3.74, 4.99] | +1.36 [+0.40, +2.29] |
| medium | 968 (0.63) | 587 | 1,823 | 3.46 [3.00, 4.02] | +0.44 [−0.43, +1.16] |
| xhigh | 1,546 (1.00) | 604 | 3,916 | 3.02 [2.38, 3.70] (leave-one-out) | — |

Gripper agreement 0.97–0.99 and parse rate 1.00 at every effort.

**Reading against the pre-registration:** `medium` meets the "cheap baseline" condition (0.63× tokens, paired error CI
includes 0). So "distill the teacher at medium" is a baseline our method must beat. `low` is clearly less precise,
consistent with the paper's 74.4 → 21.9.

**The more important finding: reasoning cost is a tail.**
- The *median* round is ~450–600 tokens at every effort; per round, low and medium use 0.86× of xhigh's median.
- The longest 20% of xhigh samples hold 62% of its tokens (recorded X1 rounds: 55%). xhigh's extra cost over medium is
  almost entirely in its p90 (3.9k vs 1.8k).
- Effort does not control the tail cleanly: `low` produced the single longest sample (23.5k tokens), and its longest
  10% hold 57% of its tokens. 15–16 of 55 rounds use *more* tokens at low/medium than at xhigh.

**Implication for the method.** Compression should target the long rounds, not shave every round. A uniform
light-effort contrast (COPD) spreads pressure evenly and, per X1, lands mostly on the opening scene read-out. A
tail-aware objective (advantage or length pressure applied in proportion to how far a round exceeds the teacher's
typical length for that state, with conclusions protected) is the candidate. Its closed-loop test needs a trained
student, i.e. Sophie's OPD pipeline.

**Cost:** pod `btf6569cfqy8yk`, ~2.0 h Secure H200, ≈ $11 for X1 + X2. Terminated and verified (no pods).
