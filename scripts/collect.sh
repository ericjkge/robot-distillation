#!/usr/bin/env bash
# Step 1: runs the teacher over LIBERO's four suites and keeps every successful episode's full multi-round context
# as an SFT example under $RUN/episodes.
#   MODEL=<teacher dir> RUN=<run dir> bash scripts/run.sh scripts/collect.sh [overrides]
# FIRST_K=<k> (default 1): each turn samples k times and keeps the first sample to finish with a usable action, the
# others aborted (opd.first_k): concise episodes, in the same format
source "$(dirname "$0")/common.sh"

MODEL_FILE=$ROOT/configs/models/teacher.sh
pear_args

RUN_ARGS+=(trainer.eval_step=0 +data.apply_chat_template_kwargs.reasoning_effort=xhigh)
FIRST_K=${FIRST_K:-1}
if [ "$FIRST_K" -gt 1 ]; then
  RUN_ARGS+=(
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/configs/agent_loop/first_k.yaml"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_FIRST_K='$FIRST_K'"
  )
fi
pear_exec collect "$@"
