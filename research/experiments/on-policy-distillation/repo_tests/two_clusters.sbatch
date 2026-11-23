#!/bin/bash
# Two Ray runs of scripts/run.sh on one machine at once, as two Slurm jobs sharing a node do. Checks:
#   - each run starts its own cluster;
#   - the short run's exit leaves the long run's cluster alive: the long run's task, started before and finished
#     after that exit, completes;
#   - no Ray process is left once both runs end;
#   - a node that fails to start fails its run within a minute, instead of leaving it waiting.
# CPU only: a stub nvidia-smi reports no GPUs. CODE_SHA: the commit under test. LOCAL=1 runs both off Slurm, as on a
# plain machine (scripts/run.sh then starts the nodes itself instead of through srun).
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/two_clusters-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE_SHA=${CODE_SHA:?the commit under test}
export WORK_DIR=$OUT/work
CODE=$(mktemp -d)
git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
git -C "$CODE" checkout -q "$CODE_SHA"
git -C "$CODE" submodule update -q --init
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname)"

STUB=$(mktemp -d)
printf '#!/bin/sh\nexit 0\n' > "$STUB/nvidia-smi"  # no GPUs listed
chmod +x "$STUB/nvidia-smi"
export PATH=$STUB:$PATH NUM_CPUS=2 LOCAL_DIR=$(mktemp -d)

# the step: names its cluster, then runs one task that takes PROBE_SECONDS
cat > scripts/probe.sh <<'EOF'
source "$(dirname "$0")/common.sh"
"$VERL_ENV/bin/python" - <<'PY'
import os, time, ray
ray.init()
print("cluster", os.environ["RAY_ADDRESS"], flush=True)
@ray.remote
def task(seconds):
    time.sleep(seconds)
    return "done"
print("task", ray.get(task.remote(int(os.environ["PROBE_SECONDS"]))), flush=True)
PY
EOF

[ "${LOCAL:-0}" = 1 ] && unset SLURM_JOB_NODELIST
PROBE_SECONDS=300 bash scripts/run.sh scripts/probe.sh > long.log 2>&1 &
LONG=$!
sleep 45  # the long run's cluster is up and its task running
PROBE_SECONDS=1 bash scripts/run.sh scripts/probe.sh > short.log 2>&1
ALIVE=$(kill -0 $LONG 2> /dev/null && echo yes || echo no)
echo "short run ended at $SECONDS s; long run still going: $ALIVE"
wait $LONG
grep -h "^cluster\|^task" long.log short.log
sleep 5
LEFT=$(pgrep -u "$USER" -f "$LOCAL_DIR" || true)

[ "$(grep -c '^task done' long.log short.log | awk -F: '{s += $2} END {print s}')" = 2 ] || { echo "FAIL: a task did not complete"; cat long.log short.log; exit 1; }
[ "$(grep -h '^cluster' long.log short.log | sort -u | wc -l)" = 2 ] || { echo "FAIL: the runs shared a cluster"; exit 1; }
[ "$ALIVE" = yes ] || { echo "FAIL: the long run ended before the short one"; exit 1; }
[ -z "$LEFT" ] || { echo "FAIL: Ray processes left: $LEFT"; ps -o pid,cmd -p $LEFT; exit 1; }

start=$SECONDS
if NUM_CPUS=none PROBE_SECONDS=1 timeout 300 bash scripts/run.sh scripts/probe.sh > broken.log 2>&1; then
  echo "FAIL: a run whose node cannot start succeeded"; exit 1
fi
echo "a node that cannot start: the run failed after $((SECONDS - start)) s"
(( SECONDS - start < 60 )) || { echo "FAIL: it took a minute or more"; tail broken.log; exit 1; }
echo PASS
