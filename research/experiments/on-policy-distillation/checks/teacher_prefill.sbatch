#!/bin/bash
# checks/teacher_prefill.py on 2 GPUs (OPD's teacher: TP2), for each teacher-memory variant of OPD's composed config
# ($OUT/checks/teacher_prefill/<variant>.yaml): as is (verl's default gpu_memory_utilization 0.5, 8192-token chunks),
# 0.85, 0.85 with 2048-token chunks, and that with the student's 164,096-token context. Peak GPU memory polled every second.
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/check-%x-%j.out
set -uo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/47200a9-mbfix
PEAR=$CODE/third_party/pear
EPISODE=$OUT/runs/collect-bf9b274/sft-set/665d9802-43b2-41dc-af59-958946203e3f.pt  # the longest: 158,599 tokens
export WORK_DIR=$OUT/work VERL_CHANGE_IN_PROGRESS=1 LOCAL_DIR=/var/tmp/$USER/pear  # as scripts/common.sh sets it
# OPD's environment (scripts/common.sh pear_exec)
source "$PEAR/scripts/gpu_process_variables.sh" "$WORK_DIR/envs/verl"
export PYTHONPATH=$PEAR/pear/utils/sitecustomize:$PEAR:$PEAR/3rdparty/verl/src:$CODE/src
export VERL_USE_EXTERNAL_MODULES=pear.rollout.server,pear.models.qwen3_5.tool_parser
# as scripts/common.sh does: Slurm sets ROCR_VISIBLE_DEVICES beside CUDA_VISIBLE_DEVICES, which verl's workers refuse
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then unset ROCR_VISIBLE_DEVICES; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$CODE"
for variant in ${VARIANTS:-as_is mem85 mem85_chunk2048 mem85_chunk2048_len164k}; do
  echo "== $(date +%H:%M:%S) $variant"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 > "/tmp/mem-$variant" &
  poll=$!
  timeout 1500 python /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/checks/teacher_prefill.py \
    "$OUT/checks/teacher_prefill/$variant.yaml" "$EPISODE" 2>&1 | grep -v 'UserWarning\|warnings.warn' | tail -40
  echo "== $variant exit ${PIPESTATUS[0]}; peak MiB per GPU: $(sort -t, -k1,1n -k2,2n "/tmp/mem-$variant" | awk -F', ' '{m[$1]=$2>m[$1]?$2:m[$1]} END {for (g in m) printf "%s:%s ", g, m[g]}')"
  kill $poll
  ray stop --force > /dev/null 2>&1
done
