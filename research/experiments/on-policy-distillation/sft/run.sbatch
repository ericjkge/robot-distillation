#!/bin/bash
# SFT of one student (STUDENT=Qwen3.5-9B | Qwen3.5-4B) on the teacher's episodes: up to 25 per task, over every
# cluster's episodes on the Hub dataset when the job starts (sft/build_set.sh, seed 0; the first job to start draws the
# set), with configs/step/sft.yaml's settings (lr 1e-5 cosine, 5% warmup, batch 32, 3 epochs), each episode split
# across the 4 GPUs, the update on the CPU.
#SBATCH --partition=gpu_requeue  # preemptible and billed at 0.1 per GPU instead of 5
#SBATCH --requeue
#SBATCH --gres=gpu:4,vram:48G  # L40S, A6000 or A100
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG: no bf16 or too small (60796702-03 failed on RTX 8000)
#SBATCH --time=12:00:00  # SFT takes about 6-8 h
#SBATCH --cpus-per-task=20
# host memory: the 4B pilot peaked at 147 GB on 2 GPUs; the 9B scales to about 330 GB. Submit the 4B with --mem=224G
#SBATCH --mem=480G  # training holds about 390 GB; a save at 400G was killed (61038523)
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/sft-%x-%j.out
set -euo pipefail
# a second copy of the job waits in another partition (sft/race.sh): it leaves if a copy already runs
if squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o %i | grep -vqx "$SLURM_JOB_ID"; then echo "another copy runs"; exit 0; fi
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
STUDENT=${STUDENT:?Qwen3.5-9B | Qwen3.5-4B}
CODE_SHA=${CODE_SHA:?the commit of the code to run}
COLLECT=${COLLECT:?the collection runs directory, runs/collect-<sha>}
CODE=$OUT/code/$CODE_SHA
export WORK_DIR=$OUT/work
exec 9> "$OUT/code/.lock"
flock 9
if [ ! -d "$CODE" ]; then
  git clone -q /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation "$CODE"
  git -C "$CODE" checkout -q "$CODE_SHA"
  git -C "$CODE" submodule update -q --init
  bash "$CODE/third_party/pear/3rdparty/verl/scripts/checkout_verl.sh"
fi
SET=$COLLECT/sft-set
if [ ! -d "$SET" ]; then
  bash /n/data1/hms/dbmi/zitnik/lab/users/edk005/robot-distillation/.wt/on-policy-distillation/research/experiments/on-policy-distillation/sft/build_set.sh
fi
flock -u 9
cd "$CODE"
echo "code $(git rev-parse HEAD) on $(hostname); $(ls "$SET"/*.pt | wc -l) episodes in $SET"

start=$SECONDS
# the fused head (as PEAR trains Qwen3.5) gives log-probs 512 tokens at a time: the full logits of the longest episode
# (159k tokens, 40k per GPU, 248k vocabulary) need 17 GiB more than a 48 GB GPU had (60884159, Nov 27)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$WORK_DIR/models/$STUDENT EPISODES_DIR=$SET RUN=$OUT/runs/sft-${CODE_SHA:0:7}/$STUDENT NGPUS=4 \
  bash scripts/sft.sh trainer.save_freq=-1 'checkpoint.save_contents=[model,extra,hf_model]' \
  model.use_fused_kernels=true
echo "sft $STUDENT: $((SECONDS - start)) s"
