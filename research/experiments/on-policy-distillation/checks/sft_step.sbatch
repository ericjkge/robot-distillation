#!/bin/bash
# SFT 60955095's exact step (code bf9b274, scripts/sft-once.sh with sft/run.sbatch's overrides: fused head,
# expandable segments, checkpoint saving) on 2 GPUs with the same tokens per GPU as on 4: max_token_len_per_gpu
# 40960 (163840 / 4), Ulysses SP 2, so a micro-batch holds up to 81,920 tokens, 40,960 per GPU. The 8 episodes
# ($OUT/checks/sft_step/episodes) include the set's two longest under 81,920 tokens (76,375 and 79,815), so the step
# reaches 97% of the per-GPU budget. Each GPU holds half the attention heads instead of a quarter, so this is no
# lighter than the 4-GPU run. One update (batch 8), then one checkpoint (model, optimizer, HF model) as every 10th
# step will write. Peak GPU memory polled every second; host peak in sacct MaxRSS.
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:2,vram:48G
#SBATCH --exclude=compute-g-17-[145-161,165],compute-gc-17-[245-247]  # Quadro RTX 8000, V100, A100 MIG
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/check-%x-%j.out
set -uo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
CODE=$OUT/code/${CODE:-bf9b274}  # bf9b274-headfix: lm_head in the root FSDP unit
export WORK_DIR=$OUT/work PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=$OUT/checks/sft_step/run-$SLURM_JOB_ID
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 > "/tmp/mem-$SLURM_JOB_ID" &
poll=$!
cd "$CODE"
MODEL=$WORK_DIR/models/Qwen3.5-9B EPISODES_DIR=$OUT/checks/sft_step/episodes RUN=$RUN NGPUS=2 \
  bash scripts/sft-once.sh trainer.save_freq=1 trainer.max_ckpt_to_keep=2 model.use_fused_kernels=true \
  data.max_token_len_per_gpu=40960 data.train_batch_size=8 trainer.total_epochs=1 2>&1 |
  grep -v 'UserWarning\|warnings.warn' | tail -60
echo "exit ${PIPESTATUS[0]}; peak MiB per GPU: $(awk -F', ' '{m[$1]=$2>m[$1]?$2:m[$1]} END {for (g in m) printf "%s:%s ", g, m[g]}' "/tmp/mem-$SLURM_JOB_ID")"
kill $poll
ls -la "$RUN"/ckpt/global_step_*/ "$RUN"/ckpt/global_step_*/huggingface 2>&1 | head -40
du -sh "$RUN"/ckpt
