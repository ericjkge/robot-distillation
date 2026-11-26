#!/bin/bash
# Rebuilds flash-attn 2 in the verl environment for sm_80. The setup ran on an L40S, so PEAR built it for sm_89 only and
# SFT 60783661 failed on A6000s (no kernel image for the device). sm_80 kernels run on every 8.x GPU (A100, A6000, A40,
# L40S), so SFT and OPD can take any of them. PEAR's own build step (scripts/install/verl/install.sh), with CUDA_ARCH 8.0.
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --output=/n/scratch/users/e/edk005/vla/on-policy-distillation/logs/fa2-build-%j.out
set -euo pipefail
OUT=/n/scratch/users/e/edk005/vla/on-policy-distillation
WORK=$OUT/work
PEAR=$OUT/code/bf9b274/third_party/pear
PY=$WORK/envs/verl/bin/python
TOOLKIT=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
export UV_CONSTRAINT=$PEAR/scripts/install/verl/constraints.txt UV_OVERRIDE=$PEAR/scripts/install/verl/overrides.txt
export UV_CACHE_DIR=$WORK/cache/uv
start=$SECONDS
CUDA_HOME=$TOOLKIT PATH=$TOOLKIT/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  MAX_JOBS=16 "$WORK/bin/uv" pip install --python "$PY" --no-build-isolation --reinstall-package flash-attn \
  --no-cache flash-attn  # no cache: uv otherwise reinstalls the sm_89 wheel it built before
/n/app/cuda/12.8.0-gcc-14.2.0/bin/cuobjdump --list-elf "$WORK"/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda*.so |
  grep -o 'sm_[0-9]*' | sort | uniq -c | tee /dev/stderr | grep -q sm_80  # fails the job (and SFT after it) otherwise
echo "built in $((SECONDS - start)) s"
