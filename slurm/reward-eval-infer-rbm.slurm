#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuH200x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-rbm
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-rbm-%j.err
#SBATCH -o slurm-rbm-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
# RBM and ReWiND (reward-fm) share robometer's RBMModel — use this script for both.
# Continuous progress: Sigmoid on progress head (robometer/models/heads.py); eval clips to [0,1].
# Discrete progress: bin logits -> softmax expectation in RBMModel (convert_bins_to_continuous).
# Examples: robometer/Robometer-4B, rewardfm/rbm_qwen_pref_prog_4frames_all_strategy,
#           rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 (ReWiND).
CHECKPOINT="${RBM_CHECKPOINT:-robometer/Robometer-4B}"
# Qwen3-VL RBM: full rollout length can OOM even on H200; subsample frames then interpolate (default 48).
RBM_MAX_FRAMES="${RBM_MAX_FRAMES:-50}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/u/asunesara/robometer}"
RESUME="${RESUME:-0}"

export PYTHONPATH="${REWARD_DESIGN_ROOT}:${ROBOMETER_ROOT}:${PYTHONPATH}"

module load pytorch-conda/2.8 || true

EXTRA=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then EXTRA+=(--resume); fi

cd "${ROBOMETER_ROOT}"
uv run python -m reward_eval.run_reward_inference \
  --manifest "${MANIFEST}" \
  --backend rbm \
  --checkpoint "${CHECKPOINT}" \
  --rbm-max-frames "${RBM_MAX_FRAMES}" \
  --predictions-dir "${CACHE_ROOT}/predictions/rbm" \
  "${EXTRA[@]}"
