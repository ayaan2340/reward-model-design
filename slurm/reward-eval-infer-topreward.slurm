#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-topreward
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-topreward-%j.err
#SBATCH -o slurm-topreward-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
# Qwen3-VL for TOPReward baseline (robometer wrapper)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/u/asunesara/robometer}"
RESUME="${RESUME:-0}"
# Default: dense prefixes (0) + max_frames=48 to align with robometer RBM (--rbm-max-frames). Override
# TOPREWARD_NUM_PREFIX_SAMPLES=15 for faster sparse+interp curves. Cost scales with frames × forwards.
TOPREWARD_NUM_PREFIX_SAMPLES="${TOPREWARD_NUM_PREFIX_SAMPLES:-0}"
TOPREWARD_MAX_FRAMES="${TOPREWARD_MAX_FRAMES:-48}"

export PYTHONPATH="${REWARD_DESIGN_ROOT}:${ROBOMETER_ROOT}:${PYTHONPATH}"

module load pytorch-conda/2.8 || module load cuda || true

EXTRA=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then EXTRA+=(--resume); fi

cd "${ROBOMETER_ROOT}"
uv run python -m reward_eval.run_reward_inference \
  --manifest "${MANIFEST}" \
  --backend topreward_qwen \
  --model-path "${MODEL_PATH}" \
  --topreward-max-frames "${TOPREWARD_MAX_FRAMES}" \
  --topreward-num-prefix-samples "${TOPREWARD_NUM_PREFIX_SAMPLES}" \
  --predictions-dir "${CACHE_ROOT}/predictions/topreward_qwen" \
  "${EXTRA[@]}"
