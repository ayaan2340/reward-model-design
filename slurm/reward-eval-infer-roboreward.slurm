#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-roboreward
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-roboreward-%j.err
#SBATCH -o slurm-roboreward-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

# RoboReward: end-of-episode discrete score (1–5) replicated across frames — slow (VLM gen per trajectory).
# Default HF: https://huggingface.co/teetone/RoboReward-8B
MODEL_PATH="${MODEL_PATH:-teetone/RoboReward-8B}"

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/u/asunesara/robometer}"
RESUME="${RESUME:-0}"

export PYTHONPATH="${REWARD_DESIGN_ROOT}:${ROBOMETER_ROOT}:${PYTHONPATH}"

module load pytorch-conda/2.8 || module load cuda || true

EXTRA=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then EXTRA+=(--resume); fi

cd "${ROBOMETER_ROOT}"
uv run python -m reward_eval.run_reward_inference \
  --manifest "${MANIFEST}" \
  --backend roboreward \
  --model-path "${MODEL_PATH}" \
  --predictions-dir "${CACHE_ROOT}/predictions/roboreward" \
  "${EXTRA[@]}"
