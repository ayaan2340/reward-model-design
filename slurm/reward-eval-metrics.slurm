#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-cpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=re-metrics
#SBATCH --time=00:30:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
# Predictions from one backend, e.g. ${CACHE_ROOT}/predictions/topreward_qwen
PRED_DIR="${PRED_DIR:?Set PRED_DIR to predictions/<backend>}"
BACKEND_LABEL="${BACKEND_LABEL:-}"
CHECKPOINT_ID="${CHECKPOINT_ID:-}"

cd "${REWARD_DESIGN_ROOT}"
module load python/3.11 || true

ARGS=(--manifest "${MANIFEST}" --predictions-dir "${PRED_DIR}")
[[ -n "${BACKEND_LABEL}" ]] && ARGS+=(--backend-label "${BACKEND_LABEL}")
[[ -n "${CHECKPOINT_ID}" ]] && ARGS+=(--checkpoint-id "${CHECKPOINT_ID}")

python -m reward_eval.compute_metrics "${ARGS[@]}"
