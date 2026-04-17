#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=re-metrics
#SBATCH --time=00:30:00
#SBATCH -e slurm-metrics-%j.err
#SBATCH -o slurm-metrics-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
# Predictions from one backend, e.g. ${CACHE_ROOT}/predictions/topreward_qwen
PRED_DIR="${PRED_DIR:-${CACHE_ROOT}/predictions/robodopamine}"
BACKEND_LABEL="${BACKEND_LABEL:-robodopamine}"
CHECKPOINT_ID="${CHECKPOINT_ID:-tanhuajie2001/Robo-Dopamine-GRM-3B}"

cd "${REWARD_DESIGN_ROOT}"

ARGS=(--manifest "${MANIFEST}" --predictions-dir "${PRED_DIR}")
[[ -n "${BACKEND_LABEL}" ]] && ARGS+=(--backend-label "${BACKEND_LABEL}")
[[ -n "${CHECKPOINT_ID}" ]] && ARGS+=(--checkpoint-id "${CHECKPOINT_ID}")

python -m reward_eval.compute_metrics "${ARGS[@]}"
