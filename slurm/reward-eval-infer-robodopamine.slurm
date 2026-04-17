#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-rbd
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-robodopamine-%j.err
#SBATCH -o slurm-robodopamine-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
MODEL_PATH="${MODEL_PATH:-tanhuajie2001/Robo-Dopamine-GRM-3B}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/u/asunesara/robometer}"
# Set RESUME=1 to skip demos that already have valid pred/gt npz (safe after partial runs).
# Set FORCE_RECOMPUTE=1 to pass --no-resume (recompute even when npz exists — needed after fixing
# parse/code: plain --resume still skips all-zero pred files because shape checks pass).
RESUME="${RESUME:-1}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
# GRM REFERENCE END: last frame of an expert preprocess run. Set ROBODOPAMINE_GOAL_NPZ to a frames.npz
# path to pin a specific demo; otherwise we use --robodopamine-auto-expert-goal (first expert_ph row in manifest).
ROBODOPAMINE_GOAL_NPZ="${ROBODOPAMINE_GOAL_NPZ:-}"

export PYTHONPATH="${REWARD_DESIGN_ROOT}:${ROBOMETER_ROOT}:${PYTHONPATH}"

cd "${REWARD_DESIGN_ROOT}"
module load pytorch-conda/2.8 || true
# Robo-Dopamine needs vLLM / torch stack per robometer/evals/README.md — use a dedicated conda env (not uv run).
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate robo-dopamine

EXTRA=()
if [[ "${FORCE_RECOMPUTE}" == "1" || "${FORCE_RECOMPUTE}" == "true" || "${FORCE_RECOMPUTE}" == "yes" ]]; then
  EXTRA+=(--no-resume)
elif [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  EXTRA+=(--resume)
fi

GOAL_EXTRA=()
if [[ -n "${ROBODOPAMINE_GOAL_NPZ}" ]]; then
  GOAL_EXTRA+=(--robodopamine-expert-goal-frames-npz "${ROBODOPAMINE_GOAL_NPZ}")
else
  GOAL_EXTRA+=(--robodopamine-auto-expert-goal)
fi

python -m reward_eval.run_reward_inference \
  --manifest "${MANIFEST}" \
  --backend robodopamine \
  --model-path "${MODEL_PATH}" \
  --predictions-dir "${CACHE_ROOT}/predictions/robodopamine" \
  --robodopamine-frame-interval 30 \
  "${GOAL_EXTRA[@]}" \
  "${EXTRA[@]}"
