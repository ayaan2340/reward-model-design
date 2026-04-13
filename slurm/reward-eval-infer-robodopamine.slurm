#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-rbd
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
MODEL_PATH="${MODEL_PATH:-tanhuajie2001/Robo-Dopamine-GRM-3B}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/u/asunesara/robometer}"
# Set RESUME=1 to skip demos that already have valid pred/gt npz (safe after partial runs).
RESUME="${RESUME:-1}"

export PYTHONPATH="${REWARD_DESIGN_ROOT}:${ROBOMETER_ROOT}:${PYTHONPATH}"

cd "${REWARD_DESIGN_ROOT}"
module load pytorch-conda/2.8 || true
# Robo-Dopamine needs vLLM / torch stack per robometer/evals/README.md — use a dedicated conda env (not uv run).
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate robo-dopamine

EXTRA=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  EXTRA+=(--resume)
fi

python -m reward_eval.run_reward_inference \
  --manifest "${MANIFEST}" \
  --backend robodopamine \
  --model-path "${MODEL_PATH}" \
  --predictions-dir "${CACHE_ROOT}/predictions/robodopamine" \
  "${EXTRA[@]}"
