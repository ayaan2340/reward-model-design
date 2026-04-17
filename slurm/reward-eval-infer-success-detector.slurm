#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-succ
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-success-detector-%j.err
#SBATCH -o slurm-success-detector-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

# Prerequisites:
# 1) extract_latent_robomimic.py per HDF5 → each --output_path has latent_videos/<train|val>/<traj_id>/*.pt
# 2) Train SuccessPredictor; set SUCCESS_CKPT
#
# Preprocess merges expert_ph (~200) + rollout_mixed (~300) — use TWO latent roots (same demo_id in both sets else collision).

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"
LATENT_ROOT_EXPERT="${LATENT_ROOT_EXPERT:-/projects/bggq/asunesara/robomimic_square_latents}"
LATENT_ROOT_ROLLOUT="${LATENT_ROOT_ROLLOUT:-/scratch/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents}"
# Optional single fallback if you only run one dataset via --dataset-filter:
LATENT_ROOT="${LATENT_ROOT:-}"

SUCCESS_CKPT="${SUCCESS_CKPT:-/projects/bggq/asunesara/success_detector_runs/run3_attention_no_fp/best.pt}"
RESUME="${RESUME:-0}"

cd "${REWARD_DESIGN_ROOT}"
module load pytorch-conda/2.8
conda activate ctrl-world

ARGS=(
  --manifest "${MANIFEST}"
  --backend success_detector
  --checkpoint "${SUCCESS_CKPT}"
  --latent-root-expert "${LATENT_ROOT_EXPERT}"
  --latent-root-rollout "${LATENT_ROOT_ROLLOUT}"
  --predictions-dir "${CACHE_ROOT}/predictions/success_detector"
)
[[ -n "${LATENT_ROOT}" ]] && ARGS+=( --latent-root "${LATENT_ROOT}" )
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then ARGS+=(--resume); fi

python -m reward_eval.run_reward_inference "${ARGS[@]}"
