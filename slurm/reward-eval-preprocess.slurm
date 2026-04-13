#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bggq-delta-cpu
#SBATCH --job-name=re-preprocess
#SBATCH --time=02:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
EXPERT_HDF5="${EXPERT_HDF5:-/scratch/bggq/asunesara/square_image_dense_three.hdf5}"
ROLLOUT_HDF5="${ROLLOUT_HDF5:-/scratch/bggq/asunesara/rollout_dataset/output_dense_shaping.hdf5}"

cd "${REWARD_DESIGN_ROOT}"
module load pytorch-conda/2.8
conda activate robomimic

python -m reward_eval.preprocess_manifest \
  --expert-hdf5 "${EXPERT_HDF5}" \
  --rollout-hdf5 "${ROLLOUT_HDF5}" \
  --output-dir "${CACHE_ROOT}" \
  --camera agentview \
  --expert-gt linear

echo "Manifest: ${CACHE_ROOT}/manifest.csv"
