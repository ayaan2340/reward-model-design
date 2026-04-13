#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=train-pi05-square
#SBATCH --time=12:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

echo "job is starting on $(hostname)"

OPENPI_ROOT="${OPENPI_ROOT:-/projects/bggq/asunesara/openpi}"
EXP_NAME="${EXP_NAME:-square_finetune-norm-stats}"

# Matches openpi TrainConfig pi05_robosuite_square + LeRobot repo_id local/robomimic_square_jvel
NORM_STATS_JSON="${OPENPI_ROOT}/assets/pi05_robosuite_square/local/robomimic_square_jvel/norm_stats.json"

module load pytorch-conda/2.8

conda activate ctrl-world

cd "${OPENPI_ROOT}"

if [[ ! -f "${NORM_STATS_JSON}" ]]; then
  echo "Norm stats missing at ${NORM_STATS_JSON}"
  echo "Run convert_robomimic_to_lerobot.py first, then this job will compute stats from the LeRobot dataset."
  uv run scripts/compute_norm_stats.py --config-name pi05_robosuite_square
fi

uv run scripts/train.py pi05_robosuite_square --exp-name "${EXP_NAME}" --resume

echo "train-pi05-square complete"
