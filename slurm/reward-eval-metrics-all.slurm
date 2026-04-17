#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=re-metrics
#SBATCH --time=00:45:00
#SBATCH -e slurm-metrics-%j.err
#SBATCH -o slurm-metrics-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"


cd "${REWARD_DESIGN_ROOT}"
bash slurm/reward-eval-metrics-all.sh
