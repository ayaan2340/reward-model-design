#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=eval-pi05-square
#SBATCH --time=01:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-eval-pi05-square-%j.err
#SBATCH -o slurm-eval-pi05-square-%j.out
#SBATCH --gpus-per-node=1   
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

echo "job is starting on $(hostname)"

module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world

OPENPI_ROOT="${OPENPI_ROOT:-/projects/bggq/asunesara/openpi}"
REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
# gs://openpi-assets/... resolves under $OPENPI_DATA_HOME/openpi-assets/... (see openpi/shared/download.py).
# Default ~/.cache/openpi breaks after moving the cache next to the openpi checkout under /projects/...
_PROJECT_ROOT="$(dirname "${OPENPI_ROOT}")"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${_PROJECT_ROOT}/.cache/openpi}"
export OPENPI_ROOT

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

SERVE_ARGS="${SERVE_ARGS:---env DROID}"

cd "${OPENPI_ROOT}"
uv run scripts/serve_policy.py ${SERVE_ARGS} --port "${PORT}" &
SERVE_PID=$!
sleep 45


cd "${REWARD_DESIGN_ROOT}"
python eval_pi05_square.py --host "${HOST}" --port "${PORT}" \
  --num-rollouts "${NUM_ROLLOUTS:-5}" \
  --video-out-path "${VIDEO_OUT:-videos/pi05_square}" \
  --results-csv "${RESULTS_CSV:-results/pi05_square.csv}" \
  --control-freq "${CONTROL_FREQ:-15}"

kill "${SERVE_PID}" 2>/dev/null || true
wait "${SERVE_PID}" 2>/dev/null || true

echo "eval-pi05-square complete"
