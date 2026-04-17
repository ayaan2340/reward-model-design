#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=eval-pi05-square-ckpts
#SBATCH --time=01:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-eval-pi05-square-ckpts-%j.err
#SBATCH -o slurm-eval-pi05-square-ckpts-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

# No `set -u`: conda's MKL deactivate hooks reference unset vars and fail under nounset.
set -eo pipefail

echo "job is starting on $(hostname)"

# pytorch-conda: provides `conda` and often `uv` in non-interactive jobs.
module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world

OPENPI_ROOT="${OPENPI_ROOT:-/projects/bggq/asunesara/openpi}"
REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
_PROJECT_ROOT="$(dirname "${OPENPI_ROOT}")"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${_PROJECT_ROOT}/.cache/openpi}"
export OPENPI_ROOT

# Norm stats: default policy loads <ckpt>/assets/local/robomimic_square_jvel/. Legacy runs (DROID stats): add
#   --checkpoint-norm-stats-asset-id droid
# to the python line below.

# Finetuned checkpoints (step subdirs under train exp square_finetune)
CKPT_ROOT="${CKPT_ROOT:-${OPENPI_ROOT}/checkpoints/pi05_robosuite_square/square_finetune-norm-stats}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_robosuite_square}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-25}"
CONTROL_FREQ="${CONTROL_FREQ:-20}"

# Use openpi's uv environment for eval (same JAX/orbax/torch pins as serve_policy / train).
# Running `python` from ctrl-world mixes conda jaxlib/orbax with openpi and breaks orbax
# (e.g. AttributeError: jax.experimental.layout has no DeviceLocalLayout).
# robosuite is not an openpi dependency; pull it in for this command only.
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

STEPS=(19999)

cd "${OPENPI_ROOT}"

for STEP in "${STEPS[@]}"; do
  CKPT_DIR="${CKPT_ROOT}/${STEP}"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: missing checkpoint directory: ${CKPT_DIR}" >&2
    exit 1
  fi

  # Absolute paths: uv run uses OPENPI_ROOT as cwd, so relative video/results paths would land in openpi/.
  VIDEO_OUT="${VIDEO_OUT_BASE:-${REWARD_DESIGN_ROOT}/videos/pi05_square_finetune-norm-stats}/step_${STEP}"
  RESULTS_CSV="${RESULTS_CSV_BASE:-${REWARD_DESIGN_ROOT}/results}/pi05_square_finetune_norm_stats_step_${STEP}.csv"

  echo "=== Evaluating step ${STEP} (${CKPT_DIR}) ==="

  uv run --with robosuite python "${REWARD_DESIGN_ROOT}/eval_pi05_square.py" \
    --use-local-policy \
    --policy-config "${POLICY_CONFIG}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --control-freq "${CONTROL_FREQ}" \
    --video-out-path "${VIDEO_OUT}" \
    --results-csv "${RESULTS_CSV}"
done

echo "eval-pi05-square-checkpoints complete"
