#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=rollout-pi-wm
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-rollout-interact-pi-%j.err
#SBATCH -o slurm-rollout-interact-pi-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

# Ctrl-World policy + world-model rollout on dataset val trajectories (offline videos/latents).
# Matches finetune paths from reward-model-design/slurm/finetune-ctrl-world.slurm and train-pi05-square.slurm.

CTRL_WORLD_ROOT="${CTRL_WORLD_ROOT:-/u/asunesara/Ctrl-World}"
OPENPI_ROOT="${OPENPI_ROOT:-/projects/bggq/asunesara/openpi}"

# Pi0.5 fine-tune: directory containing OpenPI checkpoint (same layout as training; see openpi checkpoints/<exp>/)
PI_EXP_NAME="${PI_EXP_NAME:-square_finetune-norm-stats}"
PI_CHECKPOINT_DIR="${PI_CHECKPOINT_DIR:-${OPENPI_ROOT}/checkpoints/pi05_robosuite_square/${PI_EXP_NAME}/19999}"

# World model: Ctrl-World train_wm.py output (--output_dir) and a checkpoint-*.pt inside it
WM_OUTPUT_DIR="${WM_OUTPUT_DIR:-/projects/bggq/asunesara/ctrl_world_robomimic_ckpt_2}"
WM_CHECKPOINT="${WM_CHECKPOINT:-${WM_OUTPUT_DIR}/checkpoint-10000.pt}"

# Same SVD / CLIP as finetune-ctrl-world.slurm (paths must exist on cluster)
SVD_MODEL_PATH="${SVD_MODEL_PATH:-/projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/projects/bggq/asunesara/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/}"

# Rollout dataset (same root/name as --dataset_names in train_wm.py)
ROLLOUT_DATASET_ROOT="${ROLLOUT_DATASET_ROOT:-/scratch/bggq/asunesara/rollout_dataset}"
DATASET_NAME="${DATASET_NAME:-robomimic_policy_square_latents_skip}"

# Joint-vel -> joint-delta adapter (train2.py); required for pi05 in rollout_interact_pi.py
ACTION_ADAPTER_PATH="${ACTION_ADAPTER_PATH:-/scratch/bggq/asunesara/${DATASET_NAME}_action_adapter/model2_15_latest.pth}"

# Val episodes to run (comma-separated episode_id matching annotation/val/<id>.json).
# Leave empty to auto-enumerate annotation/val and take first ROLLOUT_NUM_ROLLOUTS.
export ROLLOUT_VAL_EPISODE_IDS="${ROLLOUT_VAL_EPISODE_IDS:-}"
# Cap how many trajectories to run (first N ids after split); overrides default long list
export ROLLOUT_NUM_ROLLOUTS="${ROLLOUT_NUM_ROLLOUTS:-10}"
# Match Robosuite 400-step @ 20Hz control horizon (sets interact_num from pred_step)
export ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-400}"

export ROLLOUT_INSTRUCTION="${ROLLOUT_INSTRUCTION:-fit the square nut onto the square peg}"
export ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-${ROLLOUT_DATASET_ROOT}/${DATASET_NAME}_rollout_interact_pi}"
export ROLLOUT_VAL_DATASET_DIR="${ROLLOUT_VAL_DATASET_DIR:-${ROLLOUT_DATASET_ROOT}/${DATASET_NAME}}"

module load pytorch-conda/2.8 || true
module load ffmpeg
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate ctrl-world

export PYTHONPATH="${CTRL_WORLD_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH}"

cd "${CTRL_WORLD_ROOT}"

echo "[preflight] CTRL_WORLD_ROOT=${CTRL_WORLD_ROOT}"
echo "[preflight] PI_CHECKPOINT_DIR=${PI_CHECKPOINT_DIR}"
echo "[preflight] WM_CHECKPOINT=${WM_CHECKPOINT}"
echo "[preflight] ACTION_ADAPTER_PATH=${ACTION_ADAPTER_PATH}"
echo "[preflight] ROLLOUT_VAL_DATASET_DIR=${ROLLOUT_VAL_DATASET_DIR}"

[[ -d "${CTRL_WORLD_ROOT}" ]] || { echo "[preflight] missing CTRL_WORLD_ROOT: ${CTRL_WORLD_ROOT}"; exit 2; }
[[ -f "${WM_CHECKPOINT}" ]] || { echo "[preflight] missing WM_CHECKPOINT: ${WM_CHECKPOINT}"; exit 2; }
[[ -f "${ACTION_ADAPTER_PATH}" ]] || { echo "[preflight] missing ACTION_ADAPTER_PATH: ${ACTION_ADAPTER_PATH}"; exit 2; }
[[ -d "${ROLLOUT_VAL_DATASET_DIR}/annotation/val" ]] || {
  echo "[preflight] missing annotation/val under: ${ROLLOUT_VAL_DATASET_DIR}";
  exit 2;
}
if [[ ! -f "${PI_CHECKPOINT_DIR}/params/_METADATA" ]]; then
  echo "[preflight] expected OpenPI orbax metadata at ${PI_CHECKPOINT_DIR}/params/_METADATA"
  echo "[preflight] if you passed experiment root, set PI_CHECKPOINT_DIR to a numeric step dir (or let rollout auto-resolve)."
  exit 2
fi

python scripts/rollout_interact_pi.py \
  --task_type robomimic_square \
  --svd_model_path "${SVD_MODEL_PATH}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --ckpt_path "${WM_CHECKPOINT}" \
  --dataset_root_path "${ROLLOUT_DATASET_ROOT}" \
  --dataset_meta_info_path "${CTRL_WORLD_ROOT}/dataset_meta_info" \
  --dataset_names "${DATASET_NAME}" \
  --pi_ckpt "${PI_CHECKPOINT_DIR}" \
  --pi-config pi05_robosuite_square \
  --action_adapter "${ACTION_ADAPTER_PATH}" \
  --horizon_steps "${ROLLOUT_HORIZON_STEPS}" \
  --num_rollouts "${ROLLOUT_NUM_ROLLOUTS}"

echo "Outputs under ROLLOUT_SAVE_DIR (default: .../${DATASET_NAME}_rollout_interact_pi) and config save_dir."
