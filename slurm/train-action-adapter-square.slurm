#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=train-act-adapt
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-train-action-adapter-%j.err
#SBATCH -o slurm-train-action-adapter-%j.out
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

set -eo pipefail

# Trains the small Dynamics MLP in Ctrl-World/models/action_adapter/train2.py (joint velocity -> joint delta).
# Does NOT load pi0.5 or Ctrl-World weights; it only needs rollout annotations with the same layout as train_wm
# (observation.state.joint_position + action.joint_velocity), i.e. the dataset under --dataset-root/--dataset-name.
#
# Reference checkpoints (for your records / rollout_interact_pi.py — not used in this job):
#   Pi0.5:  OPENPI_ROOT + openpi checkpoints from train-pi05-square.slurm (EXP_NAME=square_finetune-norm-stats)
#   WM:     finetune-ctrl-world.slurm --output_dir (e.g. ctrl_world_robomimic_ckpt_2)

CTRL_WORLD_ROOT="${CTRL_WORLD_ROOT:-/u/asunesara/Ctrl-World}"
ROLLOUT_DATASET_ROOT="${ROLLOUT_DATASET_ROOT:-/projects/bggq/asunesara/rollout_dataset}"
DATASET_NAME="${DATASET_NAME:-robomimic_policy_square_latents_skip}"
META_INFO_DIR="${META_INFO_DIR:-${CTRL_WORLD_ROOT}/dataset_meta_info}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bggq/asunesara/${DATASET_NAME}_action_adapter}"

module load pytorch-conda/2.8 || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate ctrl-world

export PYTHONPATH="${CTRL_WORLD_ROOT}:${PYTHONPATH}"

python "${CTRL_WORLD_ROOT}/models/action_adapter/train2.py" \
  --source rollout \
  --dataset-root "${ROLLOUT_DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --meta-info-dir "${META_INFO_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-4 \
  --num-workers 8 \
  --num-frames 15

echo "Done. Latest checkpoint: ${OUTPUT_DIR}/model2_15_latest.pth"
echo "Point config.py action_adapter or rollout_interact_pi --action_adapter to that path."
