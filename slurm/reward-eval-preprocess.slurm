#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=re-eval-preprocess
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH -e slurm-preprocess-%j.err
#SBATCH -o slurm-preprocess-%j.out
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


# python3 << 'PY'
# import json
# import numpy as np
# from pathlib import Path

# npz_path = Path("demo_2.npz")  # change: demo_5.npz, or rollout_mixed/demo_3.npz
# meta_path = npz_path.with_suffix(".meta.json")

# z = np.load(npz_path)
# pred = np.asarray(z["pred"], dtype=np.float64).ravel()
# gt = np.asarray(z["gt"], dtype=np.float64).ravel()
# print("npz:", npz_path)
# print("keys:", list(z.files))
# print("T:", pred.size, "pred shape:", pred.shape, "gt shape:", gt.shape)
# print("pred: min=%.6g max=%.6g" % (pred.min(), pred.max()))
# print("gt:   min=%.6g max=%.6g" % (gt.min(), gt.max()))
# if "gt_definition" in z.files:
#     print("gt_definition:", z["gt_definition"])

# idxs = [0, len(pred) // 4, len(pred) // 2, 3 * len(pred) // 4, len(pred) - 1]
# idxs = sorted(set(i for i in idxs if 0 <= i < len(pred)))
# print("\nt | pred     | gt")
# print("-" * 40)
# for t in idxs:
#     print("%d | %8.5f | %8.5f" % (t, pred[t], gt[t]))

# show = min(16, len(pred))
# print("\nFirst %d steps:" % show)
# for t in range(show):
#     print("  t=%4d  pred=%.6f  gt=%.6f" % (t, pred[t], gt[t]))

# if meta_path.is_file():
#     print("\n", meta_path.name, ":")
#     print(json.dumps(json.loads(meta_path.read_text()), indent=2))
# PY
