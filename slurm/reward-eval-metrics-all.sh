#!/usr/bin/env bash
# Run compute_metrics.py once per reward backend, sequentially (same manifest + cache layout as infer scripts).
#
# Usage (login node or inside an interactive allocation):
#   export CACHE_ROOT=/scratch/bggq/asunesara/reward_eval_cache   # optional
#   ./slurm/reward-eval-metrics-all.sh
#
# Override checkpoint ids for CSV reporting only:
#   CHID_RBM=my/hf-rbm CHID_SUCCESS_CKPT=/path/to/best.pt ./slurm/reward-eval-metrics-all.sh
#
set -euo pipefail

REWARD_DESIGN_ROOT="${REWARD_DESIGN_ROOT:-/u/asunesara/reward-model-design}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/bggq/asunesara/reward_eval_cache}"
MANIFEST="${MANIFEST:-${CACHE_ROOT}/manifest.csv}"

# Labels must match prediction subfolders under CACHE_ROOT/predictions/<name>/ (see reward-eval-infer-*.slurm).
CHID_TOPREWARD="${CHID_TOPREWARD:-Qwen/Qwen3-VL-8B-Instruct}"
CHID_ROBODOPAMINE="${CHID_ROBODOPAMINE:-tanhuajie2001/Robo-Dopamine-GRM-3B}"
CHID_RBM="${CHID_RBM:-robometer/Robometer-4B}"
CHID_ROBOREWARD="${CHID_ROBOREWARD:-teetone/RoboReward-8B}"
CHID_SUCCESS_CKPT="${CHID_SUCCESS_CKPT:-/projects/bggq/asunesara/success_detector_runs/run3_attention_no_fp/best.pt}"

cd "${REWARD_DESIGN_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

run_metrics() {
  local backend_label="$1"
  local pred_subdir="$2"
  local checkpoint_id="$3"
  shift 3
  local pred_dir="${CACHE_ROOT}/predictions/${pred_subdir}"
  if [[ ! -d "${pred_dir}" ]]; then
    echo "Skipping ${backend_label}: missing predictions dir ${pred_dir}" >&2
    return 0
  fi
  echo "=== compute_metrics: ${backend_label} -> ${pred_dir} ==="
  python -m reward_eval.compute_metrics \
    --manifest "${MANIFEST}" \
    --predictions-dir "${pred_dir}" \
    --backend-label "${backend_label}" \
    --checkpoint-id "${checkpoint_id}" \
    "$@"
}

# Dense / time-varying progress predictions
run_metrics "topreward_qwen" "topreward_qwen" "${CHID_TOPREWARD}"

run_metrics "robodopamine" "robodopamine" "${CHID_ROBODOPAMINE}"

run_metrics "rbm" "rbm" "${CHID_RBM}"

# End-of-episode flat broadcast: use --eoe-flat so trajectory scalars follow mean(pred)/mean(gt) semantics.
run_metrics "roboreward" "roboreward" "${CHID_ROBOREWARD}" --eoe-flat

# Latent success detector: eval-scope auto -> success_only in summary when label contains success_detector
run_metrics "success_detector" "success_detector" "${CHID_SUCCESS_CKPT}"

echo "All metrics finished. Outputs: <predictions-dir>/metrics/ per backend under ${CACHE_ROOT}/predictions/."
