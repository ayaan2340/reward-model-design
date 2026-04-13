#!/usr/bin/env python3
"""Run a single reward backend over manifest trajectories; write per-demo npz predictions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from reward_eval.backends import RewardBackend, build_backend

logger = logging.getLogger("run_reward_inference")


def align_pred_length(pred: np.ndarray, target_len: int) -> tuple[np.ndarray, bool]:
    """Resize prediction to target_len via linear interpolation (preserves [0,1]-ish range)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if pred.size == target_len:
        return pred, False
    if pred.size == 0:
        return np.zeros(target_len, dtype=np.float64), True
    if pred.size == 1:
        return np.full(target_len, float(pred[0])), True
    x_old = np.linspace(0.0, 1.0, num=pred.size)
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, pred).astype(np.float64), True


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prediction_npz_valid(path: Path) -> bool:
    """True if existing npz has pred/gt with matching length (for --resume)."""
    try:
        z = np.load(path)
        if "pred" not in z.files or "gt" not in z.files:
            return False
        pr = np.asarray(z["pred"]).ravel()
        gt = np.asarray(z["gt"]).ravel()
        if pr.size == 0 or gt.size == 0 or pr.shape[0] != gt.shape[0]:
            return False
        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Reward model inference over preprocess manifest.")
    p.add_argument("--manifest", type=str, required=True, help="manifest.csv from preprocess_manifest")
    p.add_argument(
        "--predictions-dir",
        type=str,
        default="",
        help="Output root (default: <manifest_dir>/predictions/<backend>)",
    )
    p.add_argument(
        "--backend",
        type=str,
        required=True,
        help="topreward_qwen | robodopamine | roboreward | rbm | success_detector",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="HF id for TopReward / RoboDopamine / RoboReward (e.g. teetone/RoboReward-8B)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="HF or local dir for rbm (includes ReWiND checkpoints) or .pt for success_detector",
    )
    p.add_argument(
        "--latent-root",
        type=str,
        default="",
        help="success_detector: extract_latent_robomimic --output_path (fallback if per-dataset roots unset)",
    )
    p.add_argument(
        "--latent-root-expert",
        type=str,
        default="",
        help="success_detector: latent root for manifest dataset_name=expert_ph (~200 demos). Overrides --latent-root for that split.",
    )
    p.add_argument(
        "--latent-root-rollout",
        type=str,
        default="",
        help="success_detector: latent root for manifest dataset_name=rollout_mixed (~300 demos). Overrides --latent-root for that split.",
    )
    p.add_argument("--dataset-filter", type=str, default="", help="Only dataset_name == this (e.g. expert_ph)")
    p.add_argument("--split-filter", type=str, default="", help="Only split_tag == this")
    p.add_argument("--max-demos", type=int, default=0, help="Stop after N trajectories (debug)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip demos that already have a valid pred/gt npz (same length). Corrupt/incomplete files are recomputed.",
    )
    p.add_argument(
        "--rbm-max-frames",
        type=int,
        default=48,
        help="rbm only: max frames per model forward (uniform subsample, then interpolate to full T). "
        "Qwen3-VL uses huge memory for long videos; lower if OOM. Use 0 for no subsampling (may OOM).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    manifest_path = Path(args.manifest).expanduser()
    rows = load_manifest(manifest_path)
    backend_name = args.backend.lower().strip()

    kwargs: dict[str, Any] = {}
    if backend_name in ("topreward", "topreward_qwen"):
        if not args.model_path:
            raise SystemExit("--model-path required for TopReward (e.g. Qwen/Qwen3-VL-8B-Instruct)")
        kwargs["model_path"] = args.model_path
    elif backend_name == "robodopamine":
        mp = args.model_path or "tanhuajie2001/Robo-Dopamine-GRM-3B"
        kwargs["model_path"] = mp
    elif backend_name == "roboreward":
        kwargs["model_path"] = args.model_path or "teetone/RoboReward-8B"
    elif backend_name == "rbm":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for rbm (Robometer or ReWiND HF id / local path)")
        kwargs["checkpoint"] = args.checkpoint
        kwargs["max_frames"] = 0 if args.rbm_max_frames <= 0 else args.rbm_max_frames
    elif backend_name in ("success_detector", "success"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for success_detector")
        has_any_root = bool(
            args.latent_root.strip()
            or args.latent_root_expert.strip()
            or args.latent_root_rollout.strip()
        )
        if not has_any_root:
            raise SystemExit(
                "success_detector needs at least one of: --latent-root, or both "
                "--latent-root-expert and --latent-root-rollout (recommended when manifest mixes ~200 expert + ~300 rollout)."
            )
        kwargs["checkpoint"] = args.checkpoint
        kwargs["latent_root"] = args.latent_root
        kwargs["latent_root_expert"] = args.latent_root_expert
        kwargs["latent_root_rollout"] = args.latent_root_rollout
    else:
        raise SystemExit(f"Unknown backend {args.backend}")

    backend: RewardBackend = build_backend(
        "topreward_qwen" if backend_name in ("topreward", "topreward_qwen") else backend_name,
        **kwargs,
    )

    out_name = backend.name
    pred_root = Path(args.predictions_dir) if args.predictions_dir else manifest_path.parent / "predictions" / out_name
    pred_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for i, row in enumerate(rows):
        if args.dataset_filter and row.get("dataset_name") != args.dataset_filter:
            continue
        if args.split_filter and row.get("split_tag") != args.split_filter:
            continue

        ds = row["dataset_name"]
        demo_key = row["demo_key"]
        out_path = pred_root / ds / f"{demo_key}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and out_path.is_file():
            if _prediction_npz_valid(out_path):
                logger.info("Resume skip %s", out_path)
                continue
            logger.warning("Resume: invalid or incomplete %s — recomputing", out_path)

        frames_npz = Path(row["frames_npz"])
        gt_npy = Path(row["gt_npy"])
        data = np.load(frames_npz)
        rgb = data["rgb"]
        if rgb.ndim != 4:
            raise ValueError(f"Bad rgb shape {rgb.shape}")
        gt = np.load(gt_npy).astype(np.float64)
        t = int(gt.shape[0])
        task = row.get("task_instruction") or row.get("task", "")

        meta = {
            "dataset_name": ds,
            "demo_key": demo_key,
            "demo_id": int(row["demo_id"]),
            "T": t,
            "split_tag": row.get("split_tag", ""),
            "gt_definition": row.get("gt_definition", ""),
            "success_label": row.get("success_label", ""),
            "hdf5_done_mode": row.get("hdf5_done_mode", ""),
            "hdf5_task_success_attr": row.get("hdf5_task_success_attr", ""),
        }

        t0 = time.time()
        try:
            pred_raw, extra = backend.predict_dense(rgb, str(task), meta=meta)
        except Exception as e:
            logger.exception("Failed demo %s/%s: %s", ds, demo_key, e)
            err_path = out_path.with_suffix(".error.txt")
            err_path.write_text(repr(e), encoding="utf-8")
            continue

        pred, resized = align_pred_length(pred_raw, t)
        elapsed = time.time() - t0
        sidecar = {
            "backend": out_name,
            "demo_key": demo_key,
            "dataset_name": ds,
            "elapsed_sec": elapsed,
            "aligned_interpolation": resized,
            "extra": extra,
            "success_label": meta.get("success_label"),
            "hdf5_done_mode": meta.get("hdf5_done_mode"),
            "hdf5_task_success_attr": meta.get("hdf5_task_success_attr"),
        }
        np.savez_compressed(
            out_path,
            pred=pred.astype(np.float32),
            gt=gt.astype(np.float32),
            gt_definition=np.array(row.get("gt_definition", "")),
        )
        with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)
        logger.info("OK %s/%s pred_len=%d gt_len=%d (%.1fs)", ds, demo_key, pred.shape[0], t, elapsed)
        n_done += 1
        if args.max_demos and n_done >= args.max_demos:
            break

    logger.info("Finished %d trajectories -> %s", n_done, pred_root)


if __name__ == "__main__":
    main()
