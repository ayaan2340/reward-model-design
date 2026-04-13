#!/usr/bin/env python3
"""Aggregate metrics from prediction npz files + manifest.

Progress RMs (default ``eval_scope=progress``):

  * **Temporal alignment (dense models):** ``spearman_intra_traj_mean`` — average within-trajectory
    Spearman(pred, GT) over time (scale-free). Skips constant (e.g. RoboReward) preds per clip.
  * **Calibration vs GT:** ``mae_dense_mean`` — mean over demos of frame MAE |pred−GT|; skips flat preds.
  * **Global pooled shape:** ``pearson_norm_frames`` — Pearson on per-traj min–max normalized frames
    (legacy / global view; can be weak for some baselines).
  * **Cross-episode rollout ranking:** ``kendall_tau_episode_means``, ``spearman_episode_means``,
    ``pearson_episode_means`` — compare per-episode **mean(pred)** vs **mean(GT)** across rollout
    demos (symmetric, interpretable; SciPy Kendall handles ties).
  * **Success:** AUROC/AUPRC (mean / last / max pred), pairwise S vs F.

``success_only`` (success-detector): AUROC/AUPRC + pairwise only.

RoboReward (flat pred): intra-traj Spearman is skipped; use **episode-mean** metrics and success AUROC.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from reward_eval.metric_utils import (
    compute_pearson,
    compute_spearman,
    kendall_tau_scipy,
    mean_dense_mae,
    mean_intra_trajectory_spearman,
    pooled_pearson_normalized_frames,
)

# Manifest mixes expert (linear-time GT) and rollout (cumulative-normalized GT); do not use a single gt_definition.
GT_DEFINITION_MIXED = "mixed: expert_ph=linear_time_expert, rollout_*=cumulative_normalized"

logger = logging.getLogger("compute_metrics")

FLAT_PRED_PTP_EPS = 1e-5


def _parse_success_label(val: str) -> int:
    s = str(val).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no", ""):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def traj_score(pred: np.ndarray, how: str = "sum") -> float:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if how == "sum":
        return float(pred.sum())
    if how == "mean":
        return float(pred.mean())
    raise ValueError(how)


def pred_is_eoe_flat(pred: np.ndarray) -> bool:
    p = np.asarray(pred, dtype=np.float64).ravel()
    if p.size == 0:
        return True
    return float(np.ptp(p)) < FLAT_PRED_PTP_EPS


def traj_pred_scalar(pred: np.ndarray, how: str) -> float:
    if pred_is_eoe_flat(pred):
        return float(np.mean(pred))
    return traj_score(pred, how)


def traj_gt_scalar(gt: np.ndarray, pred: np.ndarray, how: str) -> float:
    if pred_is_eoe_flat(pred):
        g = np.asarray(gt, dtype=np.float64).ravel()
        return float(np.mean(g)) if g.size else float("nan")
    return traj_score(gt, how)


def dataset_is_all_eoe_flat(by_split: dict[str, list[dict[str, Any]]]) -> bool:
    n = 0
    for demos in by_split.values():
        for d in demos:
            n += 1
            if not pred_is_eoe_flat(d["pred"]):
                return False
    return n > 0


def pairwise_accuracy(
    succ_preds: list[np.ndarray],
    fail_preds: list[np.ndarray],
    *,
    traj_score_how: str,
    max_pairs: int | None,
    pairwise_seed: int,
) -> tuple[float, int, int]:
    rng = np.random.RandomState(pairwise_seed)
    pairs = [(s, f) for s in range(len(succ_preds)) for f in range(len(fail_preds))]
    if max_pairs is not None and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    if not pairs:
        return float("nan"), 0, 0
    correct = 0
    for si, fi in pairs:
        ss = traj_pred_scalar(succ_preds[si], traj_score_how)
        fs = traj_pred_scalar(fail_preds[fi], traj_score_how)
        if ss >= fs:
            correct += 1
    return correct / len(pairs), correct, len(pairs)


def affine_fit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y = y_true.ravel()
    x = y_pred.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return 1.0, 0.0
    x_mean, y_mean = x.mean(), y.mean()
    var = np.sum((x - x_mean) ** 2)
    if var < 1e-12:
        return 1.0, float(y_mean - x_mean)
    a = float(np.sum((x - x_mean) * (y - y_mean)) / var)
    b = float(y_mean - a * x_mean)
    return a, b


def infer_eval_scope(backend_label: str) -> str:
    t = backend_label.lower()
    if "success_detector" in t:
        return "success_only"
    return "progress"


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute reward eval metrics from predictions (core suite).")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--predictions-dir", type=str, required=True, help="Directory with dataset/demo_key.npz")
    p.add_argument("--backend-label", type=str, default="", help="Label for CSV (default: folder name)")
    p.add_argument("--checkpoint-id", type=str, default="", help="HF id or path for reporting")
    p.add_argument("--traj-score", choices=("sum", "mean"), default="sum", help="Only for pairwise traj scalar when not EoE-flat")
    p.add_argument(
        "--pairwise-max-pairs",
        type=int,
        default=0,
        help="0 = all success×failure pairs; else sample this many",
    )
    p.add_argument("--pairwise-seed", type=int, default=0)
    p.add_argument(
        "--affine-calibration",
        action="store_true",
        help="Least-squares affine fit pred<-gt on expert_ph frames before metrics",
    )
    p.add_argument("--out-dir", type=str, default="", help="Default: <predictions-dir>/metrics")
    p.add_argument(
        "--eval-scope",
        choices=("auto", "progress", "success_only"),
        default="auto",
        help="auto: infer from --backend-label (success_detector -> success_only)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest_path = Path(args.manifest).expanduser()
    pred_root = Path(args.predictions_dir).expanduser()
    label = args.backend_label or pred_root.name
    out_dir = Path(args.out_dir) if args.out_dir else pred_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = args.eval_scope
    if scope == "auto":
        scope = infer_eval_scope(label)
    logger.info("eval_scope=%s (backend_label=%s)", scope, label)

    manifest_rows = {(r["dataset_name"], r["demo_key"]): r for r in load_manifest(manifest_path)}

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for npz_path in sorted(pred_root.rglob("*.npz")):
        if npz_path.name.endswith(".meta.json"):
            continue
        rel = npz_path.relative_to(pred_root)
        parts = rel.parts
        if len(parts) < 2:
            continue
        dataset_name, stem = parts[0], npz_path.stem
        key = (dataset_name, stem)
        if key not in manifest_rows:
            logger.warning("No manifest row for %s", key)
            continue
        row = manifest_rows[key]
        z = np.load(npz_path)
        pred = np.asarray(z["pred"], dtype=np.float64)
        gt = np.asarray(z["gt"], dtype=np.float64)
        split_tag = row["split_tag"]
        by_split[split_tag].append(
            {
                "dataset_name": dataset_name,
                "demo_key": stem,
                "pred": pred,
                "gt": gt,
                "success_label": _parse_success_label(row.get("success_label", "0")),
                "gt_definition": row.get("gt_definition", ""),
                "hdf5_done_mode": row.get("hdf5_done_mode", ""),
                "hdf5_task_success_attr": row.get("hdf5_task_success_attr", ""),
            }
        )

    eoe_flat_run = dataset_is_all_eoe_flat(by_split)
    if args.affine_calibration and eoe_flat_run:
        logger.warning("Disabling --affine-calibration: not meaningful for EoE-flat predictions.")
    if args.affine_calibration and not eoe_flat_run:
        exp = by_split.get("expert_ph", [])
        if exp:
            y_all = np.concatenate([d["gt"] for d in exp])
            p_all = np.concatenate([d["pred"] for d in exp])
            a, b = affine_fit(y_all, p_all)
            logger.info("Affine fit on expert: a=%.6f b=%.6f", a, b)
        else:
            a, b = 1.0, 0.0
    else:
        a, b = 1.0, 0.0

    use_affine = bool(args.affine_calibration and not eoe_flat_run)

    def scaled_pred(d: dict[str, Any]) -> np.ndarray:
        p = np.asarray(d["pred"], dtype=np.float64)
        return p * a + b if use_affine else p

    def scale_fn(d: dict[str, Any]) -> np.ndarray:
        return scaled_pred(d)

    summary_rows: list[dict[str, Any]] = []
    gt_def_global = GT_DEFINITION_MIXED
    calib_note = "affine_expert" if use_affine else "none"

    summary_rows.append(
        {
            "backend": label,
            "checkpoint_id": args.checkpoint_id,
            "dataset": "meta",
            "subset": "eval_scope",
            "metric_name": "eval_scope",
            "value": float("nan"),
            "gt_definition": gt_def_global,
            "pred_calibration": calib_note,
            "notes": scope,
        }
    )

    roll_tags = list(by_split.get("rollout_success", [])) + list(by_split.get("rollout_failure", []))
    all_demo_list: list[dict[str, Any]] = []
    for demos in by_split.values():
        all_demo_list.extend(demos)

    def add_auroc_block(
        demos: list[dict[str, Any]],
        *,
        dataset_key: str,
        subset_key: str,
        note_prefix: str,
    ) -> None:
        if not demos or len({d["success_label"] for d in demos}) < 2:
            for name in (
                "auroc_mean_pred",
                "auroc_last_frame_pred",
                "auroc_max_pred",
                "auprc_mean_pred",
                "auprc_last_frame_pred",
                "auprc_max_pred",
            ):
                summary_rows.append(
                    {
                        "backend": label,
                        "checkpoint_id": args.checkpoint_id,
                        "dataset": dataset_key,
                        "subset": subset_key,
                        "metric_name": name,
                        "value": float("nan"),
                        "gt_definition": gt_def_global,
                        "pred_calibration": calib_note,
                        "notes": "need both success classes",
                    }
                )
            return
        y_bin = np.array([int(d["success_label"]) for d in demos], dtype=np.int32)
        s_mean = np.array(
            [float(np.mean(np.asarray(scaled_pred(d), dtype=np.float64))) for d in demos],
            dtype=np.float64,
        )
        s_last = np.array(
            [float(np.asarray(scaled_pred(d), dtype=np.float64).ravel()[-1]) for d in demos],
            dtype=np.float64,
        )
        s_max = np.array(
            [float(np.max(np.asarray(scaled_pred(d), dtype=np.float64))) for d in demos],
            dtype=np.float64,
        )
        summary_rows.extend(
            [
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auroc_mean_pred",
                    "value": _safe_auroc(y_bin, s_mean),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} mean(frame pred); length-normalized for dense traces",
                },
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auroc_last_frame_pred",
                    "value": _safe_auroc(y_bin, s_last),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} pred at final timestep",
                },
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auroc_max_pred",
                    "value": _safe_auroc(y_bin, s_max),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} max_t pred (trajectory-wide signal)",
                },
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auprc_mean_pred",
                    "value": _safe_auprc(y_bin, s_mean),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} PR-AUC vs mean pred",
                },
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auprc_last_frame_pred",
                    "value": _safe_auprc(y_bin, s_last),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} PR-AUC vs last frame",
                },
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": dataset_key,
                    "subset": subset_key,
                    "metric_name": "auprc_max_pred",
                    "value": _safe_auprc(y_bin, s_max),
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": f"{note_prefix} PR-AUC vs max pred",
                },
            ]
        )

    # Pairwise (success vs failure rollouts) — meaningful for any scalar signal
    succ = [scaled_pred(d) for d in by_split.get("rollout_success", [])]
    fail = [scaled_pred(d) for d in by_split.get("rollout_failure", [])]
    max_pairs = args.pairwise_max_pairs if args.pairwise_max_pairs > 0 else None
    if not succ or not fail:
        p_acc, n_cor, n_tot = float("nan"), 0, 0
    else:
        p_acc, n_cor, n_tot = pairwise_accuracy(
            succ,
            fail,
            traj_score_how=args.traj_score,
            max_pairs=max_pairs,
            pairwise_seed=args.pairwise_seed,
        )
    summary_rows.append(
        {
            "backend": label,
            "checkpoint_id": args.checkpoint_id,
            "dataset": "rollout_mixed",
            "subset": "pairwise_sf",
            "metric_name": "pairwise_preference_accuracy",
            "value": p_acc,
            "gt_definition": gt_def_global,
            "pred_calibration": "none",
            "notes": f"P(success_rollout score >= failure_rollout); correct={n_cor} total={n_tot}",
        }
    )

    # Binary success metrics (rollouts and all demos)
    add_auroc_block(roll_tags, dataset_key="rollout_mixed", subset_key="binary_success", note_prefix="rollouts:")
    add_auroc_block(
        all_demo_list,
        dataset_key="all_demos",
        subset_key="binary_success",
        note_prefix="expert+rollouts:",
    )

    if scope == "progress":
        for tag in ("expert_ph", "rollout_success", "rollout_failure"):
            demos = by_split.get(tag, [])
            v_pool = pooled_pearson_normalized_frames(demos, scale_pred=scale_fn) if demos else float("nan")
            summary_rows.append(
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": "pooled",
                    "subset": tag,
                    "metric_name": "pearson_norm_frames",
                    "value": v_pool,
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": "Pearson on pooled per-traj min–max normalized pred & GT (global correlation)",
                }
            )
            v_intra = (
                mean_intra_trajectory_spearman(demos, scale_pred=scale_fn) if demos else float("nan")
            )
            summary_rows.append(
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": "pooled",
                    "subset": tag,
                    "metric_name": "spearman_intra_traj_mean",
                    "value": v_intra,
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": "Mean over demos of Spearman(pred,GT) along time; skips constant preds (EoE-flat)",
                }
            )
            v_mae = mean_dense_mae(demos, scale_pred=scale_fn) if demos else float("nan")
            summary_rows.append(
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": "pooled",
                    "subset": tag,
                    "metric_name": "mae_dense_mean",
                    "value": v_mae,
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": "Mean over demos of mean |pred−GT| per frame; skips flat preds",
                }
            )

        # Cross-episode ordering on rollouts: symmetric scalars mean(pred) vs mean(GT dense).
        if len(roll_tags) >= 2:
            pr_mean = [float(np.mean(np.asarray(scaled_pred(d), dtype=np.float64))) for d in roll_tags]
            gt_mean = [float(np.mean(np.asarray(d["gt"], dtype=np.float64))) for d in roll_tags]
            kt = kendall_tau_scipy(gt_mean, pr_mean)
            sp = compute_spearman(gt_mean, pr_mean)
            pe = compute_pearson(gt_mean, pr_mean)
        else:
            kt = float("nan")
            sp = float("nan")
            pe = float("nan")

        for mname, val, note in (
            (
                "kendall_tau_episode_means",
                kt,
                "SciPy Kendall τ between episode scalars mean(pred) and mean(GT) on rollout set",
            ),
            (
                "spearman_episode_means",
                sp,
                "Spearman rank corr: per-episode mean(pred) vs mean(GT) (rollouts)",
            ),
            (
                "pearson_episode_means",
                pe,
                "Pearson: per-episode mean(pred) vs mean(GT) (rollouts)",
            ),
        ):
            summary_rows.append(
                {
                    "backend": label,
                    "checkpoint_id": args.checkpoint_id,
                    "dataset": "rollout_mixed",
                    "subset": "episode_means",
                    "metric_name": mname,
                    "value": val,
                    "gt_definition": gt_def_global,
                    "pred_calibration": calib_note,
                    "notes": note,
                }
            )
    else:
        logger.info("success_only scope: skipping dense alignment metrics")

    summary_path = out_dir / "summary_metrics.csv"
    fields = [
        "backend",
        "checkpoint_id",
        "dataset",
        "subset",
        "metric_name",
        "value",
        "gt_definition",
        "pred_calibration",
        "notes",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    per_traj_path = out_dir / "per_trajectory_scores.csv"
    pt_fields = [
        "backend",
        "dataset_name",
        "demo_key",
        "split_tag",
        "pred_eoe_flat",
        "pred_last",
        "pred_mean",
        "pred_max",
        "gt_last",
        "gt_mean",
        "traj_score_pred",
        "traj_score_gt",
        "success_label",
        "hdf5_done_mode",
        "hdf5_task_success_attr",
    ]
    with open(per_traj_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=pt_fields)
        w.writeheader()
        for st, demos in by_split.items():
            for d in demos:
                pp = scaled_pred(d)
                flat = pred_is_eoe_flat(pp)
                g = np.asarray(d["gt"], dtype=np.float64).ravel()
                p = np.asarray(pp, dtype=np.float64).ravel()
                pl = float(p[-1]) if p.size else float("nan")
                gl = float(g[-1]) if g.size else float("nan")
                gm = float(np.mean(g)) if g.size else float("nan")
                pm = float(np.mean(p)) if p.size else float("nan")
                pmax = float(np.max(p)) if p.size else float("nan")
                w.writerow(
                    {
                        "backend": label,
                        "dataset_name": d["dataset_name"],
                        "demo_key": d["demo_key"],
                        "split_tag": st,
                        "pred_eoe_flat": int(flat),
                        "pred_last": pl,
                        "pred_mean": pm,
                        "pred_max": pmax,
                        "gt_last": gl,
                        "gt_mean": gm,
                        "traj_score_pred": traj_pred_scalar(pp, args.traj_score),
                        "traj_score_gt": traj_gt_scalar(d["gt"], pp, args.traj_score),
                        "success_label": d["success_label"],
                        "hdf5_done_mode": d.get("hdf5_done_mode", ""),
                        "hdf5_task_success_attr": d.get("hdf5_task_success_attr", ""),
                    }
                )

    logger.info("Wrote %s and %s", summary_path, per_traj_path)


if __name__ == "__main__":
    main()
