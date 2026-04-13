#!/usr/bin/env python3
"""Plot summary_metrics.csv from reward_eval runs: one PNG per metric.

Progress charts (rbm, roboreward, topreward_qwen): intra-traj Spearman, pooled Pearson norm, dense MAE,
episode-mean Kendall/Spearman/Pearson on rollouts, pairwise. Success charts add success_detector.

Example:
  python -m reward_eval.plot_eval_metrics \\
    --predictions-root /path/to/reward_eval_cache/predictions \\
    --output-dir /path/to/reward_eval_cache/metric_plots
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

ChartKind = Literal["progress_alignment", "success_classification"]


def _float_or_nan(x: str) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")


def read_summary(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def include_backend(backend: str, kind: ChartKind) -> bool:
    b = backend.lower()
    if "robodopamine" in b:
        return False
    if kind == "progress_alignment" and "success_detector" in b:
        return False
    return True


def lookup_metric(
    rows: list[tuple[str, dict[str, str]]],
    backend: str,
    dataset: str,
    subset: str,
    metric_name: str,
) -> tuple[float, int]:
    """Return (value, match_count). Last row wins if duplicates exist."""
    best = float("nan")
    n = 0
    for be, r in rows:
        if be != backend:
            continue
        if r.get("dataset", "").strip() != dataset or r.get("subset", "").strip() != subset:
            continue
        if r.get("metric_name", "").strip() != metric_name:
            continue
        n += 1
        best = _float_or_nan(str(r.get("value", "nan")))
    return best, n


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot reward_eval summary_metrics (one PNG per chart).")
    ap.add_argument(
        "--predictions-root",
        type=str,
        default="",
        help="Parent of backend folders, each containing metrics/summary_metrics.csv",
    )
    ap.add_argument(
        "--summaries",
        type=str,
        nargs="*",
        default=[],
        help="Explicit paths to summary_metrics.csv (overrides --predictions-root if set)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to write metric_*.png files (required unless --no-save)",
    )
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print lookup keys, per-backend values, match counts, NaNs, and backend list",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write PNGs (useful with --debug)",
    )
    args = ap.parse_args()

    if not args.output_dir.strip() and not args.no_save:
        raise SystemExit("Provide --output-dir or use --no-save.")

    paths: list[Path] = []
    if args.summaries:
        paths = [Path(p).expanduser() for p in args.summaries]
    elif args.predictions_root:
        root = Path(args.predictions_root).expanduser()
        paths = sorted(root.glob("*/metrics/summary_metrics.csv"))
        if not paths:
            paths = sorted(root.rglob("summary_metrics.csv"))
    else:
        raise SystemExit("Provide --predictions-root or one or more --summaries paths.")

    if not paths:
        raise SystemExit("No summary_metrics.csv files found.")

    if args.debug:
        print("=== plot_eval_metrics debug ===")
        print("summary files (order defines bar order):")
        for p in paths:
            print(f"  {p}")
        print()

    backends: list[str] = []
    all_rows: list[tuple[str, dict[str, str]]] = []
    for p in paths:
        label = p.parent.parent.name if p.parent.name == "metrics" else p.parent.name
        if label not in backends:
            backends.append(label)
        for r in read_summary(p):
            all_rows.append((label, r))

    if args.debug:
        print(f"backend labels ({len(backends)}): {backends}")
        print()

    # (filename stem, title, dataset, subset, metric_name, chart kind)
    plot_specs: list[tuple[str, str, str, str, str, ChartKind]] = [
        (
            "pearson_norm_expert_ph",
            "Pearson: min–max norm pred vs GT (frames pooled)\nexpert PH",
            "pooled",
            "expert_ph",
            "pearson_norm_frames",
            "progress_alignment",
        ),
        (
            "pearson_norm_rollout_success",
            "Pearson: min–max norm pred vs GT (frames pooled)\nrollout success",
            "pooled",
            "rollout_success",
            "pearson_norm_frames",
            "progress_alignment",
        ),
        (
            "pearson_norm_rollout_failure",
            "Pearson: min–max norm pred vs GT (frames pooled)\nrollout failure",
            "pooled",
            "rollout_failure",
            "pearson_norm_frames",
            "progress_alignment",
        ),
        (
            "spearman_intra_expert_ph",
            "Mean intra-traj Spearman(pred, GT)\nexpert PH (temporal alignment)",
            "pooled",
            "expert_ph",
            "spearman_intra_traj_mean",
            "progress_alignment",
        ),
        (
            "spearman_intra_rollout_success",
            "Mean intra-traj Spearman(pred, GT)\nrollout success",
            "pooled",
            "rollout_success",
            "spearman_intra_traj_mean",
            "progress_alignment",
        ),
        (
            "spearman_intra_rollout_failure",
            "Mean intra-traj Spearman(pred, GT)\nrollout failure",
            "pooled",
            "rollout_failure",
            "spearman_intra_traj_mean",
            "progress_alignment",
        ),
        (
            "mae_dense_expert_ph",
            "Dense MAE: mean over demos of mean |pred−GT|\nexpert PH (flat preds skipped)",
            "pooled",
            "expert_ph",
            "mae_dense_mean",
            "progress_alignment",
        ),
        (
            "mae_dense_rollout_success",
            "Dense MAE: mean over demos of mean |pred−GT|\nrollout success",
            "pooled",
            "rollout_success",
            "mae_dense_mean",
            "progress_alignment",
        ),
        (
            "mae_dense_rollout_failure",
            "Dense MAE: mean over demos of mean |pred−GT|\nrollout failure",
            "pooled",
            "rollout_failure",
            "mae_dense_mean",
            "progress_alignment",
        ),
        (
            "kendall_episode_means_rollouts",
            "Kendall τ (SciPy): mean(pred) vs mean(GT) per rollout\nepisode scalars",
            "rollout_mixed",
            "episode_means",
            "kendall_tau_episode_means",
            "progress_alignment",
        ),
        (
            "spearman_episode_means_rollouts",
            "Spearman: mean(pred) vs mean(GT) per rollout episode",
            "rollout_mixed",
            "episode_means",
            "spearman_episode_means",
            "progress_alignment",
        ),
        (
            "pearson_episode_means_rollouts",
            "Pearson: mean(pred) vs mean(GT) per rollout episode",
            "rollout_mixed",
            "episode_means",
            "pearson_episode_means",
            "progress_alignment",
        ),
        (
            "pairwise_success_vs_failure",
            "Pairwise preference: P(score success rollout ≥ failure rollout)",
            "rollout_mixed",
            "pairwise_sf",
            "pairwise_preference_accuracy",
            "progress_alignment",
        ),
        (
            "auroc_mean_pred_rollouts",
            "AUROC vs binary success — mean(frame pred)\nrollout demos only",
            "rollout_mixed",
            "binary_success",
            "auroc_mean_pred",
            "success_classification",
        ),
        (
            "auroc_last_frame_rollouts",
            "AUROC vs binary success — last-frame pred\nrollout demos only",
            "rollout_mixed",
            "binary_success",
            "auroc_last_frame_pred",
            "success_classification",
        ),
        (
            "auroc_max_pred_rollouts",
            "AUROC vs binary success — max_t pred\nrollout demos only",
            "rollout_mixed",
            "binary_success",
            "auroc_max_pred",
            "success_classification",
        ),
        (
            "auroc_mean_pred_all_demos",
            "AUROC vs binary success — mean(frame pred)\nexpert + rollouts",
            "all_demos",
            "binary_success",
            "auroc_mean_pred",
            "success_classification",
        ),
        (
            "auprc_mean_pred_rollouts",
            "AUPRC vs binary success — mean(frame pred)\nrollout demos only",
            "rollout_mixed",
            "binary_success",
            "auprc_mean_pred",
            "success_classification",
        ),
    ]

    out_dir = Path(args.output_dir).expanduser()
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    for stem, title, ds, sub, mname, kind in plot_specs:
        included = [be for be in backends if include_backend(be, kind)]
        if not included:
            if args.debug:
                print(f"[{stem}] skipped: no backends after filter (kind={kind})")
            continue
        lookups = [lookup_metric(all_rows, be, ds, sub, mname) for be in included]
        vals = [v for v, _ in lookups]
        counts = [c for _, c in lookups]

        if args.debug:
            print(f"--- {stem} ---")
            print(f"  lookup: dataset={ds!r} subset={sub!r} metric_name={mname!r}  chart_kind={kind!r}")
            for be, v, cnt in zip(included, vals, counts):
                status = "OK" if cnt == 1 else ("MISSING" if cnt == 0 else f"DUP×{cnt}")
                v_str = f"{v:.6g}" if (v is not None and math.isfinite(v)) else "nan"
                print(f"    {be:20s}  value={v_str:>14s}  rows_matched={cnt}  [{status}]")
            n_bad = sum(1 for v in vals if v is None or not math.isfinite(v))
            if n_bad:
                print(
                    f"  NOTE: {n_bad} NaN/inf — bars use height 0 + 'n/a' (plotting 0 would fake a real score).",
                )
                if mname == "pearson_norm_frames":
                    print(
                        "  HINT: RoboReward is EoE-constant per episode → min–max pred is flat → pooled Pearson vs GT is undefined (var(pred_norm)=0).",
                    )
                if mname == "spearman_intra_traj_mean":
                    print(
                        "  HINT: NaN for RoboReward is expected (constant pred per clip → no temporal rank variation).",
                    )
                if mname in ("kendall_tau_episode_means", "spearman_episode_means", "pearson_episode_means"):
                    print(
                        "  HINT: Cross-episode metrics use mean(pred) vs mean(GT) per rollout; re-run compute_metrics if CSV still has old traj_last rows.",
                    )
            print()

        # Bar height: never use fake 0.0 for NaN on [-1,1] metrics (misleading).
        plot_heights: list[float] = []
        for v in vals:
            if v is not None and math.isfinite(v):
                plot_heights.append(float(v))
            else:
                plot_heights.append(0.0)

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(len(included))
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(included)))
        bars = ax.bar(x, plot_heights, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(included, rotation=28, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        if "AUROC" in title or "AUPRC" in title or "Pairwise" in title:
            ax.set_ylim(0.0, 1.05)
        elif "Dense MAE" in title:
            ymax_m = max((float(v) for v in vals if v is not None and math.isfinite(v)), default=0.0)
            ax.set_ylim(0.0, ymax_m * 1.2 + 0.02 if ymax_m > 0 else 1.0)
        elif "Kendall" in title or "Spearman" in title or "Pearson" in title:
            ax.set_ylim(-1.05, 1.05)

        y0, y1 = ax.get_ylim()
        span = y1 - y0 if y1 > y0 else 1.0
        for b, v, h in zip(bars, vals, plot_heights):
            if v is not None and math.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    min(float(v) + 0.02 * span, y1 - 0.02 * span),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    0.02 * span + y0,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )

        fig.tight_layout()
        if not args.no_save:
            outp = out_dir / f"{stem}.png"
            fig.savefig(outp, dpi=args.dpi, bbox_inches="tight")
            print(f"Wrote {outp}")
        plt.close(fig)


if __name__ == "__main__":
    main()
