#!/usr/bin/env python3
"""Generate bar charts from reward-eval ``summary_metrics.csv`` files.

Reads one CSV per backend under ``<predictions_root>/<subdir>/metrics/summary_metrics.csv``,
matches model colors to ``plot_reward_trajectory_videos.py``, and writes PNGs into the repo root
(``reward-model-design/`` by default).

Model inclusion rules (see module docstring in generated figures):
  * Frame-level agreement (Pearson, Spearman/Kendall vs GT, VOC): TopReward, Robometer,
    RoboDopamine — excludes RoboReward (EoE-flat / non-informative per-frame) and the success
    detector (binary success head, not a progress model).
  * Trajectory success classification & confusion metrics: TopReward, Robometer, RoboReward,
    Success detector — excludes RoboDopamine (no finite ``traj_success_pred`` in this eval).
  * Scalar prediction summaries (mean/min/max last-frame pred): same as frame-level, plus
    RoboReward to show near-constant trajectories.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# Align with plot_reward_trajectory_videos.py
COLOR_TOPREWARD = "#eb0707"
COLOR_ROBODOPAMINE = "#0729eb"
COLOR_RBM = "#38eb07"
COLOR_ROBOREWARD = "#9802f5"
COLOR_SUCCESS_DET = "#f1f502"

# Bar labels: 1.25× prior defaults (7.5 → 9.375, 7.0 → 8.75)
GROUPED_BAR_LABEL_FONTSIZE = 7.5 * 1.25
RATE_BAR_LABEL_FONTSIZE = 7.0 * 1.25
# Vertical gap from axis/spine to label (figure pixels; see ``data_y_after_pad_px_up``)
AXIS_LABEL_PAD_PX = 6.0

SUBDIR_TO_KEY = {
    "topreward_qwen": "topreward_qwen",
    "rbm": "rbm",
    "robodopamine": "robodopamine",
    "roboreward": "roboreward",
    "success_detector": "success_detector",
}

MODEL_META: dict[str, dict[str, str]] = {
    "topreward_qwen": {"label": "TopReward", "color": COLOR_TOPREWARD},
    "rbm": {"label": "Robometer", "color": COLOR_RBM},
    "robodopamine": {"label": "RoboDopamine", "color": COLOR_ROBODOPAMINE},
    "roboreward": {"label": "RoboReward", "color": COLOR_ROBOREWARD},
    "success_detector": {"label": "Success detector", "color": COLOR_SUCCESS_DET},
}

# Subset column in CSV -> short label on bars (no redundant "trajectory" wording)
SUBSET_SHORT: dict[str, str] = {
    "pearson": "Hybrid",
    "pearson_expert_ph": "Expert",
    "pearson_rollout_success": "Success",
    "pearson_rollout_failure": "Failure",
    "rankcorr": "Hybrid",
    "rankcorr_expert_ph": "Expert",
    "rankcorr_nonexpert": "Non-expert",
    "rankcorr_rollout_success": "Success",
    "rankcorr_rollout_failure": "Failure",
    "voc": "All",
    "voc_expert_ph": "Expert",
    "voc_rollout_success": "Success",
    "voc_rollout_failure": "Failure",
    "success_det_all": "All",
    "success_det_expert_ph": "Expert",
    "success_det_nonexpert_rollouts": "Non-expert",
    "success_det_rollout_success": "Success",
    "success_det_rollout_failure": "Failure",
}

PEARSON_SUBSETS = ("pearson", "pearson_expert_ph", "pearson_rollout_success", "pearson_rollout_failure")
RANK_SUBSETS = (
    "rankcorr",
    "rankcorr_expert_ph",
    "rankcorr_nonexpert",
    "rankcorr_rollout_success",
    "rankcorr_rollout_failure",
)
VOC_SUBSETS = ("voc", "voc_expert_ph", "voc_rollout_success", "voc_rollout_failure")
SUCCESS_DET_SUBSETS = (
    "success_det_all",
    "success_det_expert_ph",
    "success_det_nonexpert_rollouts",
    "success_det_rollout_success",
    "success_det_rollout_failure",
)

# Linear GT metrics are undefined for rollout_failure rows (NaN in CSV)
LINEAR_METRICS = frozenset(
    {
        "mean_pearson_linear_time_pred_raw",
        "mean_pearson_linear_time_pred_delta",
        "mean_spearman_linear_time_pred_raw",
        "mean_kendall_linear_time_pred_raw",
        "mean_spearman_linear_time_pred_delta",
        "mean_kendall_linear_time_pred_delta",
    }
)


def load_summary_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_val(s: str) -> float | None:
    s = (s or "").strip()
    if not s or s.lower() == "nan":
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def index_metrics(rows: list[dict[str, str]]) -> dict[tuple[str, str], float | None]:
    out: dict[tuple[str, str], float | None] = {}
    for r in rows:
        sub = str(r.get("subset", "")).strip()
        name = str(r.get("metric_name", "")).strip()
        out[(sub, name)] = parse_val(str(r.get("value", "")))
    return out


def y_bounds_for_metric(metric_name: str) -> tuple[float, float] | None:
    """Fixed axis limits when the statistic has a standard bounded range; else autoscale."""
    # Correlation-style metrics ∈ [-1, 1]
    if (
        metric_name.startswith("mean_pearson_")
        or metric_name.startswith("mean_spearman_")
        or metric_name.startswith("mean_kendall_")
        or metric_name == "mean_voc"
    ):
        return (-1.0, 1.0)
    # Rates / accuracies ∈ [0, 1]
    if metric_name.startswith("success_rate_") or metric_name == "mean_success_pred_abs_match":
        return (0.0, 1.0)
    if metric_name.startswith("success_mean_success_early_vs_last_"):
        return (0.0, 1.0)
    return None


def shade_palette(base_hex: str, n: int) -> list[tuple[float, float, float]]:
    """Return ``n`` RGB tuples from saturated base to lighter (blend toward white)."""
    r, g, b = to_rgb(base_hex)
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        t = i / max(1, n - 1) if n > 1 else 0.0
        # i=0 darkest (slightly darkened), i=n-1 lightest
        u = 0.15 + 0.85 * (1.0 - t)  # darkness factor
        wr = r * u + (1 - u) * 0.98
        wg = g * u + (1 - u) * 0.98
        wb = b * u + (1 - u) * 0.98
        out.append((wr, wg, wb))
    return out


def axis_baseline_y(ax: Any) -> float:
    """Y-value of the bottom horizontal axis line in data coordinates (visible spine)."""
    y0_ax, y1_ax = ax.get_ylim()
    if y0_ax < 0.0 < y1_ax:
        return 0.0
    if y0_ax >= 0.0:
        return float(y0_ax)
    return float(y1_ax)


def data_y_after_pad_px_up(ax: Any, fig: Any, y_line: float, pad_px: float) -> float:
    """Return data y that is ``pad_px`` **figure pixels** above ``y_line`` (larger y in data).

    ``ax.transData`` maps larger data *y* to *smaller* display *y* (y-down screen coords). So adding pixels
    to the display *y* from ``transform((x, y_line))`` moves the point **up** the plot and yields a **larger**
    data *y* after inversion — the direction we want. (Subtracting display y was incorrect and placed text
    **below** the spine.)
    """
    fig.canvas.draw()
    _, y_disp = ax.transData.transform((0.0, float(y_line)))
    y_disp_padded = float(y_disp) + float(pad_px)
    _, y_data = ax.transData.inverted().transform((0.0, y_disp_padded))
    return float(y_data)


def place_vertical_bar_label(
    ax: Any,
    bx: float,
    h: float,
    lab: str,
    bar_rgb: tuple[float, float, float],
    *,
    fontsize: float = GROUPED_BAR_LABEL_FONTSIZE,
    y_text_data: float,
) -> None:
    """Vertical label with bottom of text at ``y_text_data`` (precomputed with :func:`data_y_after_pad_px_up`)."""
    del bar_rgb  # kept for call-site compatibility; labels use a single high-contrast color
    del h  # bar height unused; anchor is always from the axis / plot floor
    ax.text(
        bx,
        y_text_data,
        lab,
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=fontsize,
        color="#1a1a1a",
        clip_on=False,
    )


def grouped_bar_plot(
    *,
    models: list[str],
    subsets: tuple[str, ...],
    metric_name: str,
    data: dict[str, dict[tuple[str, str], float | None]],
    title: str,
    ylabel: str,
    outfile: Path,
    figsize: tuple[float, float] = (9.5, 4.8),
    ylim: tuple[float | None, float | None] = (None, None),
    skip_subset_for_metric: frozenset[tuple[str, str]] | None = None,
) -> None:
    """One grouped bar chart: x = models, within each model = touching bars for each subset."""
    n_m = len(models)
    n_s = len(subsets)
    bar_w = 0.2
    offsets = np.linspace(-(n_s - 1) * bar_w / 2, (n_s - 1) * bar_w / 2, n_s)
    group_gap = 0.65
    x_centers = np.arange(n_m, dtype=np.float64) * (n_s * bar_w + group_gap)

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    all_vals: list[float] = []
    label_specs: list[tuple[float, float, str, tuple[float, float, float]]] = []

    for mi, key in enumerate(models):
        base = MODEL_META[key]["color"]
        shades = shade_palette(base, n_s)
        x0 = x_centers[mi]
        for si, sub in enumerate(subsets):
            val = data[key].get((sub, metric_name))
            skip = skip_subset_for_metric and (sub, metric_name) in skip_subset_for_metric
            if skip or val is None:
                continue
            h = float(val)
            all_vals.append(h)
            bx = x0 + offsets[si]
            ax.bar(
                bx,
                h,
                width=bar_w * 0.92,
                color=shades[si],
                edgecolor="#222222",
                linewidth=0.35,
            )
            label_specs.append((bx, h, SUBSET_SHORT.get(sub, sub), shades[si]))

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_META[m]["label"] for m in models], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    fixed = y_bounds_for_metric(metric_name)
    if ylim[0] is not None or ylim[1] is not None:
        ax.set_ylim(ylim[0], ylim[1])
    elif fixed is not None:
        ax.set_ylim(fixed[0], fixed[1])
    elif all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    y_line = axis_baseline_y(ax)
    y_text = data_y_after_pad_px_up(ax, fig, y_line, AXIS_LABEL_PAD_PX)
    for bx, h, lab, bar_rgb in label_specs:
        place_vertical_bar_label(
            ax,
            bx,
            h,
            lab,
            bar_rgb,
            fontsize=GROUPED_BAR_LABEL_FONTSIZE,
            y_text_data=y_text,
        )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def linear_skip_pairs(metric_name: str) -> frozenset[tuple[str, str]]:
    """Linear-time metrics are not defined on rollout_failure subgroup."""
    if metric_name not in LINEAR_METRICS:
        return frozenset()
    return frozenset(
        {
            ("pearson_rollout_failure", metric_name),
            ("rankcorr_rollout_failure", metric_name),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("/scratch/bggq/asunesara/reward_eval_cache/predictions"),
        help="Directory containing per-backend folders with metrics/summary_metrics.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Output directory for PNG files (default: reward-model-design/)",
    )
    args = ap.parse_args()

    root: Path = args.predictions_root
    out_dir: Path = args.out_dir

    # Discover backends
    data: dict[str, dict[tuple[str, str], float | None]] = {}
    for subdir, bkey in SUBDIR_TO_KEY.items():
        csv_path = root / subdir / "metrics" / "summary_metrics.csv"
        if not csv_path.is_file():
            continue
        data[bkey] = index_metrics(load_summary_csv(csv_path))

    if not data:
        raise SystemExit(f"No summary_metrics.csv found under {root}")

    frame_models = [m for m in ("topreward_qwen", "rbm", "robodopamine") if m in data]
    success_models = [m for m in ("topreward_qwen", "rbm", "roboreward", "success_detector") if m in data]
    pred_stat_models = [m for m in ("topreward_qwen", "rbm", "robodopamine", "roboreward") if m in data]

    # --- Pearson (4 metrics × 1 file)
    pearson_metrics = [
        ("mean_pearson_linear_time_pred_raw", "Pearson r vs linear-time GT (raw pred)", "Pearson r"),
        ("mean_pearson_linear_time_pred_delta", "Pearson r vs linear-time GT (delta pred)", "Pearson r"),
        ("mean_pearson_sim_reward_pred_raw", "Pearson r vs simulator step rewards (raw pred)", "Pearson r"),
        ("mean_pearson_sim_reward_pred_delta", "Pearson r vs simulator step rewards (delta pred)", "Pearson r"),
    ]
    for metric, title, ylab in pearson_metrics:
        skip = linear_skip_pairs(metric)
        grouped_bar_plot(
            models=frame_models,
            subsets=PEARSON_SUBSETS,
            metric_name=metric,
            data=data,
            title=title,
            ylabel=ylab,
            outfile=out_dir / f"summary_metric_{metric.removeprefix('mean_')}.png",
            skip_subset_for_metric=skip,
        )

    # --- Spearman / Kendall (8 files each family)
    spearman_metrics = [
        ("mean_spearman_linear_time_pred_raw", "Spearman ρ vs linear-time GT (raw pred)", "Spearman ρ"),
        ("mean_spearman_linear_time_pred_delta", "Spearman ρ vs linear-time GT (delta pred)", "Spearman ρ"),
        ("mean_spearman_sim_reward_pred_raw", "Spearman ρ vs simulator rewards (raw pred)", "Spearman ρ"),
        ("mean_spearman_sim_reward_pred_delta", "Spearman ρ vs simulator rewards (delta pred)", "Spearman ρ"),
    ]
    for metric, title, ylab in spearman_metrics:
        skip = linear_skip_pairs(metric)
        grouped_bar_plot(
            models=frame_models,
            subsets=RANK_SUBSETS,
            metric_name=metric,
            data=data,
            title=title,
            ylabel=ylab,
            outfile=out_dir / f"summary_metric_{metric.removeprefix('mean_')}.png",
            skip_subset_for_metric=skip,
        )

    kendall_metrics = [
        ("mean_kendall_linear_time_pred_raw", "Kendall τ vs linear-time GT (raw pred)", "Kendall τ"),
        ("mean_kendall_linear_time_pred_delta", "Kendall τ vs linear-time GT (delta pred)", "Kendall τ"),
        ("mean_kendall_sim_reward_pred_raw", "Kendall τ vs simulator rewards (raw pred)", "Kendall τ"),
        ("mean_kendall_sim_reward_pred_delta", "Kendall τ vs simulator rewards (delta pred)", "Kendall τ"),
    ]
    for metric, title, ylab in kendall_metrics:
        skip = linear_skip_pairs(metric)
        grouped_bar_plot(
            models=frame_models,
            subsets=RANK_SUBSETS,
            metric_name=metric,
            data=data,
            title=title,
            ylabel=ylab,
            outfile=out_dir / f"summary_metric_{metric.removeprefix('mean_')}.png",
            skip_subset_for_metric=skip,
        )

    # --- VOC
    grouped_bar_plot(
        models=frame_models,
        subsets=VOC_SUBSETS,
        metric_name="mean_voc",
        data=data,
        title="VOC: Spearman(time index, predicted score) per trajectory (mean over trajectories)",
        ylabel="VOC (Spearman ρ)",
        outfile=out_dir / "summary_metric_voc_mean.png",
    )

    # --- Binary trajectory success match (models with success head)
    def single_bar_models(
        models: list[str],
        metric_name: str,
        title: str,
        ylabel: str,
        fname: str,
        ylim: tuple[float | None, float | None] = (0.0, 1.0),
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
        x = np.arange(len(models))
        for i, m in enumerate(models):
            v = data[m].get(("success_pred_accuracy", metric_name))
            h = float(v) if v is not None else float("nan")
            ax.bar(
                i,
                h,
                color=MODEL_META[m]["color"],
                edgecolor="#222",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_META[m]["label"] for m in models], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        b = y_bounds_for_metric(metric_name)
        if b is not None:
            ax.set_ylim(b[0], b[1])
        else:
            ax.set_ylim(ylim[0], ylim[1])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    single_bar_models(
        success_models,
        "mean_success_pred_abs_match",
        "Trajectory success label: agreement with manifest (binary)",
        "Accuracy",
        "summary_metric_success_pred_accuracy.png",
    )

    # --- Success detection rates (subset = success_det_all)
    rate_metrics = [
        ("success_rate_accuracy", "Accuracy"),
        ("success_rate_tpr_recall", "Recall (TPR)"),
        ("success_rate_tnr_specificity", "Specificity (TNR)"),
        ("success_rate_precision", "Precision"),
    ]
    sub0 = "success_det_all"
    n_r = len(rate_metrics)
    n_m = len(success_models)
    bar_w = 0.18
    offs = np.linspace(-(n_r - 1) * bar_w / 2, (n_r - 1) * bar_w / 2, n_r)
    gap = 0.55
    xc = np.arange(n_m) * (n_r * bar_w + gap)

    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=140)
    rate_label_specs: list[tuple[float, float, str, tuple[float, float, float]]] = []
    for mi, m in enumerate(success_models):
        base = MODEL_META[m]["color"]
        shades = shade_palette(base, n_r)
        for ri, (metric, lab) in enumerate(rate_metrics):
            v = data[m].get((sub0, metric))
            h = float(v) if v is not None else 0.0
            bx = xc[mi] + offs[ri]
            ax.bar(
                bx,
                h,
                width=bar_w * 0.92,
                color=shades[ri],
                edgecolor="#222",
                linewidth=0.35,
            )
            rate_label_specs.append((bx, h, lab, shades[ri]))
    ax.set_xticks(xc)
    ax.set_xticklabels([MODEL_META[m]["label"] for m in success_models], fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Success prediction vs manifest (all trajectories with finite pred success)", fontsize=12, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    y_line_rates = axis_baseline_y(ax)
    y_text_rates = data_y_after_pad_px_up(ax, fig, y_line_rates, AXIS_LABEL_PAD_PX)
    for bx, h, lab, bar_rgb in rate_label_specs:
        place_vertical_bar_label(
            ax,
            bx,
            h,
            lab,
            bar_rgb,
            fontsize=RATE_BAR_LABEL_FONTSIZE,
            y_text_data=y_text_rates,
        )
    fig.savefig(out_dir / "summary_metric_success_det_rates_all.png", bbox_inches="tight")
    plt.close(fig)

    # --- Grouped success metrics by subset: accuracy / recall / precision where defined
    for metric_key, ylab, fname_suffix in [
        ("success_rate_accuracy", "Accuracy", "accuracy"),
        ("success_rate_tpr_recall", "Recall (TPR)", "recall_tpr"),
        ("success_rate_tnr_specificity", "Specificity (TNR)", "specificity_tnr"),
        ("success_rate_precision", "Precision", "precision"),
    ]:
        grouped_bar_plot(
            models=success_models,
            subsets=SUCCESS_DET_SUBSETS,
            metric_name=metric_key,
            data=data,
            title=f"{ylab} by trajectory subset (finite success prediction)",
            ylabel=ylab,
            outfile=out_dir / f"summary_metric_success_det_{fname_suffix}_by_subset.png",
        )

    # --- Confusion counts (all subset) — stacked bars TP/TN/FP/FN normalized to show composition
    count_keys = [
        ("success_count_tp", "TP"),
        ("success_count_tn", "TN"),
        ("success_count_fp", "FP"),
        ("success_count_fn", "FN"),
    ]
    sub_c = "success_det_all"
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=140)
    bottoms = np.zeros(n_m)
    cmap = ["#2d6a4f", "#40916c", "#e85d04", "#9d0208"]
    for ki, (mk, lab) in enumerate(count_keys):
        vals = []
        for m in success_models:
            v = data[m].get((sub_c, mk))
            vals.append(float(v) if v is not None else 0.0)
        vals = np.array(vals)
        ax.bar(
            np.arange(n_m),
            vals,
            bottom=bottoms,
            color=cmap[ki],
            edgecolor="#111",
            linewidth=0.3,
            label=lab,
        )
        bottoms += vals
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([MODEL_META[m]["label"] for m in success_models], fontsize=11)
    ax.set_ylabel("Trajectory count", fontsize=11)
    ax.set_title("Success prediction confusion counts (all trajectories, finite pred success)", fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_metric_success_det_confusion_counts_stacked.png", bbox_inches="tight")
    plt.close(fig)

    # --- Lead time (GT success trajectories): success_det_all
    lead_metrics = [
        ("success_mean_lead_frames_gt_minus_first", "Mean lead (frames): GT success − first pred success"),
        ("success_mean_lead_frames_gt_minus_timing", "Mean lead (frames): GT success − timing pred success"),
    ]
    for mk, ttitle in lead_metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
        vals = []
        for i, m in enumerate(success_models):
            v = data[m].get(("success_det_all", mk))
            h = float(v) if v is not None else float("nan")
            ax.bar(i, h, color=MODEL_META[m]["color"], edgecolor="#222", linewidth=0.4)
            if math.isfinite(h):
                vals.append(h)
        ax.set_xticks(np.arange(n_m))
        ax.set_xticklabels([MODEL_META[m]["label"] for m in success_models], fontsize=11)
        ax.set_ylabel("Frames", fontsize=11)
        ax.set_title(ttitle, fontsize=12, pad=10)
        if vals:
            ax.set_ylim(0, max(vals) * 1.12)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"summary_metric_{mk.removeprefix('success_mean_')}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    # --- Early success signal vs last frame (success_det_all)
    early_metrics = [
        ("success_mean_success_early_vs_last_1", "Early vs last-frame success pred (window=1)"),
        ("success_mean_success_early_vs_last_5", "Early vs last-frame success pred (window=5)"),
        ("success_mean_success_early_vs_last_10", "Early vs last-frame success pred (window=10)"),
    ]
    for mk, ttitle in early_metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
        for i, m in enumerate(success_models):
            v = data[m].get(("success_det_all", mk))
            h = float(v) if v is not None else float("nan")
            ax.bar(i, h, color=MODEL_META[m]["color"], edgecolor="#222", linewidth=0.4)
        ax.set_xticks(np.arange(n_m))
        ax.set_xticklabels([MODEL_META[m]["label"] for m in success_models], fontsize=11)
        ax.set_ylabel("Agreement rate", fontsize=11)
        ax.set_title(ttitle + " — GT success trajectories", fontsize=12, pad=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"summary_metric_{mk.removeprefix('success_mean_')}.png", bbox_inches="tight")
        plt.close(fig)

    # --- Prediction value summaries (hybrid / all trajectories) — includes RoboReward
    pred_keys = [
        ("mean_pred_mean", "Mean of per-trajectory mean frame pred"),
        ("mean_pred_last", "Mean of per-trajectory last-frame pred"),
        ("mean_pred_min", "Mean of per-trajectory min frame pred"),
        ("mean_pred_max", "Mean of per-trajectory max frame pred"),
    ]
    for metric, desc in pred_keys:
        grouped_bar_plot(
            models=pred_stat_models,
            subsets=("rankcorr", "rankcorr_expert_ph", "rankcorr_nonexpert", "rankcorr_rollout_success"),
            metric_name=metric,
            data=data,
            title=f"{desc} (scalar summaries of dense predictions)",
            ylabel="Pred value (model units)",
            outfile=out_dir / f"summary_metric_pred_stats_{metric.removeprefix('mean_')}.png",
        )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
