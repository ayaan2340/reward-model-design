#!/usr/bin/env python3
"""Export per-trajectory stats from predictions + preprocess manifest.

Preprocess writes three *progress* curves from the same episode length (and raw step rewards):
``gt_dense.npy`` (primary target, chosen per split), ``gt_linear_time.npy`` (always t/(T-1)),
``gt_cumulative_normalized.npy`` (always cumsum(rewards)/sum(rewards)). The primary curve is
*identical* to exactly one of the linear-time or cumulative columns (see ``dense_target_kind``);
we do not duplicate those scalars as extra ``gt_primary_*`` columns. Raw ``frames.npz`` ``rewards``
are the simulator/shaped *step* rewards — different units and meaning than the [0, 1] progress curves.
In this script only, per-trajectory ``simulator_reward_*`` columns and Pearson vs simulator use
min–max normalized rewards in ``[0, 1]`` (manifest / ``frames.npz`` on disk stay raw).

For time-varying predictions (not ``--eoe-flat`` and not numerically flat preds), Pearson r is computed
per trajectory vs ``gt_linear_time`` and vs min–max normalized simulator rewards: raw pairs use min-max
normalized pred per trajectory; delta pairs use ``(x[t]-x[t-1])/(max(x)-min(x))`` per series.

Linear-time Pearson is **not** computed for ``split_tag == rollout_failure`` (task did not succeed; linear
progress-to-goal is not meaningful there). Simulator-reward Pearson is still computed for those trajectories.
Summary rows include overall means (linear: expert + rollout successes only; sim: all trajectories) and
subgroups ``pearson_expert_ph``, ``pearson_rollout_success``, ``pearson_rollout_failure`` (failure subset:
sim metrics only).

**VOC (value–order correlation)** (Ma et al., 2024; Lee et al., 2026) is Spearman rank correlation
between chronological frame indices ``(0, 1, …, K-1)`` and the per-frame predicted values
``(s_0, …, s_{K-1})`` — i.e. rank agreement between video order and predicted score order. It does **not**
use ground-truth progress. Flat / EoE-flat preds yield NaN. Subgroup VOC means average this per-trajectory
Spearman over the listed splits.

**Spearman / Kendall** use the same per-frame pairings as Pearson (min–max ``pred`` vs ``gt_linear_time`` /
simulator rewards; plus delta streams). Linear-time rank metrics are omitted for ``rollout_failure`` (NaN).
Summary subgroups: all (hybrid), expert, non-expert (both rollouts), rollout success, rollout failure.

**Pred distribution** (per-trajectory ``pred_mean`` / ``pred_min`` / ``pred_max`` / ``pred_last``) is summarized with the
same five subsets as rank correlation: ``rankcorr`` (all trajectories), ``rankcorr_expert_ph``,
``rankcorr_nonexpert``, ``rankcorr_rollout_success``, ``rankcorr_rollout_failure``. Emits mean of per-traj
means/mins/maxes, max of per-traj maxes, min of per-traj mins, and mean/min/max of per-traj last-frame preds.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reward_eval.metric_utils import compute_spearman, kendall_tau_scipy

logger = logging.getLogger("compute_metrics")

FLAT_PRED_PTP_EPS = 1e-5

SUMMARY_CSV_FIELDS = [
    "backend",
    "checkpoint_id",
    "dataset",
    "subset",
    "metric_name",
    "value",
    "pred_calibration",
    "notes",
]

PER_TRAJECTORY_CSV_FIELDS = [
    "backend",
    "dataset_name",
    "demo_key",
    "split_tag",
    "dense_target_kind",
    "pred_eoe_flat",
    "pred_last",
    "pred_mean",
    "pred_min",
    "pred_max",
    "gt_linear_time_last",
    "gt_linear_time_mean",
    "gt_cumulative_normalized_last",
    "gt_cumulative_normalized_mean",
    "simulator_reward_last",
    "simulator_reward_mean",
    "simulator_reward_sum",
    "traj_score_pred",
    "traj_score_gt",
    "pearson_linear_time_pred_raw",
    "pearson_linear_time_pred_delta",
    "pearson_sim_reward_pred_raw",
    "pearson_sim_reward_pred_delta",
    "voc",
    "spearman_linear_time_pred_raw",
    "kendall_linear_time_pred_raw",
    "spearman_linear_time_pred_delta",
    "kendall_linear_time_pred_delta",
    "spearman_sim_reward_pred_raw",
    "kendall_sim_reward_pred_raw",
    "spearman_sim_reward_pred_delta",
    "kendall_sim_reward_pred_delta",
    "success_label",
    "traj_success_pred",
    "success_pred_abs_match",
    "traj_length_T",
    "gt_success_frame_idx",
    "success_pred_first_frame",
    "success_pred_timing_frame",
    "success_time_lead_gt_minus_first",
    "success_time_lead_gt_minus_timing",
    "success_cm",
    "success_early_vs_last_1",
    "success_early_vs_last_5",
    "success_early_vs_last_10",
    "hdf5_done_mode",
    "hdf5_task_success_attr",
]

PEARSON_TRAJ_KEYS = (
    "pearson_linear_time_pred_raw",
    "pearson_linear_time_pred_delta",
    "pearson_sim_reward_pred_raw",
    "pearson_sim_reward_pred_delta",
)

PEARSON_LINEAR_KEYS = PEARSON_TRAJ_KEYS[:2]
PEARSON_SIM_KEYS = PEARSON_TRAJ_KEYS[2:]

RANK_TRAJ_KEYS = (
    "spearman_linear_time_pred_raw",
    "kendall_linear_time_pred_raw",
    "spearman_linear_time_pred_delta",
    "kendall_linear_time_pred_delta",
    "spearman_sim_reward_pred_raw",
    "kendall_sim_reward_pred_raw",
    "spearman_sim_reward_pred_delta",
    "kendall_sim_reward_pred_delta",
)
RANK_LINEAR_KEYS = RANK_TRAJ_KEYS[:4]
RANK_SIM_KEYS = RANK_TRAJ_KEYS[4:]

EARLY_LAST_FRAME_KS: tuple[int, ...] = (1, 5, 10)

ALL_CORR_KEYS: tuple[str, ...] = tuple(dict.fromkeys(PEARSON_TRAJ_KEYS + RANK_TRAJ_KEYS))


@dataclass(frozen=True)
class RowBuckets:
    """Pre-indexed ``per_traj_rows`` by ``split_tag`` to avoid repeated full scans in summary aggregation."""

    all_rows: list[dict[str, Any]]
    expert_ph: list[dict[str, Any]]
    rollout_success: list[dict[str, Any]]
    rollout_failure: list[dict[str, Any]]
    nonexpert: list[dict[str, Any]]


def build_row_buckets(per_traj_rows: list[dict[str, Any]]) -> RowBuckets:
    expert_ph: list[dict[str, Any]] = []
    rollout_success: list[dict[str, Any]] = []
    rollout_failure: list[dict[str, Any]] = []
    for r in per_traj_rows:
        st = str(r.get("split_tag", "")).strip()
        if st == "expert_ph":
            expert_ph.append(r)
        elif st == "rollout_success":
            rollout_success.append(r)
        elif st == "rollout_failure":
            rollout_failure.append(r)
    nonexpert = rollout_success + rollout_failure
    return RowBuckets(per_traj_rows, expert_ph, rollout_success, rollout_failure, nonexpert)


def include_linear_gt_pearson_for_split(split_tag: str) -> bool:
    """Whether linear-time GT Pearson is meaningful for this trajectory.

    Failed rollouts never completed the task; linear-time correlation vs preds is omitted (NaN in per-row
    columns and excluded from linear-time summary means). Simulator-reward Pearson is still computed.
    """
    return str(split_tag).strip() != "rollout_failure"


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


def _npz_optional_float(z: Any, key: str) -> float:
    """Load optional 0-d or length-1 float from npz; NaN if missing."""
    files = getattr(z, "files", ())
    if key not in files:
        return float("nan")
    a = np.asarray(z[key]).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(a[0])


def _success_pred_match(traj_success_pred: float, success_label: int) -> float:
    """1.0 if binary prediction matches label, NaN if prediction is unknown."""
    if traj_success_pred != traj_success_pred:  # NaN
        return float("nan")
    pred_bin = 1 if float(traj_success_pred) >= 0.5 else 0
    return 1.0 if pred_bin == int(success_label) else 0.0


def _gt_success_frame_idx(traj_T: int, success_label: int) -> float:
    """Ground-truth success at last frame index when ``success_label`` is 1 (manifest alignment)."""
    if int(success_label) != 1 or traj_T <= 0:
        return float("nan")
    return float(traj_T - 1)


def _success_confusion_bucket(trsp: float, success_label: int) -> str:
    if trsp != trsp:
        return ""
    pb = 1 if float(trsp) >= 0.5 else 0
    gt = int(success_label)
    if gt == 1 and pb == 1:
        return "tp"
    if gt == 1 and pb == 0:
        return "fn"
    if gt == 0 and pb == 1:
        return "fp"
    return "tn"


def _early_vs_last_k(
    success_label: int,
    traj_success_pred: float,
    first_frame: float,
    traj_T: int,
    K: int,
) -> float:
    """1.0 if GT success, predicted success, and first positive frame is before the last-``K``-frame suffix."""
    if int(success_label) != 1 or traj_success_pred != traj_success_pred or float(traj_success_pred) < 0.5:
        return float("nan")
    if first_frame != first_frame or traj_T <= 0:
        return float("nan")
    return 1.0 if float(first_frame) < float(traj_T - K) else 0.0


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_index_by_demo(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["dataset_name"], r["demo_key"]): r for r in rows}


def load_additional_ground_truth(
    manifest_row: dict[str, str],
    prediction_npz: Any | None = None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    if prediction_npz is not None:
        files = getattr(prediction_npz, "files", ())
        if "gt_linear_time" in files:
            out["linear_time"] = np.asarray(prediction_npz["gt_linear_time"], dtype=np.float64).ravel()
        if "gt_cumulative_normalized" in files:
            out["cumulative_normalized"] = np.asarray(
                prediction_npz["gt_cumulative_normalized"], dtype=np.float64
            ).ravel()

    for canonical, col in (
        ("linear_time", "gt_linear_time_npy"),
        ("cumulative_normalized", "gt_cumulative_normalized_npy"),
    ):
        if canonical in out:
            continue
        raw = (manifest_row.get(col) or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_file():
            logger.debug("Additional GT path missing or not a file: %s (%s)", p, col)
            continue
        out[canonical] = np.load(p).astype(np.float64).ravel()

    return out


def load_simulator_rewards_from_frames(manifest_row: dict[str, str]) -> np.ndarray | None:
    """Raw per-step shaped/simulator rewards from preprocess ``frames.npz`` (``rewards`` key).

    On-disk / manifest data are unchanged. For metrics, use :func:`simulator_rewards_for_metrics`.
    """
    raw = (manifest_row.get("frames_npz") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_file():
        logger.debug("frames_npz missing or not a file: %s", p)
        return None
    z = np.load(p)
    if "rewards" not in getattr(z, "files", ()):
        return None
    return np.asarray(z["rewards"], dtype=np.float64).ravel()


def _last_mean(a: np.ndarray | None) -> tuple[float, float]:
    if a is None:
        return (float("nan"), float("nan"))
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(x[-1]), float(np.mean(x)))


def _last_mean_sum(a: np.ndarray | None) -> tuple[float, float, float]:
    if a is None:
        return (float("nan"), float("nan"), float("nan"))
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(x[-1]), float(np.mean(x)), float(np.sum(x)))


def minmax_normalize_per_traj(a: np.ndarray) -> np.ndarray:
    """Map trajectory values to [0, 1] using (x - min) / (max - min); zeros if constant."""
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    r = hi - lo
    if r < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / r


def simulator_rewards_for_metrics(manifest_row: dict[str, str]) -> np.ndarray | None:
    """Same source as :func:`load_simulator_rewards_from_frames`, min-max scaled to [0, 1] per trajectory."""
    raw = load_simulator_rewards_from_frames(manifest_row)
    if raw is None:
        return None
    return minmax_normalize_per_traj(raw)


def normalized_frame_deltas(a: np.ndarray) -> np.ndarray:
    """Per-step (x[t]-x[t-1]) / (max(x)-min(x)); empty if T<2 or zero range."""
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size < 2:
        return np.array([], dtype=np.float64)
    r = float(np.ptp(x))
    if r < 1e-12:
        return np.array([], dtype=np.float64)
    return np.diff(x) / r


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_fields_for_demo(
    pred: np.ndarray,
    additional_gt: dict[str, Any],
    simulator_rewards: np.ndarray | None,
    *,
    split_tag: str,
    eoe_flat_declared: bool,
    effective_flat: bool | None = None,
) -> dict[str, float]:
    """Pearson, Spearman, and Kendall for the same (pred, GT) pairings in one pass (shared normalization).

    If ``effective_flat`` is provided (from :func:`effective_eoe_flat`), skips a redundant ``ptp`` on ``pred``.
    """
    out = {k: float("nan") for k in ALL_CORR_KEYS}
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if effective_flat is None:
        flat = eoe_flat_declared or pred_is_eoe_flat(pred)
    else:
        flat = effective_flat
    if flat:
        return out
    do_linear = include_linear_gt_pearson_for_split(split_tag)
    if do_linear:
        lt_raw = additional_gt.get("linear_time") if additional_gt else None
        if lt_raw is not None:
            lt = np.asarray(lt_raw, dtype=np.float64).ravel()
            n_lt = min(pred.size, lt.size)
            if n_lt >= 2:
                pred_l = pred[:n_lt]
                lt = lt[:n_lt]
                pred_nm = minmax_normalize_per_traj(pred_l)
                out["pearson_linear_time_pred_raw"] = pearson_r(pred_nm, lt)
                out["spearman_linear_time_pred_raw"] = compute_spearman(pred_nm, lt)
                out["kendall_linear_time_pred_raw"] = kendall_tau_scipy(pred_nm, lt)

                dlt = normalized_frame_deltas(lt)
                dpl = normalized_frame_deltas(pred_l)
                if dlt.size == dpl.size and dlt.size > 0:
                    out["pearson_linear_time_pred_delta"] = pearson_r(dpl, dlt)
                    out["spearman_linear_time_pred_delta"] = compute_spearman(dpl, dlt)
                    out["kendall_linear_time_pred_delta"] = kendall_tau_scipy(dpl, dlt)

    if simulator_rewards is None:
        return out
    sim = np.asarray(simulator_rewards, dtype=np.float64).ravel()
    n_sim = min(pred.size, sim.size)
    if n_sim < 2:
        return out
    pred_s = pred[:n_sim]
    sim_s = sim[:n_sim]
    pred_nm_s = minmax_normalize_per_traj(pred_s)
    out["pearson_sim_reward_pred_raw"] = pearson_r(pred_nm_s, sim_s)
    out["spearman_sim_reward_pred_raw"] = compute_spearman(pred_nm_s, sim_s)
    out["kendall_sim_reward_pred_raw"] = kendall_tau_scipy(pred_nm_s, sim_s)

    dsim = normalized_frame_deltas(sim_s)
    dps = normalized_frame_deltas(pred_s)
    if dsim.size == dps.size and dsim.size > 0:
        out["pearson_sim_reward_pred_delta"] = pearson_r(dps, dsim)
        out["spearman_sim_reward_pred_delta"] = compute_spearman(dps, dsim)
        out["kendall_sim_reward_pred_delta"] = kendall_tau_scipy(dps, dsim)

    return out


def pearson_fields_for_demo(
    pred: np.ndarray,
    additional_gt: dict[str, Any],
    simulator_rewards: np.ndarray | None,
    *,
    split_tag: str,
    eoe_flat_declared: bool,
    effective_flat: bool | None = None,
) -> dict[str, float]:
    """Dense (non-flat) preds: Pearson vs gt_linear_time and vs simulator rewards."""
    c = correlation_fields_for_demo(
        pred,
        additional_gt,
        simulator_rewards,
        split_tag=split_tag,
        eoe_flat_declared=eoe_flat_declared,
        effective_flat=effective_flat,
    )
    return {k: c[k] for k in PEARSON_TRAJ_KEYS}


def spearman_kendall_fields_for_demo(
    pred: np.ndarray,
    additional_gt: dict[str, Any],
    simulator_rewards: np.ndarray | None,
    *,
    split_tag: str,
    eoe_flat_declared: bool,
    effective_flat: bool | None = None,
) -> dict[str, float]:
    """Spearman ρ and Kendall τ for the same (pred, GT) pairings as :func:`pearson_fields_for_demo`."""
    c = correlation_fields_for_demo(
        pred,
        additional_gt,
        simulator_rewards,
        split_tag=split_tag,
        eoe_flat_declared=eoe_flat_declared,
        effective_flat=effective_flat,
    )
    return {k: c[k] for k in RANK_TRAJ_KEYS}


def _mean_finite_tuple(vals: list[float]) -> tuple[float, int]:
    a = np.asarray(vals, dtype=np.float64)
    m = np.isfinite(a)
    n = int(m.sum())
    return (float(np.mean(a[m])) if n else float("nan"), n)


def _max_finite_tuple(vals: list[float]) -> tuple[float, int]:
    a = np.asarray(vals, dtype=np.float64)
    m = np.isfinite(a)
    n = int(m.sum())
    return (float(np.max(a[m])) if n else float("nan"), n)


def _min_finite_tuple(vals: list[float]) -> tuple[float, int]:
    a = np.asarray(vals, dtype=np.float64)
    m = np.isfinite(a)
    n = int(m.sum())
    return (float(np.min(a[m])) if n else float("nan"), n)


def aggregate_metric_means_for_keys(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, tuple[float, int]]:
    """One pass over ``rows``: mean per key ignoring NaNs."""
    if not rows:
        return {k: (float("nan"), 0) for k in keys}
    cols: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            cols[k].append(float(row.get(k, float("nan"))))
    return {k: _mean_finite_tuple(vs) for k, vs in cols.items()}


def aggregate_pearson_means(
    rows: Iterable[dict[str, Any]],
    *,
    keys: tuple[str, ...] = PEARSON_TRAJ_KEYS,
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, tuple[float, int]]:
    """Mean Pearson over trajectories with finite values; count excludes NaNs."""
    rows_list = [r for r in rows if row_filter is None or row_filter(r)]
    return aggregate_metric_means_for_keys(rows_list, keys)


def aggregate_hybrid_pearson_overall(rows: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Overall summary: linear-time Pearson over expert + non-expert successes only; sim Pearson over all."""
    out: dict[str, tuple[float, int]] = {}
    lin_lists: dict[str, list[float]] = {k: [] for k in PEARSON_LINEAR_KEYS}
    sim_lists: dict[str, list[float]] = {k: [] for k in PEARSON_SIM_KEYS}
    for r in rows:
        st = str(r.get("split_tag", "")).strip()
        for k in PEARSON_SIM_KEYS:
            sim_lists[k].append(float(r.get(k, float("nan"))))
        if include_linear_gt_pearson_for_split(st):
            for k in PEARSON_LINEAR_KEYS:
                lin_lists[k].append(float(r.get(k, float("nan"))))
    for k in PEARSON_LINEAR_KEYS:
        out[k] = _mean_finite_tuple(lin_lists[k])
    for k in PEARSON_SIM_KEYS:
        out[k] = _mean_finite_tuple(sim_lists[k])
    return out


def pearson_summary_aggregates(b: RowBuckets) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """(subset, aggregate_dict, notes_suffix) for summary_metrics.csv.

    * ``pearson``: hybrid overall (linear: expert + rollout_success; sim: all).
    * ``pearson_expert_ph`` / ``pearson_rollout_success`` / ``pearson_rollout_failure``: per-split_tag.
    Failure subgroup only reports simulator-reward Pearson (linear keys omitted from output).
    """
    sections: list[tuple[str, dict[str, tuple[float, int]], str]] = []

    sections.append(
        (
            "pearson",
            aggregate_hybrid_pearson_overall(b.all_rows),
            "linear=expert_ph+rollout_success; sim=all",
        )
    )

    sections.append(
        (
            "pearson_expert_ph",
            aggregate_metric_means_for_keys(b.expert_ph, PEARSON_TRAJ_KEYS),
            "",
        )
    )
    sections.append(
        (
            "pearson_rollout_success",
            aggregate_metric_means_for_keys(b.rollout_success, PEARSON_TRAJ_KEYS),
            "",
        )
    )
    sections.append(
        (
            "pearson_rollout_failure",
            aggregate_metric_means_for_keys(b.rollout_failure, PEARSON_SIM_KEYS),
            "linear_gt_time Pearson omitted for failed rollouts; sim only",
        )
    )
    return sections


def success_summary_aggregates(b: RowBuckets) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """Mean per-trajectory ``success_pred_abs_match`` where ``traj_success_pred`` is finite."""
    keys = ("success_pred_abs_match",)
    return [
        (
            "success_pred_accuracy",
            aggregate_metric_means_for_keys(b.all_rows, keys),
            "binary traj_success_pred vs manifest success_label; excludes trajectories without inference success",
        )
    ]


def success_detection_subset_sections(
    b: RowBuckets,
) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """Confusion counts/rates, mean lead (GT frame minus first predicted success frame), early-stop rates.

    Splits mirror other summaries: all, expert_ph, non-expert rollouts, rollout_success, rollout_failure.
    Confusion uses trajectories with finite ``traj_success_pred`` only.
    """
    sections: list[tuple[str, dict[str, tuple[float, int]], str]] = []

    subset_lists: list[tuple[str, list[dict[str, Any]]]] = [
        ("all", b.all_rows),
        ("expert_ph", b.expert_ph),
        ("nonexpert_rollouts", b.nonexpert),
        ("rollout_success", b.rollout_success),
        ("rollout_failure", b.rollout_failure),
    ]

    for subset_name, sub in subset_lists:
        agg: dict[str, tuple[float, int]] = {}
        tp = tn = fp = fn = 0
        for r in sub:
            tsp = r.get("traj_success_pred")
            if tsp != tsp:
                continue
            pb = 1 if float(tsp) >= 0.5 else 0
            gt = int(r["success_label"])
            if gt == 1 and pb == 1:
                tp += 1
            elif gt == 1 and pb == 0:
                fn += 1
            elif gt == 0 and pb == 1:
                fp += 1
            else:
                tn += 1
        n_clf = tp + tn + fp + fn
        agg["success_count_tp"] = (float(tp), tp)
        agg["success_count_tn"] = (float(tn), tn)
        agg["success_count_fp"] = (float(fp), fp)
        agg["success_count_fn"] = (float(fn), fn)
        agg["success_count_classified"] = (float(n_clf), n_clf)
        if n_clf > 0:
            agg["success_rate_accuracy"] = (float(tp + tn) / float(n_clf), n_clf)
        pos_gt = tp + fn
        if pos_gt > 0:
            agg["success_rate_tpr_recall"] = (float(tp) / float(pos_gt), pos_gt)
        neg_gt = fp + tn
        if neg_gt > 0:
            agg["success_rate_tnr_specificity"] = (float(tn) / float(neg_gt), neg_gt)
        pred_pos = tp + fp
        if pred_pos > 0:
            agg["success_rate_precision"] = (float(tp) / float(pred_pos), pred_pos)

        leads = [
            float(r["success_time_lead_gt_minus_first"])
            for r in sub
            if int(r["success_label"]) == 1
            and r.get("success_time_lead_gt_minus_first") == r.get("success_time_lead_gt_minus_first")
        ]
        agg["success_mean_lead_frames_gt_minus_first"] = _mean_finite_tuple(leads)

        leads_t = [
            float(r["success_time_lead_gt_minus_timing"])
            for r in sub
            if int(r["success_label"]) == 1
            and r.get("success_time_lead_gt_minus_timing") == r.get("success_time_lead_gt_minus_timing")
        ]
        agg["success_mean_lead_frames_gt_minus_timing"] = _mean_finite_tuple(leads_t)

        for K in EARLY_LAST_FRAME_KS:
            keyf = f"success_early_vs_last_{K}"
            ev = [float(r[keyf]) for r in sub if r.get(keyf) == r.get(keyf)]
            agg[f"success_mean_{keyf}"] = _mean_finite_tuple(ev)

        sections.append(
            (
                f"success_det_{subset_name}",
                agg,
                "confusion: finite traj_success_pred; leads: GT success + finite frame cols; early: GT success + pred success",
            )
        )
    return sections


def voc_summary_aggregates(b: RowBuckets) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """Subgroup VOC means: mean of per-traj ``voc`` (Spearman time index vs pred)."""

    def voc_tuple(rows: list[dict[str, Any]]) -> tuple[float, int]:
        vals = [float(r.get("voc", float("nan"))) for r in rows]
        return _mean_finite_tuple(vals)

    sections: list[tuple[str, dict[str, tuple[float, int]], str]] = [
        (
            "voc",
            {"voc": voc_tuple(b.all_rows)},
            "Spearman(chronological_index, pred); all trajectories",
        ),
        ("voc_expert_ph", {"voc": voc_tuple(b.expert_ph)}, ""),
        ("voc_rollout_success", {"voc": voc_tuple(b.rollout_success)}, ""),
        ("voc_rollout_failure", {"voc": voc_tuple(b.rollout_failure)}, ""),
    ]
    return sections


def aggregate_hybrid_rank_overall(rows: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Overall: linear rank metrics over expert + rollout successes; sim rank metrics over all trajectories."""
    out: dict[str, tuple[float, int]] = {}
    lin_lists: dict[str, list[float]] = {k: [] for k in RANK_LINEAR_KEYS}
    sim_lists: dict[str, list[float]] = {k: [] for k in RANK_SIM_KEYS}
    for r in rows:
        st = str(r.get("split_tag", "")).strip()
        for k in RANK_SIM_KEYS:
            sim_lists[k].append(float(r.get(k, float("nan"))))
        if include_linear_gt_pearson_for_split(st):
            for k in RANK_LINEAR_KEYS:
                lin_lists[k].append(float(r.get(k, float("nan"))))
    for k in RANK_LINEAR_KEYS:
        out[k] = _mean_finite_tuple(lin_lists[k])
    for k in RANK_SIM_KEYS:
        out[k] = _mean_finite_tuple(sim_lists[k])
    return out


def rank_summary_aggregates(b: RowBuckets) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """Spearman/Kendall means for all traj (hybrid), expert, non-expert, rollout success, rollout failure."""
    sections: list[tuple[str, dict[str, tuple[float, int]], str]] = [
        (
            "rankcorr",
            aggregate_hybrid_rank_overall(b.all_rows),
            "linear=expert_ph+rollout_success; sim=all",
        ),
        (
            "rankcorr_expert_ph",
            aggregate_metric_means_for_keys(b.expert_ph, RANK_TRAJ_KEYS),
            "",
        ),
        (
            "rankcorr_nonexpert",
            aggregate_metric_means_for_keys(b.nonexpert, RANK_TRAJ_KEYS),
            "rollout_success + rollout_failure",
        ),
        (
            "rankcorr_rollout_success",
            aggregate_metric_means_for_keys(b.rollout_success, RANK_TRAJ_KEYS),
            "",
        ),
        (
            "rankcorr_rollout_failure",
            aggregate_metric_means_for_keys(b.rollout_failure, RANK_SIM_KEYS),
            "linear rank metrics omitted for failed rollouts; sim only",
        ),
    ]
    return sections


def pred_stats_for_rows(
    rows: list[dict[str, Any]],
    *,
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, tuple[float, int]]:
    """Aggregate per-trajectory pred_mean / pred_min / pred_max / pred_last over a subset of rows."""
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    lasts: list[float] = []
    for row in rows:
        if row_filter is not None and not row_filter(row):
            continue
        means.append(float(row.get("pred_mean", float("nan"))))
        mins.append(float(row.get("pred_min", float("nan"))))
        maxs.append(float(row.get("pred_max", float("nan"))))
        lasts.append(float(row.get("pred_last", float("nan"))))
    return {
        "mean_pred_mean": _mean_finite_tuple(means),
        "mean_pred_min": _mean_finite_tuple(mins),
        "mean_pred_max": _mean_finite_tuple(maxs),
        "max_pred_max": _max_finite_tuple(maxs),
        "min_pred_min": _min_finite_tuple(mins),
        "mean_pred_last": _mean_finite_tuple(lasts),
        "min_pred_last": _min_finite_tuple(lasts),
        "max_pred_last": _max_finite_tuple(lasts),
    }


def pred_stats_summary_aggregates(b: RowBuckets) -> list[tuple[str, dict[str, tuple[float, int]], str]]:
    """Pred distribution summaries (incl. last-frame preds) aligned with ``rank_summary_aggregates`` subset names."""
    return [
        ("rankcorr", pred_stats_for_rows(b.all_rows, row_filter=None), "all trajectories"),
        ("rankcorr_expert_ph", pred_stats_for_rows(b.expert_ph, row_filter=None), ""),
        ("rankcorr_nonexpert", pred_stats_for_rows(b.nonexpert, row_filter=None), "rollout_success + rollout_failure"),
        ("rankcorr_rollout_success", pred_stats_for_rows(b.rollout_success, row_filter=None), ""),
        ("rankcorr_rollout_failure", pred_stats_for_rows(b.rollout_failure, row_filter=None), ""),
    ]


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


def effective_eoe_flat(pred: np.ndarray, *, cli_eoe_flat: bool) -> bool:
    if cli_eoe_flat:
        return True
    return pred_is_eoe_flat(pred)


def voc_chronological(pred: np.ndarray, *, effective_flat: bool) -> float:
    """Spearman ρ between time indices and predictions (standard VOC; no GT)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if effective_flat or pred.size < 2:
        return float("nan")
    t = np.arange(pred.size, dtype=np.float64)
    return compute_spearman(t, pred)


def traj_pred_scalar(
    pred: np.ndarray, how: str, *, effective_flat: bool | None = None, cli_eoe_flat: bool = False
) -> float:
    eff = effective_flat if effective_flat is not None else effective_eoe_flat(pred, cli_eoe_flat=cli_eoe_flat)
    if eff:
        return float(np.mean(np.asarray(pred, dtype=np.float64).ravel()))
    return traj_score(pred, how)


def traj_gt_scalar(
    gt: np.ndarray,
    pred: np.ndarray,
    how: str,
    *,
    effective_flat: bool | None = None,
    cli_eoe_flat: bool = False,
) -> float:
    eff = effective_flat if effective_flat is not None else effective_eoe_flat(pred, cli_eoe_flat=cli_eoe_flat)
    if eff:
        g = np.asarray(gt, dtype=np.float64).ravel()
        return float(np.mean(g)) if g.size else float("nan")
    return traj_score(gt, how)


def infer_eval_scope(backend_label: str) -> str:
    t = backend_label.lower()
    if "success_detector" in t:
        return "success_only"
    return "progress"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export per-trajectory scores from predictions + manifest.")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--predictions-dir", type=str, required=True, help="Directory with dataset/demo_key.npz")
    p.add_argument("--backend-label", type=str, default="", help="Label for CSV (default: folder name)")
    p.add_argument("--checkpoint-id", type=str, default="", help="HF id or path for reporting")
    p.add_argument(
        "--traj-score",
        choices=("sum", "mean"),
        default="sum",
        help="Trajectory scalar for per_trajectory_scores.csv when preds are not EoE-flat",
    )
    p.add_argument("--out-dir", type=str, default="", help="Default: <predictions-dir>/metrics")
    p.add_argument(
        "--eval-scope",
        choices=("auto", "progress", "success_only"),
        default="auto",
        help="auto: infer from --backend-label (success_detector -> success_only); recorded in summary only",
    )
    p.add_argument(
        "--eoe-flat",
        action="store_true",
        help=(
            "Declare all predictions are end-of-episode flat (one score broadcast per trajectory). "
            "Trajectory scalars use mean(pred) and mean(gt); --traj-score ignored for those. "
            "pred_eoe_flat column set to 1. Warns if any trajectory is not numerically flat."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_eval_scope(requested: str, backend_label: str) -> str:
    if requested == "auto":
        return infer_eval_scope(backend_label)
    return requested


def metrics_output_dir(pred_root: Path, out_dir_arg: str) -> Path:
    return Path(out_dir_arg).expanduser() if out_dir_arg else pred_root / "metrics"


def collect_demos_by_split(
    pred_root: Path,
    demo_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for npz_path in sorted(pred_root.rglob("*.npz")):
        if npz_path.name.endswith(".meta.json"):
            continue
        rel = npz_path.relative_to(pred_root)
        if len(rel.parts) < 2:
            continue
        dataset_name, stem = rel.parts[0], npz_path.stem
        key = (dataset_name, stem)
        if key not in demo_rows:
            logger.warning("No manifest row for %s", key)
            continue
        row = demo_rows[key]
        z = np.load(npz_path)
        add_gt = load_additional_ground_truth(row, z)
        sl = _parse_success_label(row.get("success_label", "0"))
        tsp = _npz_optional_float(z, "traj_success_pred")
        pred_a = np.asarray(z["pred"], dtype=np.float64)
        traj_T = int(pred_a.size)
        sp_first = _npz_optional_float(z, "success_pred_first_frame")
        sp_timing = _npz_optional_float(z, "success_pred_timing_frame")
        gt_sf = _gt_success_frame_idx(traj_T, sl)
        lead_first = (
            float(gt_sf - sp_first) if (gt_sf == gt_sf and sp_first == sp_first) else float("nan")
        )
        lead_timing = (
            float(gt_sf - sp_timing) if (gt_sf == gt_sf and sp_timing == sp_timing) else float("nan")
        )
        by_split[row["split_tag"]].append(
            {
                "dataset_name": dataset_name,
                "demo_key": stem,
                "pred": pred_a,
                "gt": np.asarray(z["gt"], dtype=np.float64),
                "additional_gt": add_gt,
                "simulator_rewards": simulator_rewards_for_metrics(row),
                "success_label": sl,
                "traj_success_pred": tsp,
                "traj_length_T": traj_T,
                "gt_success_frame_idx": gt_sf,
                "success_pred_first_frame": sp_first,
                "success_pred_timing_frame": sp_timing,
                "success_time_lead_gt_minus_first": lead_first,
                "success_time_lead_gt_minus_timing": lead_timing,
                "dense_target_kind": (row.get("gt_definition") or "").strip(),
                "hdf5_done_mode": row.get("hdf5_done_mode", ""),
                "hdf5_task_success_attr": row.get("hdf5_task_success_attr", ""),
            }
        )
    return by_split


def count_non_flat_if_declared_eoe_flat(
    by_split: dict[str, list[dict[str, Any]]],
    eoe_flat_declared: bool,
) -> int:
    if not eoe_flat_declared:
        return 0
    return sum(
        1
        for demos in by_split.values()
        for d in demos
        if not pred_is_eoe_flat(d["pred"])
    )


def summary_rows_for_run(
    backend: str,
    checkpoint_id: str,
    eval_scope: str,
    eoe_flat_declared: bool,
    pearson_sections: list[tuple[str, dict[str, tuple[float, int]], str]] | None = None,
    pred_stat_sections: list[tuple[str, dict[str, tuple[float, int]], str]] | None = None,
) -> list[dict[str, Any]]:
    base = {
        "backend": backend,
        "checkpoint_id": checkpoint_id,
        "dataset": "meta",
        "pred_calibration": "none",
    }
    rows: list[dict[str, Any]] = [
        {
            **base,
            "subset": "eval_scope",
            "metric_name": "eval_scope",
            "value": float("nan"),
            "notes": eval_scope,
        }
    ]
    if eoe_flat_declared:
        rows.append(
            {
                **base,
                "subset": "input_mode",
                "metric_name": "eoe_flat_declared",
                "value": 1.0,
                "notes": "set via --eoe-flat; per-traj scalars use mean(pred)/mean(gt)",
            }
        )
    if pearson_sections:
        for subset, pearson_agg, section_notes in pearson_sections:
            for key, (mean_v, n_v) in pearson_agg.items():
                note_parts = [f"n_trajectories_finite={n_v}"]
                if section_notes:
                    note_parts.append(section_notes)
                rows.append(
                    {
                        **base,
                        "subset": subset,
                        "metric_name": f"mean_{key}",
                        "value": mean_v,
                        "notes": "; ".join(note_parts),
                    }
                )
    if pred_stat_sections:
        for subset, pred_agg, section_notes in pred_stat_sections:
            for key, (val_v, n_v) in pred_agg.items():
                note_parts = [f"n_trajectories_finite={n_v}"]
                if section_notes:
                    note_parts.append(section_notes)
                rows.append(
                    {
                        **base,
                        "subset": subset,
                        "metric_name": key,
                        "value": val_v,
                        "notes": "; ".join(note_parts),
                    }
                )
    return rows


def iter_per_trajectory_rows(
    by_split: dict[str, list[dict[str, Any]]],
    *,
    backend: str,
    traj_score_how: str,
    eoe_flat_declared: bool,
) -> Iterator[dict[str, Any]]:
    for split_tag, demos in by_split.items():
        for d in demos:
            pred = np.asarray(d["pred"], dtype=np.float64)
            pv = pred.ravel()
            ag = d.get("additional_gt") or {}
            lt_last, lt_mean = _last_mean(ag.get("linear_time"))
            cn_last, cn_mean = _last_mean(ag.get("cumulative_normalized"))
            sr_last, sr_mean, sr_sum = _last_mean_sum(d.get("simulator_rewards"))
            eff_flat = effective_eoe_flat(pred, cli_eoe_flat=eoe_flat_declared)
            corr = correlation_fields_for_demo(
                d["pred"],
                d.get("additional_gt") or {},
                d.get("simulator_rewards"),
                split_tag=split_tag,
                eoe_flat_declared=eoe_flat_declared,
                effective_flat=eff_flat,
            )
            pf = {k: corr[k] for k in PEARSON_TRAJ_KEYS}
            rk = {k: corr[k] for k in RANK_TRAJ_KEYS}
            trsp = float(d.get("traj_success_pred", float("nan")))
            sl_i = int(d["success_label"])
            traj_T = int(d.get("traj_length_T", pred.size))
            sf = float(d.get("success_pred_first_frame", float("nan")))
            stim = float(d.get("success_pred_timing_frame", float("nan")))
            yield {
                "backend": backend,
                "dataset_name": d["dataset_name"],
                "demo_key": d["demo_key"],
                "split_tag": split_tag,
                "dense_target_kind": d.get("dense_target_kind", ""),
                "pred_eoe_flat": int(eff_flat),
                "pred_last": float(pv[-1]) if pv.size else float("nan"),
                "pred_mean": float(np.mean(pv)) if pv.size else float("nan"),
                "pred_min": float(np.min(pv)) if pv.size else float("nan"),
                "pred_max": float(np.max(pv)) if pv.size else float("nan"),
                "gt_linear_time_last": lt_last,
                "gt_linear_time_mean": lt_mean,
                "gt_cumulative_normalized_last": cn_last,
                "gt_cumulative_normalized_mean": cn_mean,
                "simulator_reward_last": sr_last,
                "simulator_reward_mean": sr_mean,
                "simulator_reward_sum": sr_sum,
                "traj_score_pred": traj_pred_scalar(pred, traj_score_how, effective_flat=eff_flat),
                "traj_score_gt": traj_gt_scalar(d["gt"], pred, traj_score_how, effective_flat=eff_flat),
                **pf,
                **rk,
                "voc": voc_chronological(pred, effective_flat=eff_flat),
                "success_label": d["success_label"],
                "traj_success_pred": trsp,
                "success_pred_abs_match": _success_pred_match(trsp, sl_i),
                "traj_length_T": traj_T,
                "gt_success_frame_idx": float(d.get("gt_success_frame_idx", float("nan"))),
                "success_pred_first_frame": sf,
                "success_pred_timing_frame": stim,
                "success_time_lead_gt_minus_first": float(d.get("success_time_lead_gt_minus_first", float("nan"))),
                "success_time_lead_gt_minus_timing": float(d.get("success_time_lead_gt_minus_timing", float("nan"))),
                "success_cm": _success_confusion_bucket(trsp, sl_i),
                "success_early_vs_last_1": _early_vs_last_k(sl_i, trsp, sf, traj_T, 1),
                "success_early_vs_last_5": _early_vs_last_k(sl_i, trsp, sf, traj_T, 5),
                "success_early_vs_last_10": _early_vs_last_k(sl_i, trsp, sf, traj_T, 10),
                "hdf5_done_mode": d.get("hdf5_done_mode", ""),
                "hdf5_task_success_attr": d.get("hdf5_task_success_attr", ""),
            }


def write_dict_rows_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest_path = Path(args.manifest).expanduser()
    pred_root = Path(args.predictions_dir).expanduser()
    backend = args.backend_label or pred_root.name
    out_dir = metrics_output_dir(pred_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_scope = resolve_eval_scope(args.eval_scope, backend)
    logger.info("eval_scope=%s (backend_label=%s)", eval_scope, backend)
    if args.eoe_flat:
        logger.info("eoe_flat mode: trajectory scalars use per-traj mean(pred)/mean(gt); --traj-score ignored")

    demo_rows = manifest_index_by_demo(load_manifest(manifest_path))
    by_split = collect_demos_by_split(pred_root, demo_rows)

    n_mismatch = count_non_flat_if_declared_eoe_flat(by_split, args.eoe_flat)
    if n_mismatch:
        logger.warning(
            "--eoe-flat: %d trajectories have ptp(pred) >= %g (not numerically constant); "
            "still using flat semantics for scalars",
            n_mismatch,
            FLAT_PRED_PTP_EPS,
        )

    per_traj_rows = list(
        iter_per_trajectory_rows(
            by_split,
            backend=backend,
            traj_score_how=args.traj_score,
            eoe_flat_declared=args.eoe_flat,
        )
    )
    buckets = build_row_buckets(per_traj_rows)
    pearson_sections = pearson_summary_aggregates(buckets)
    voc_sections = voc_summary_aggregates(buckets)
    rank_sections = rank_summary_aggregates(buckets)
    success_sections = success_summary_aggregates(buckets)
    pred_stat_sections = pred_stats_summary_aggregates(buckets) + success_detection_subset_sections(buckets)

    summary_path = out_dir / "summary_metrics.csv"
    write_dict_rows_csv(
        summary_path,
        SUMMARY_CSV_FIELDS,
        summary_rows_for_run(
            backend,
            args.checkpoint_id,
            eval_scope,
            args.eoe_flat,
            pearson_sections=pearson_sections + voc_sections + rank_sections + success_sections,
            pred_stat_sections=pred_stat_sections,
        ),
    )

    per_traj_path = out_dir / "per_trajectory_scores.csv"
    write_dict_rows_csv(per_traj_path, PER_TRAJECTORY_CSV_FIELDS, per_traj_rows)

    logger.info("Wrote %s and %s", summary_path, per_traj_path)


if __name__ == "__main__":
    main()