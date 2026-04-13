"""Pearson / Spearman / Kendall tau-a and per-trajectory normalization helpers."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, kendalltau, pearsonr, spearmanr


def compute_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = pearsonr(a, b)
        except Exception:
            corr = float("nan")
    return float(corr)


def compute_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation; scale-invariant vs Pearson — fairer across differently scaled models."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            corr, _ = spearmanr(a, b, nan_policy="omit")
        except Exception:
            corr = float("nan")
    if corr is None or (isinstance(corr, float) and np.isnan(corr)):
        return float("nan")
    return float(corr)


def kendall_tau_a(x: Sequence[float], y: Sequence[float]) -> float:
    x = list(x)
    y = list(y)
    n = len(x)
    if n < 2:
        return float("nan")
    c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])
        if dx == 0 or dy == 0:
            continue
        if dx == dy:
            c += 1
        else:
            d += 1
    denom = n * (n - 1) / 2.0
    if c + d == 0:
        return float("nan")
    return (c - d) / denom


def minmax01_per_traj(z: np.ndarray, *, constant_fill: float = 0.5) -> np.ndarray:
    """Min–max normalize a 1D trajectory to [0, 1]; constant vectors map to ``constant_fill`` everywhere."""
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return z
    lo = float(np.min(z))
    hi = float(np.max(z))
    if hi - lo < 1e-12:
        return np.full_like(z, constant_fill, dtype=np.float64)
    return (z - lo) / (hi - lo)


def pooled_pearson_normalized_frames(
    demos: list[dict],
    pred_key: str = "pred",
    gt_key: str = "gt",
    *,
    scale_pred: callable | None = None,
) -> float:
    """Pearson between per-frame min–max normalized pred and GT, pooled over trajectories.

    ``scale_pred`` optionally maps each demo's pred array (e.g. affine calibration).
    """
    ys: list[float] = []
    ps: list[float] = []
    for d in demos:
        gt = np.asarray(d[gt_key], dtype=np.float64).ravel()
        if scale_pred is not None:
            pr = np.asarray(scale_pred(d), dtype=np.float64).ravel()
        else:
            pr = np.asarray(d[pred_key], dtype=np.float64).ravel()
        n = min(gt.size, pr.size)
        if n == 0:
            continue
        gt = gt[:n]
        pr = pr[:n]
        ys.extend(minmax01_per_traj(gt).tolist())
        ps.extend(minmax01_per_traj(pr).tolist())
    if len(ys) < 2:
        return float("nan")
    return compute_pearson(ys, ps)


def mean_intra_trajectory_spearman(
    demos: list[dict],
    *,
    scale_pred: Callable | None = None,
    pred_key: str = "pred",
    gt_key: str = "gt",
    min_len: int = 4,
    flat_ptp_eps: float = 1e-5,
) -> float:
    """Mean Spearman(pred, gt) along time within each demo (scale-free temporal alignment).

    Skips trajectories shorter than ``min_len``, constant predictions (no rank variation), and
    undefined correlations.
    """
    vals: list[float] = []
    for d in demos:
        gt = np.asarray(d[gt_key], dtype=np.float64).ravel()
        if scale_pred is not None:
            pr = np.asarray(scale_pred(d), dtype=np.float64).ravel()
        else:
            pr = np.asarray(d[pred_key], dtype=np.float64).ravel()
        n = min(gt.size, pr.size)
        if n < min_len:
            continue
        gt = gt[:n]
        pr = pr[:n]
        if float(np.ptp(pr)) < flat_ptp_eps:
            continue
        c = compute_spearman(gt, pr)
        if np.isfinite(c):
            vals.append(float(c))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def mean_dense_mae(
    demos: list[dict],
    *,
    scale_pred: Callable | None = None,
    pred_key: str = "pred",
    gt_key: str = "gt",
    flat_ptp_eps: float = 1e-5,
) -> float:
    """Mean over demos of mean |pred - gt| per frame. Skips constant (EoE-flat) preds."""
    maes: list[float] = []
    for d in demos:
        gt = np.asarray(d[gt_key], dtype=np.float64).ravel()
        if scale_pred is not None:
            pr = np.asarray(scale_pred(d), dtype=np.float64).ravel()
        else:
            pr = np.asarray(d[pred_key], dtype=np.float64).ravel()
        n = min(gt.size, pr.size)
        if n == 0:
            continue
        pr = pr[:n]
        if float(np.ptp(pr)) < flat_ptp_eps:
            continue
        gt = gt[:n]
        maes.append(float(np.mean(np.abs(pr - gt))))
    if not maes:
        return float("nan")
    return float(np.mean(maes))


def kendall_tau_scipy(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall τ (SciPy); tie-aware unlike τ_a on raw concordance."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    tau, _ = kendalltau(a, b)
    if tau is None or not np.isfinite(tau):
        return float("nan")
    return float(tau)
