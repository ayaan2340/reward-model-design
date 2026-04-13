"""Dense ground-truth reward curves from robomimic HDF5 `rewards` arrays."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np


class GTDefinition(str, Enum):
    cumulative_normalized = "cumulative_normalized"
    linear_time_expert = "linear_time_expert"


def dense_gt_from_rewards(
    rewards: np.ndarray,
    *,
    definition: Literal["cumulative_normalized", "linear_time_expert"] = "cumulative_normalized",
) -> np.ndarray:
    """Build per-frame dense target in [0, 1].

    - cumulative_normalized: r_t = (sum_{k<=t} r_k) / max(sum_k r_k, eps). If total is 0, all zeros.
    - linear_time_expert: r_t = t / (T-1) for T>1, else 0.0 (for all-success expert demos).
    """
    rewards = np.asarray(rewards, dtype=np.float64).ravel()
    t = len(rewards)
    if t == 0:
        return np.array([], dtype=np.float32)

    if definition == "linear_time_expert":
        if t == 1:
            return np.zeros(1, dtype=np.float32)
        idx = np.arange(t, dtype=np.float64)
        return (idx / float(t - 1)).astype(np.float32)

    c = np.cumsum(rewards)
    c_total = float(c[-1]) if len(c) else 0.0
    eps = 1e-8
    if c_total <= eps:
        return np.zeros(t, dtype=np.float32)
    out = (c / max(c_total, eps)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def trajectory_return(rewards: np.ndarray) -> float:
    return float(np.sum(np.asarray(rewards, dtype=np.float64)))
