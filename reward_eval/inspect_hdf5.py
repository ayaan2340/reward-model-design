#!/usr/bin/env python3
"""Read-only inspection of robomimic-style HDF5 for reward evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _demo_sort_key(name: str) -> int:
    if name.startswith("demo_"):
        try:
            return int(name[5:])
        except ValueError:
            return 0
    return 0


def list_demos(f: h5py.File) -> list[str]:
    data = f["data"]
    demos = [k for k in data.keys() if isinstance(k, str) and k.startswith("demo_")]
    return sorted(demos, key=_demo_sort_key)


def summarize_rewards(rewards: np.ndarray) -> dict[str, Any]:
    r = np.asarray(rewards, dtype=np.float64).ravel()
    if r.size == 0:
        return {"len": 0, "min": None, "max": None, "sum": None, "nonzero_frac": None}
    return {
        "len": int(r.size),
        "min": float(r.min()),
        "max": float(r.max()),
        "mean": float(r.mean()),
        "sum": float(r.sum()),
        "nonzero_frac": float(np.mean(r != 0)),
        "unique_approx": int(len(np.unique(r))),
    }


def inspect_episode(ep_grp: h5py.Group, demo_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {"demo_key": demo_key}
    if "rewards" not in ep_grp:
        out["error"] = "missing rewards"
        return out
    rw = ep_grp["rewards"][()]
    out["rewards"] = summarize_rewards(rw)
    if "dones" in ep_grp:
        d = np.asarray(ep_grp["dones"][()]).ravel()
        n_pos = int(np.sum(d > 0))
        out["dones"] = {
            "len": len(d),
            "num_positive": n_pos,
            "any_positive": bool(n_pos > 0),
        }
    # success heuristics (legacy; see done_mode note)
    succ_reward = bool(np.any(np.asarray(rw) > 0))
    succ_done = bool(np.any(np.asarray(ep_grp["dones"][()]) > 0)) if "dones" in ep_grp else None
    out["success_heuristic_reward_pos"] = succ_reward
    out["success_heuristic_done"] = succ_done
    if "task_success" in ep_grp.attrs:
        out["task_success_attr"] = bool(int(round(float(ep_grp.attrs["task_success"]))))
    if "done_mode" in ep_grp.attrs:
        try:
            dm = int(ep_grp.attrs["done_mode"])
            out["done_mode_attr"] = dm
            # For done_mode 2, any_positive dones is NOT a success indicator (terminal on failure too).
            if dm == 2 and "dones" in out:
                out["dones"]["note"] = (
                    "done_mode=2: last timestep is done=1 even on failure; use task_success_attr, not any_positive"
                )
            if dm == 0 and "dones" in out:
                out["success_from_dones_done_mode_0"] = bool(out["dones"]["any_positive"])
        except (TypeError, ValueError):
            out["done_mode_attr"] = None
    # obs keys
    obs = ep_grp.get("obs")
    if obs is not None:
        keys = list(obs.keys()) if isinstance(obs, h5py.Group) else []
        img_keys = [k for k in keys if k.endswith("_image")]
        out["obs_image_keys"] = sorted(img_keys)
        if img_keys:
            k0 = img_keys[0]
            shp = obs[k0].shape
            out["sample_image_shape"] = list(shp)
    return out


def inspect_file(path: str, *, max_demos: int | None = None) -> dict[str, Any]:
    path = str(Path(path).expanduser())
    summary: dict[str, Any] = {"hdf5_path": path, "demos": []}
    with h5py.File(path, "r") as f:
        if "data" not in f:
            summary["error"] = "no top-level data group"
            return summary
        mask_info: dict[str, Any] = {}
        if "mask" in f:
            mg = f["mask"]
            for split in ("train", "valid", "test"):
                k = f"mask/{split}"
                if k in f:
                    raw = f[k][()]
                    mask_info[split] = len(raw)
            summary["mask"] = mask_info
        demos = list_demos(f)
        if max_demos is not None:
            demos = demos[: max_demos]
        all_nonzero_frac = []
        all_unique = []
        for dk in demos:
            ep = f["data"][dk]
            row = inspect_episode(ep, dk)
            summary["demos"].append(row)
            if "rewards" in row:
                all_nonzero_frac.append(row["rewards"]["nonzero_frac"])
                all_unique.append(row["rewards"]["unique_approx"])
        if all_nonzero_frac:
            summary["aggregate"] = {
                "num_demos": len(demos),
                "reward_nonzero_frac_mean": float(np.mean(all_nonzero_frac)),
                "reward_unique_max": int(max(all_unique)) if all_unique else 0,
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect robomimic HDF5 rewards and obs for reward eval.")
    p.add_argument("hdf5_path", type=str, help="Path to dataset.hdf5")
    p.add_argument("--max-demos", type=int, default=None, help="Only first N demos (debug)")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout")
    p.add_argument(
        "--strict-dense",
        action="store_true",
        help="Exit non-zero if rewards look purely sparse (nonzero_frac mean ~0 for all demos).",
    )
    args = p.parse_args()

    data = inspect_file(args.hdf5_path, max_demos=args.max_demos)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(json.dumps(data, indent=2))

    if args.strict_dense and "aggregate" in data:
        nz = data["aggregate"].get("reward_nonzero_frac_mean", 0.0)
        # allow "dense shaping": many steps with small positive reward
        if nz < 1e-6:
            print(
                "strict-dense: rewards appear all-zero on average — check shaping.",
                file=sys.stderr,
            )
            sys.exit(2)


if __name__ == "__main__":
    main()
