#!/usr/bin/env python3
"""Build manifest.csv, per-demo frames.npz, gt_dense.npy, and optional mp4."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from reward_eval import DEFAULT_SQUARE_INSTRUCTION
from reward_eval.ground_truth import dense_gt_from_rewards
from reward_eval.inspect_hdf5 import _demo_sort_key, list_demos

logger = logging.getLogger("preprocess_manifest")


def _rgb_key(camera: str) -> str:
    return f"obs/{camera}_image"


def load_demo_frames(ep_grp: h5py.Group, camera: str) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB."""
    key = _rgb_key(camera)
    if key not in ep_grp:
        raise KeyError(f"Missing {key} in episode (camera={camera})")
    x = ep_grp[key][()]
    x = np.asarray(x, dtype=np.uint8)
    if x.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C), got {x.shape}")
    if x.shape[-1] >= 3:
        return x[..., :3]
    raise ValueError(f"Unexpected channels {x.shape}")


def _read_done_mode(ep_grp: h5py.Group) -> int | None:
    if "done_mode" not in ep_grp.attrs:
        return None
    try:
        return int(ep_grp.attrs["done_mode"])
    except (TypeError, ValueError):
        return None


def demo_success_label(
    ep_grp: h5py.Group,
    *,
    dense_max_reward_threshold: float = 0.99,
) -> bool:
    """Infer trajectory success for rollout splits.

    Priority:
    1. ``task_success`` attr from ``run_trained_agent.py`` (required for dense/shaped rewards).
    2. ``dones`` + ``done_mode`` attrs (same semantics as ``dataset_states_to_obs.py``):
       for ``done_mode == 0``, any positive ``done`` means a task-success transition was recorded.
       For ``done_mode == 2``, **do not** infer from ``dones`` alone (failed timeouts also end with done=1).
    3. Sparse binary reward heuristic (legacy rollout HDF5 without attrs).
    4. Dense max-reward fallback if no attrs.
    """
    if "task_success" in ep_grp.attrs:
        v = ep_grp.attrs["task_success"]
        try:
            return bool(int(round(float(v))))
        except (TypeError, ValueError):
            pass

    dm = _read_done_mode(ep_grp)
    if dm == 0 and "dones" in ep_grp:
        d = np.asarray(ep_grp["dones"][()]).ravel()
        if d.size and np.any(d > 0):
            return True

    rw = np.asarray(ep_grp["rewards"][()], dtype=np.float64).ravel()
    if rw.size == 0:
        return False

    uniq = np.unique(rw)
    # Sparse binary-style rewards (typical unshaped policy rollouts; legacy files without task_success)
    if len(uniq) <= 4 and float(np.max(rw)) <= 1.0 + 1e-5:
        if np.any(rw > 0):
            return True
        return False

    # Dense shaping, no task_success / done_mode-0 signal: weak fallback
    if float(np.max(rw)) >= dense_max_reward_threshold:
        return True
    return False


def write_mp4(frames_hwc: np.ndarray, out_path: Path, fps: int = 10) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        logger.warning("imageio not installed; skipping mp4 for %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames_hwc, fps=fps, codec="libx264")


def process_hdf5(
    hdf5_path: str,
    dataset_name: str,
    output_root: Path,
    *,
    camera: str,
    expert_linear_gt: bool,
    is_expert_dataset: bool,
    write_videos: bool,
    dense_success_max_reward: float = 0.99,
) -> list[dict[str, str | int | float | bool]]:
    """Returns list of manifest rows (dicts)."""
    rows: list[dict[str, str | int | float | bool]] = []
    hdf5_path = str(Path(hdf5_path).expanduser())
    out_ds = output_root / dataset_name
    out_ds.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        demos = list_demos(f)
        for demo_key in demos:
            ep = f["data"][demo_key]
            demo_id = int(demo_key.split("_", 1)[1])
            rewards = np.asarray(ep["rewards"][()], dtype=np.float64).ravel()
            t = len(rewards)
            if t == 0:
                logger.warning("Skip empty %s/%s", dataset_name, demo_key)
                continue

            if is_expert_dataset:
                split_tag = "expert_ph"
                gt_def = "linear_time_expert" if expert_linear_gt else "cumulative_normalized"
                gt_vec = dense_gt_from_rewards(
                    rewards,
                    definition="linear_time_expert" if expert_linear_gt else "cumulative_normalized",
                )
            else:
                succ = demo_success_label(ep, dense_max_reward_threshold=dense_success_max_reward)
                split_tag = "rollout_success" if succ else "rollout_failure"
                gt_def = "cumulative_normalized"
                gt_vec = dense_gt_from_rewards(rewards, definition="cumulative_normalized")

            frames = load_demo_frames(ep, camera)
            if frames.shape[0] != t:
                logger.warning(
                    "T mismatch demo=%s rewards_T=%d frames_T=%d; trimming to min",
                    demo_key,
                    t,
                    frames.shape[0],
                )
                m = min(t, frames.shape[0])
                rewards = rewards[:m]
                frames = frames[:m]
                t = m
                if is_expert_dataset:
                    gt_vec = dense_gt_from_rewards(
                        rewards,
                        definition="linear_time_expert" if expert_linear_gt else "cumulative_normalized",
                    )
                else:
                    gt_vec = dense_gt_from_rewards(rewards, definition="cumulative_normalized")

            demo_dir = out_ds / demo_key
            demo_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(demo_dir / "frames.npz", rgb=frames, rewards=rewards)
            np.save(demo_dir / "gt_dense.npy", gt_vec.astype(np.float32))

            hdf5_done_mode = ""
            hdf5_task_success_attr = ""
            dm = _read_done_mode(ep)
            if dm is not None:
                hdf5_done_mode = str(dm)
            if "task_success" in ep.attrs:
                try:
                    hdf5_task_success_attr = str(int(round(float(ep.attrs["task_success"]))))
                except (TypeError, ValueError):
                    hdf5_task_success_attr = ""

            meta = {
                "dataset_name": dataset_name,
                "hdf5_path": hdf5_path,
                "demo_key": demo_key,
                "demo_id": demo_id,
                "T": t,
                "success_label": int(
                    demo_success_label(ep, dense_max_reward_threshold=dense_success_max_reward)
                    if not is_expert_dataset
                    else True
                ),
                "split_tag": split_tag,
                "gt_definition": gt_def,
                "camera": camera,
                "frames_npz": str(demo_dir / "frames.npz"),
                "gt_npy": str(demo_dir / "gt_dense.npy"),
                "task_instruction": DEFAULT_SQUARE_INSTRUCTION,
                "hdf5_done_mode": hdf5_done_mode,
                "hdf5_task_success_attr": hdf5_task_success_attr,
            }
            with open(demo_dir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            if write_videos:
                write_mp4(frames, demo_dir / "video.mp4")

            rows.append(meta)
            logger.info("Wrote %s %s T=%d split=%s", dataset_name, demo_key, t, split_tag)

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build manifest and cached frames for reward eval.")
    p.add_argument(
        "--expert-hdf5",
        type=str,
        default="/scratch/bggq/asunesara/square_image_dense_three.hdf5",
        help="200 PH expert demos",
    )
    p.add_argument(
        "--rollout-hdf5",
        type=str,
        default="/scratch/bggq/asunesara/rollout_dataset/output_dense_shaping.hdf5",
        help="300 mixed rollout demos",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/bggq/asunesara/reward_eval_cache",
        help="Cache root for frames/gt and manifest.csv",
    )
    p.add_argument("--camera", type=str, default="agentview", help="obs/{camera}_image")
    p.add_argument(
        "--expert-gt",
        choices=("cumulative", "linear"),
        default="linear",
        help="Expert PH dense GT: linear-in-time vs cumulative-normalized",
    )
    p.add_argument("--write-videos", action="store_true", help="Also write per-demo mp4 (large)")
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--skip-expert", action="store_true")
    p.add_argument(
        "--dense-success-max-reward",
        type=float,
        default=0.99,
        help="For rollout HDF5 without task_success attr and with shaped/dense rewards, "
        "treat episode as success if max(rewards) >= this (only if sparse heuristic does not apply).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_root = Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    expert_linear = args.expert_gt == "linear"
    all_rows: list[dict[str, str | int | float | bool]] = []

    if not args.skip_expert:
        rows_e = process_hdf5(
            args.expert_hdf5,
            "expert_ph",
            out_root,
            camera=args.camera,
            expert_linear_gt=expert_linear,
            is_expert_dataset=True,
            write_videos=args.write_videos,
        )
        all_rows.extend(rows_e)

    if not args.skip_rollout:
        rows_r = process_hdf5(
            args.rollout_hdf5,
            "rollout_mixed",
            out_root,
            camera=args.camera,
            expert_linear_gt=False,
            is_expert_dataset=False,
            write_videos=args.write_videos,
            dense_success_max_reward=args.dense_success_max_reward,
        )
        all_rows.extend(rows_r)

    manifest_path = out_root / "manifest.csv"
    if not all_rows:
        logger.error("No rows written — check HDF5 paths and permissions.")
        raise SystemExit(1)

    fieldnames = list(all_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    logger.info("Wrote %s (%d trajectories)", manifest_path, len(all_rows))


if __name__ == "__main__":
    main()
