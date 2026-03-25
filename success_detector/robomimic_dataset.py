import os
import glob
import json
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


class RobomimicLatentDataset(Dataset):
    def __init__(
        self,
        latent_root: str,
        hdf5_path: str,
        split: str = "train",
        rgb_skip: int = 1,
        preload: bool = True,
        temporal_window: int = 1,
    ):
        self.latent_root = latent_root
        self.split = split
        self.rgb_skip = rgb_skip
        self.preload = preload
        self.temporal_window = max(1, int(temporal_window))
        anno_dir = os.path.join(latent_root, "annotation", split)
        if not os.path.isdir(anno_dir):
            available = os.listdir(os.path.join(latent_root, "annotation"))
            raise FileNotFoundError(
                f"Annotation directory not found: {anno_dir}. "
                f"Available splits: {available}"
            )

        anno_files = sorted(
            glob.glob(os.path.join(anno_dir, "*.json")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        self.annotations: list[dict] = []
        for p in anno_files:
            with open(p) as fh:
                self.annotations.append(json.load(fh))

        self.num_cameras: int = len(self.annotations[0]["latent_videos"])

        self._hdf5_file: Optional[h5py.File] = None
        self._per_traj_dones: dict[int, np.ndarray] = {}
        self._per_traj_rewards: dict[int, np.ndarray] = {}  
        self._per_traj_frame_success: dict[int, np.ndarray] = {}

        self._hdf5_file = h5py.File(hdf5_path, "r")
        for anno in self.annotations:
            ep_id = anno["episode_id"]
            ep_grp = self._hdf5_file[f"data/demo_{ep_id}"]

            raw_dones = ep_grp["dones"][()]
            raw_rewards = ep_grp["rewards"][()]
            n_latent = anno["video_length"]

            sub_dones = np.zeros(n_latent, dtype=np.float32)
            sub_rewards = np.zeros(n_latent, dtype=np.float32)
            for i in range(n_latent):
                lo = i * rgb_skip
                hi = len(raw_dones) if i == n_latent - 1 else min(lo + rgb_skip, len(raw_dones))
                sub_dones[i] = float(np.any(raw_dones[lo:hi] > 0))
                sub_rewards[i] = float(np.sum(raw_rewards[lo:hi]))

            self._per_traj_dones[ep_id] = sub_dones
            self._per_traj_rewards[ep_id] = sub_rewards


            cumulative_reward = np.cumsum(sub_rewards)
            self._per_traj_frame_success[ep_id] = (cumulative_reward > 0).astype(np.float32)


            hdf5_success = bool(np.any(raw_rewards > 0))
            if anno["success"] == 0 and hdf5_success:
                anno["success"] = 1

        self._latents: dict[int, list[torch.Tensor]] = {}
        if preload:
            for traj_idx, anno in enumerate(self.annotations):
                self._latents[traj_idx] = [
                    torch.load(
                        os.path.join(latent_root, ci["latent_video_path"]),
                        map_location="cpu",
                        weights_only=True,
                    )
                    for ci in anno["latent_videos"]
                ]

        self._index: list[tuple[int, int]] = []
        for traj_idx, anno in enumerate(self.annotations):
            for frame_idx in range(anno["video_length"]):
                self._index.append((traj_idx, frame_idx))


    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        traj_idx, frame_idx = self._index[idx]
        anno = self.annotations[traj_idx]
        ep_id = anno["episode_id"]

        cam_latents = self._load_latents(traj_idx)
        tw = self.temporal_window
        T = anno["video_length"]
        start = frame_idx - tw + 1
        latent_views = []
        for c in range(self.num_cameras):
            chunks = []
            for t in range(start, frame_idx + 1):
                tt = 0 if t < 0 else t
                chunks.append(cam_latents[c][tt])
            stacked = torch.cat(chunks, dim=0)  # [C*tw, H, W]
            latent_views.append(stacked)

        if ep_id in self._per_traj_frame_success:
            success_label = float(self._per_traj_frame_success[ep_id][frame_idx])
        else:
            success_label = float(anno["success"])

        sample = {
            "latent_views": latent_views,
            "success": torch.tensor(success_label, dtype=torch.float32),
            "traj_success": torch.tensor(float(anno["success"]), dtype=torch.float32),
            "timestep": frame_idx,
            "traj_id": ep_id,
            "num_frames": anno["video_length"],
            "distance_to_goal": torch.tensor(
                float(T - frame_idx - 1), dtype=torch.float32
            ),
        }

        if ep_id in self._per_traj_dones:
            sample["done"] = torch.tensor(
                self._per_traj_dones[ep_id][frame_idx], dtype=torch.float32
            )
            sample["reward"] = torch.tensor(
                self._per_traj_rewards[ep_id][frame_idx], dtype=torch.float32
            )

        return sample

    # ---------------------------------------------------------------------- #
    #  Trajectory-level access
    # ---------------------------------------------------------------------- #

    def frame_label(self, traj_idx: int, frame_idx: int) -> float:
        """Return the per-frame success label for a given frame."""
        ep_id = self.annotations[traj_idx]["episode_id"]
        if ep_id in self._per_traj_frame_success:
            return float(self._per_traj_frame_success[ep_id][frame_idx])
        return float(self.annotations[traj_idx]["success"])

    def get_trajectory(self, traj_idx: int) -> dict:
        """Return the full trajectory (all frames, all cameras) at *traj_idx*."""
        anno = self.annotations[traj_idx]
        ep_id = anno["episode_id"]
        cam_latents = self._load_latents(traj_idx)

        traj: dict = {
            "latent_views": cam_latents,
            "traj_success": anno["success"],
            "traj_id": ep_id,
            "num_frames": anno["video_length"],
        }
        if anno.get("states"):
            traj["states"] = torch.tensor(anno["states"], dtype=torch.float32)
        if ep_id in self._per_traj_dones:
            traj["dones"] = torch.from_numpy(self._per_traj_dones[ep_id])
            traj["rewards"] = torch.from_numpy(self._per_traj_rewards[ep_id])
        if ep_id in self._per_traj_frame_success:
            traj["frame_success"] = torch.from_numpy(self._per_traj_frame_success[ep_id])
        return traj

    @property
    def num_trajectories(self) -> int:
        return len(self.annotations)

    # ---------------------------------------------------------------------- #
    #  Internals
    # ---------------------------------------------------------------------- #

    def _load_latents(self, traj_idx: int) -> list[torch.Tensor]:
        if traj_idx in self._latents:
            return self._latents[traj_idx]
        anno = self.annotations[traj_idx]
        return [
            torch.load(
                os.path.join(self.latent_root, ci["latent_video_path"]),
                map_location="cpu",
                weights_only=True,
            )
            for ci in anno["latent_videos"]
        ]

    def close(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass
            self._hdf5_file = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"trajectories={self.num_trajectories}, "
            f"frames={len(self)}, "
            f"cameras={self.num_cameras}, "
            f"temporal_window={self.temporal_window})"
        )


def build_combined_dataset(
    latent_roots: list[str],
    hdf5_paths: list[str],
    split: str = "train",
    rgb_skip: int = 1,
    preload: bool = True,
) -> ConcatDataset:
    """Convenience helper to merge multiple latent datasets (e.g. downloaded
    demonstrations + policy rollouts) into a single ``ConcatDataset``."""
    datasets = []
    for root, h5 in zip(latent_roots, hdf5_paths):
        anno_dir = os.path.join(root, "annotation", split)
        if not os.path.isdir(anno_dir):
            continue
        datasets.append(
            RobomimicLatentDataset(
                latent_root=root,
                hdf5_path=h5,
                split=split,
                rgb_skip=rgb_skip,
                preload=preload,
                temporal_window=temporal_window,
            )
        )
    return ConcatDataset(datasets)
