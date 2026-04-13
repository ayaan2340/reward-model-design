import argparse
import json
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


CAMERA_ORDER = ["agentview", "frontview", "robot0_eye_in_hand"]


def accumulate_deltas(actions_20hz, start, skip):
    window = actions_20hz[start:start + skip]  # (skip, 7)
    pos_delta = window[:, :3].sum(axis=0)

    rot_composed = Rotation.identity()
    for j in range(window.shape[0]):
        rot_composed = Rotation.from_rotvec(window[j, 3:6]) * rot_composed
    rot_delta = rot_composed.as_rotvec()

    gripper = window[-1, 6]
    return np.concatenate([pos_delta, rot_delta, [gripper]])


def subsample_demo(src_grp, dst_grp, skip):
    orig_actions = src_grp["actions"][()]  # (orig_len, 7) -- 20Hz deltas
    orig_len = orig_actions.shape[0]

    obs_len_actual = src_grp["obs/robot0_eef_pos"].shape[0]
    obs_len = min(obs_len_actual, orig_len + 1)

    obs_indices = np.arange(0, obs_len, skip)
    n_sub = len(obs_indices) - 1
    if n_sub < 1:
        return 0

    accumulated = []
    for i in range(n_sub):
        act_start = i * skip
        act_end = min(act_start + skip, orig_len)
        actual_skip = act_end - act_start
        accumulated.append(accumulate_deltas(orig_actions, act_start, actual_skip))
    actions_5hz = np.stack(accumulated).astype(np.float32)  # (n_sub, 7)

    dst_grp.create_dataset("actions", data=actions_5hz)

    src_obs_indices = obs_indices[:n_sub]

    obs_grp = dst_grp.create_group("obs")

    for cam in CAMERA_ORDER:
        key = f"{cam}_image"
        obs_grp.create_dataset(key, data=src_grp[f"obs/{key}"][()][src_obs_indices])

    for low_dim_key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
        obs_grp.create_dataset(low_dim_key, data=src_grp[f"obs/{low_dim_key}"][()][src_obs_indices])

    orig_dones = src_grp["dones"][()]
    sub_dones = orig_dones[skip - 1::skip][:n_sub]
    if len(sub_dones) < n_sub:
        sub_dones = np.concatenate([sub_dones, np.zeros(n_sub - len(sub_dones))])
    dst_grp.create_dataset("dones", data=sub_dones)

    dst_grp.attrs["num_samples"] = n_sub
    return n_sub


def main():
    parser = argparse.ArgumentParser(
        description="Subsample robosuite HDF5 from 20Hz to 5Hz with accumulated delta actions"
    )
    parser.add_argument("--input", type=str, required=True, help="path to original 20Hz robosuite HDF5")
    parser.add_argument("--output", type=str, required=True, help="path for output 5Hz HDF5")
    parser.add_argument("--skip", type=int, default=4, help="subsample factor (4 = 20Hz->5Hz)")
    args = parser.parse_args()

    src = h5py.File(args.input, "r")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dst = h5py.File(args.output, "w")

    demos = sorted(src["data"].keys(), key=lambda x: int(x[5:]))
    dst_data = dst.create_group("data")
    total_samples = 0

    for ep in tqdm(demos, desc="Subsampling demos"):
        ep_grp = dst_data.create_group(ep)
        n = subsample_demo(src[f"data/{ep}"], ep_grp, args.skip)
        total_samples += n

    dst_data.attrs["total"] = total_samples

    if "env_args" in src["data"].attrs:
        env_args = json.loads(src["data"].attrs["env_args"])
        if "env_kwargs" in env_args:
            env_args["env_kwargs"]["control_freq"] = 20 // args.skip
        dst_data.attrs["env_args"] = json.dumps(env_args, indent=4)

    if "mask" in src:
        mask_grp = dst.create_group("mask")
        for split in src["mask"]:
            mask_grp.create_dataset(split, data=src[f"mask/{split}"][()])

    src.close()
    dst.close()
    print(f"Wrote {total_samples} total subsampled transitions across {len(demos)} demos to {args.output}")


if __name__ == "__main__":
    main()
