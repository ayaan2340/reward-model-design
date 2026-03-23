import mediapy
import os
import torch
import numpy as np
import h5py
import json
from scipy.spatial.transform import Rotation
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset
from accelerate import Accelerator

"""Number of cameras is inferred from the dataset metadata."""
NUM_CAMERAS_EXPECTED = None

class EncodeLatentDataset(Dataset):
    def __init__(self, hdf5_path, new_path, svd_path, device, size=(192, 320), rgb_skip=4):
        self.hdf5_path = hdf5_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        # Build demos arrat sorted by demo index
        self.f = h5py.File(hdf5_path, "r")
        demos = list(self.f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        self.demos = [demos[i] for i in inds]

        # Mark each demo as train/val
        self.demo_to_split = {}
        if "mask" in self.f:
            if "train" in self.f["mask"]:
                for demo in self.f["mask/train"][()]:
                    key = demo.decode() if isinstance(demo, bytes) else demo
                    self.demo_to_split[key] = "train"
            if "valid" in self.f["mask"]:
                for demo in self.f["mask/valid"][()]:
                    key = demo.decode() if isinstance(demo, bytes) else demo
                    self.demo_to_split[key] = "val"


        # Derive camera ordering from actual obs keys to avoid missing views that were
        # saved via obs_camera_names during rollout (e.g., frontview even if the policy
        # didn't consume it). We enforce the desired ordering and require all three.
        first_ep = self.demos[0]
        obs_keys = list(self.f[f"data/{first_ep}/obs"].keys())
        cam_keys = [k for k in obs_keys if k.endswith("_image")]
        available_cams = [k[:-6] for k in cam_keys]  # strip trailing _image

        desired_order = ["agentview", "frontview", "robot0_eye_in_hand"]
        missing = [c for c in desired_order if c not in available_cams]
        if missing:
            raise ValueError(f"Dataset is missing required camera views: {missing}. Available: {available_cams}")

        ordered = [c for c in desired_order if c in available_cams]
        extras = [c for c in available_cams if c not in desired_order]
        self.camera_names = ordered + sorted(extras)

        self.instruction = "The robot must fit the square nut onto the square peg"

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        ep = self.demos[idx]
        traj_id = int(ep[5:])

        data_type = self.demo_to_split.get(ep, "train")

        ep_grp = self.f[f"data/{ep}"]


        # eef_pos (N,3) and eef_quat (N,4) to cartesian_position as [x,y,z,roll,pitch,yaw] for DROID compatibility
        eef_pos = ep_grp["obs/robot0_eef_pos"][()]       
        eef_quat = ep_grp["obs/robot0_eef_quat"][()]
        eef_euler = Rotation.from_quat(eef_quat).as_euler("xyz")
        obs_cartesian = np.concatenate((eef_pos, eef_euler), axis=-1)

        obs_joint = ep_grp["obs/robot0_joint_pos"][()]

        gripper_qpos = ep_grp["obs/robot0_gripper_qpos"][()]
        obs_gripper = gripper_qpos.min(axis=-1) # Get minimum gripper value between two fingers for DROID compatibility (single scalar gripper position)

        actions_arr = ep_grp["actions"][()]
        action_cartesian = actions_arr[:, :6]
        action_gripper = actions_arr[:, 6:7]
        obs_joint_vel = ep_grp["obs/robot0_joint_vel"][()]

        dones = ep_grp["dones"][()]
        success = bool(np.any(dones > 0))

        raw_images = {}
        image_arrays = []
        # Preserve ordering from env meta so paths stay deterministic
        for cam_name in self.camera_names:
            if f"obs/{cam_name}_image" not in ep_grp:
                raise KeyError(f"Camera '{cam_name}' not found in dataset obs keys")
            arr = ep_grp[f"obs/{cam_name}_image"][()]
            image_arrays.append(arr)
            raw_images[cam_name] = arr

        # Some downstream consumers expect three views; if the dataset only has two,
        # keep the available ordering without inserting None placeholders.

        traj_info = {
            "success": success,
            "observation.state.cartesian_position": obs_cartesian.tolist(),
            "observation.state.joint_position": obs_joint.tolist(),
            "observation.state.gripper_position": obs_gripper.tolist(),
            "action.cartesian_position": action_cartesian.tolist(),
            "action.gripper_position": action_gripper.tolist(),
            "action.joint_velocity": obs_joint_vel.tolist(),
        }

        print(f"[{idx+1}/{len(self.demos)}] Processing traj {traj_id} ({data_type}), length={len(obs_joint)}")
        try:
            self.process_traj(
                image_arrays, traj_info, self.instruction, self.new_path,
                traj_id=traj_id, data_type=data_type, size=self.size,
                rgb_skip=self.skip, device=self.vae.device,
            )
            print(f"[{idx+1}/{len(self.demos)}] Done traj {traj_id}")
        except Exception as e:
            print(f"[{idx+1}/{len(self.demos)}] Error on traj {traj_id}: {e}, skipping...")
            return 0

        return 0

    def process_traj(self, image_arrays, traj_info, instruction, save_root,
                     traj_id=0, data_type="val", size=(192, 320), rgb_skip=4, device="cuda"):
        num_cameras = len(image_arrays)

        for video_id, images in enumerate(image_arrays):
            frames = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
            frames = frames[::rgb_skip]  # Subsample frames down to 5 Hz
            x = frames

            save_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            save_video = save_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", save_video, fps=5)

            # Encode through SVD VAE and save latent
            x = x.to(device)
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i + batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")

        cartesian_pose = np.array(traj_info["observation.state.cartesian_position"])
        gripper_scalar = np.array(traj_info["observation.state.gripper_position"])[:, None]
        cartesian_states = np.concatenate((cartesian_pose, gripper_scalar), axis=-1)[::rgb_skip].tolist()

        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info["success"]),
            "video_length": frames.shape[0],
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info["observation.state.cartesian_position"]),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/{i}.mp4"}
                for i in range(num_cameras)
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/{i}.pt"}
                for i in range(num_cameras)
            ],
            "states": cartesian_states,
            # Full-rate observation states (consumed by Dataset_mix via state_id indexing)
            "observation.state.cartesian_position": traj_info["observation.state.cartesian_position"],
            "observation.state.joint_position": traj_info["observation.state.joint_position"],
            "observation.state.gripper_position": traj_info["observation.state.gripper_position"],
            # Supplementary action fields (not consumed during Ctrl-World training)
            "action.cartesian_position": traj_info["action.cartesian_position"],
            "action.gripper_position": traj_info["action.gripper_position"],
            "action.joint_velocity": traj_info["action.joint_velocity"],
        }
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True,
                        help="path to robomimic HDF5 dataset (e.g. square_image_dense.hdf5)")
    parser.add_argument("--output_path", type=str, default="dataset_example/robomimic_square",
                        help="output directory for extracted videos, latents, and annotations")
    parser.add_argument("--svd_path", type=str, default="/cephfs/shared/llm/stable-video-diffusion-img2vid",
                        help="path to stable-video-diffusion model (contains vae subfolder)")
    parser.add_argument("--rgb_skip", type=int, default=4,
                        help="frame skip factor (robomimic is 20Hz; use 4 for ~5fps, 1 for no skip)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        hdf5_path=args.hdf5_path,
        new_path=args.output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=args.rgb_skip,
    )
    tmp_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples")

# Example usage:
# accelerate launch reward-model-design/extract_latent_robomimic.py \
#     --hdf5_path /projects/bggq/asunesara/square_image_dense_reshaped.hdf5 \
#     --output_path dataset_example/robomimic_square \
#     --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid \
#     --rgb_skip 4 --debug

