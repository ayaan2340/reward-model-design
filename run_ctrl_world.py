"""
Roll out a Ctrl-World world model using a robomimic BC-Transformer-GMM policy
trained on 5Hz accumulated delta actions.

The policy predicts 5 delta actions (1 second at 5Hz) per forward pass.  These
deltas are accumulated from the trajectory's initial absolute EEF pose into
absolute cartesian poses, then normalized with stat.json bounds and fed as
action conditioning to the world model.

Starting frames come from a pre-extracted Ctrl-World dataset (produced by
extract_latent_robomimic.py).  The script saves predicted video rollouts
to an output folder.

Usage:
    python run_ctrl_world.py \
        --robomimic_ckpt /path/to/bc_transformer.pth \
        --ctrl_world_ckpt /path/to/checkpoint-XXXXX.pt \
        --dataset_dir /path/to/extracted_dataset \
        --output_dir /path/to/output \
        --num_rollouts 5 \
        --svd_model_path /path/to/stable-video-diffusion-img2vid \
        --clip_model_path /path/to/clip-vit-base-patch32 \
        --data_stat_path /path/to/stat.json
"""

import argparse
import datetime
import json
import os
import sys

import einops
import mediapy
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robomimic"))
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Ctrl-World"))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld

# Camera view ordering -- must match Ctrl-World latent stacking and policy training
CAMERA_ORDER = ["agentview", "frontview", "robot0_eye_in_hand"]
INSTRUCTION = "The robot must fit the square nut onto the square peg"

# Ctrl-World world model hyperparameters (from config.py defaults)
NUM_FRAMES = 5
NUM_HISTORY = 6
ACTION_DIM = 7
PRED_STEP = 5
INTERACT_NUM = 20
HISTORY_IDX = [0, 0, -12, -9, -6, -3]


def normalize_bound(data, data_min, data_max, eps=1e-8):
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, -1, 1)


def denormalize_bound(data, data_min, data_max):
    return (data + 1) / 2 * (data_max - data_min) + data_min


def load_ctrl_world(args, device, dtype):
    """Load and return the Ctrl-World model ready for inference."""

    class WMArgs:
        pass

    wm_args = WMArgs()
    wm_args.svd_model_path = args.svd_model_path
    wm_args.clip_model_path = args.clip_model_path
    wm_args.action_dim = ACTION_DIM
    wm_args.num_frames = NUM_FRAMES
    wm_args.num_history = NUM_HISTORY
    wm_args.text_cond = True
    wm_args.frame_level_cond = True
    wm_args.his_cond_zero = False
    wm_args.width = 320
    wm_args.height = 192
    wm_args.motion_bucket_id = 127
    wm_args.fps = 7
    wm_args.guidance_scale = 2
    wm_args.num_inference_steps = 50
    wm_args.decode_chunk_size = 7

    model = CrtlWorld(wm_args)
    # train_wm.py deletes vae and image_encoder from the module before training, so
    # checkpoints only store trainable weights (unet, action_encoder, text_encoder, ...).
    # Keep frozen VAE + image_encoder from svd_model_path; load the rest from ckpt.
    ckpt = torch.load(args.ctrl_world_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and not any(k.startswith("unet.") for k in ckpt):
        for key in ("state_dict", "model", "model_state_dict", "ema_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(
            "load_ctrl_world: ignored missing keys in checkpoint (expected for frozen "
            f"VAE/image_encoder if absent): {len(missing)} keys"
        )
    if unexpected:
        print(f"load_ctrl_world: unexpected keys in checkpoint: {unexpected}")
    model.to(device).to(dtype)
    model.eval()
    print("Loaded Ctrl-World model")
    return model, wm_args


def load_robomimic_policy(ckpt_path, device):
    """Load BC-Transformer policy and return RolloutPolicy, ckpt_dict, and action norm stats."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    policy.start_episode()

    action_norm_stats = ckpt_dict.get("action_normalization_stats", None)
    if action_norm_stats is not None:
        for key in action_norm_stats:
            for stat_key in action_norm_stats[key]:
                action_norm_stats[key][stat_key] = np.array(action_norm_stats[key][stat_key])
    return policy, ckpt_dict, action_norm_stats


def get_traj_initial_state(dataset_dir, episode_id, start_idx, state_p01, state_p99, device, dtype, vae):
    """Load annotation, latent videos, and initial state for one trajectory."""
    # try val first, then train
    for split in ["val", "train"]:
        ann_path = os.path.join(dataset_dir, "annotation", split, f"{episode_id}.json")
        if os.path.exists(ann_path):
            break
    with open(ann_path, "r") as f:
        anno = json.load(f)

    instruction = anno["texts"][0] if anno.get("texts") else INSTRUCTION

    # load states (5Hz, already cartesian+gripper, 7D)
    states = np.array(anno["states"])  # (T, 7)

    # load latent videos for each camera
    video_latents = []
    for cam_idx in range(len(anno["latent_videos"])):
        lpath = anno["latent_videos"][cam_idx]["latent_video_path"]
        lpath = os.path.join(dataset_dir, lpath)
        lat = torch.load(lpath, map_location="cpu")  # (T, 4, H, W)
        video_latents.append(lat.to(device).to(dtype))

    # load pixel videos for initial obs construction
    from decord import VideoReader, cpu as decord_cpu
    pixel_videos = []
    for cam_idx in range(len(anno["videos"])):
        vpath = anno["videos"][cam_idx]["video_path"]
        vpath = os.path.join(dataset_dir, vpath)
        vr = VideoReader(vpath, ctx=decord_cpu(0), num_threads=2)
        try:
            frames = vr.get_batch(range(len(vr))).asnumpy()
        except Exception:
            frames = vr.get_batch(range(len(vr))).numpy()
        pixel_videos.append(frames)

    initial_state = states[start_idx]  # (7,) absolute cartesian pose
    initial_state_norm = normalize_bound(initial_state[None, :], state_p01, state_p99)  # (1, 7)

    # stack 3-camera latents at start_idx into (4, 72, 40)
    first_latent = torch.cat([v[start_idx] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)

    initial_pixel_obs = [v[start_idx] for v in pixel_videos]  # list of 3 x (H, W, 3)

    return {
        "instruction": instruction,
        "states": states,
        "video_latents": video_latents,
        "pixel_videos": pixel_videos,
        "initial_state": initial_state,
        "initial_state_norm": initial_state_norm,
        "first_latent": first_latent,
        "initial_pixel_obs": initial_pixel_obs,
        "start_idx": start_idx,
    }


def decode_latents_to_pixels(latents, pipeline, decode_chunk_size, dtype):
    """Decode VAE latents to uint8 pixel arrays. latents: (B, F, C, H, W)"""
    bsz, frame_num = latents.shape[:2]
    x = latents.flatten(0, 1)
    decoded = []
    decode_kwargs = {}
    for i in range(0, x.shape[0], decode_chunk_size):
        chunk = x[i:i + decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    videos = torch.cat(decoded, dim=0)
    videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
    videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    return videos  # (B, F, H, W, 3)


def build_obs_dict_from_pixels(pixel_frames, eef_state_raw):
    """
    Build a robomimic-compatible observation dict from decoded pixel frames
    and raw (denormalized) EEF state.

    Images are passed at their native resolution (192x320 from the WM decoder)
    so that the policy's CropRandomizer (trained on 192x320) works correctly.

    Args:
        pixel_frames: list of 3 np arrays, each (H, W, 3) uint8 for
                      [agentview, frontview, robot0_eye_in_hand]
        eef_state_raw: (7,) array [x,y,z,roll,pitch,yaw,gripper_scalar]
    """
    obs = {}
    cam_keys = ["agentview_image", "frontview_image", "robot0_eye_in_hand_image"]
    for cam_key, frame in zip(cam_keys, pixel_frames):
        obs[cam_key] = frame

    obs["robot0_eef_pos"] = eef_state_raw[:3].astype(np.float32)
    obs["robot0_eef_quat"] = Rotation.from_euler("xyz", eef_state_raw[3:6]).as_quat().astype(np.float32)
    obs["robot0_gripper_qpos"] = np.array([eef_state_raw[6], -eef_state_raw[6]], dtype=np.float32)
    return obs


def get_delta_chunk(policy, obs_buffer, action_norm_stats):
    """
    Run the BC-Transformer-GMM policy on an observation buffer to get the
    full 5-delta action chunk, bypassing get_action's single-step reduction.

    The network output is in min_max-normalized space; this function
    unnormalizes it back to raw delta actions.

    Args:
        policy: RolloutPolicy wrapper
        obs_buffer: list of 5 obs dicts (one per context frame)
        action_norm_stats: dict with "actions" -> {"scale": ..., "offset": ...}

    Returns:
        deltas: (5, 7) numpy array of raw (unnormalized) delta actions
    """
    stacked_obs = {}
    for k in obs_buffer[0]:
        stacked_obs[k] = np.stack([ob[k] for ob in obs_buffer], axis=0)

    prepared = policy._prepare_observation(stacked_obs, batched_ob=False)
    with torch.no_grad():
        chunk = policy.policy.nets["policy"](prepared, actions=None)
    chunk_np = chunk[0].cpu().numpy()  # (5, 7) normalized

    scale = action_norm_stats["actions"]["scale"]
    offset = action_norm_stats["actions"]["offset"]
    return chunk_np * scale + offset  # unnormalize: x * scale + offset


def deltas_to_absolute_poses_norm(deltas_5hz, initial_state, state_p01, state_p99):
    """Convert 5 raw delta actions into 5 normalized absolute poses for the WM.

    Args:
        deltas_5hz: (5, 7) unnormalized delta actions [dx,dy,dz,dax,day,daz,gripper]
        initial_state: (7,) [x,y,z,roll,pitch,yaw,gripper] starting absolute pose
        state_p01: (1, 7) normalization lower bound from stat.json
        state_p99: (1, 7) normalization upper bound from stat.json

    Returns:
        poses_norm: (5, 7) normalized absolute poses in [-1, 1]
    """
    current_pos = initial_state[:3].copy()
    current_rot = Rotation.from_euler("xyz", initial_state[3:6])
    poses = []
    for t in range(deltas_5hz.shape[0]):
        current_pos = current_pos + deltas_5hz[t, :3]
        delta_rot = Rotation.from_rotvec(deltas_5hz[t, 3:6])
        current_rot = delta_rot * current_rot
        euler = current_rot.as_euler("xyz")
        gripper = deltas_5hz[t, 6]
        poses.append(np.concatenate([current_pos.copy(), euler, [gripper]]))
    poses = np.stack(poses)  # (5, 7)
    return normalize_bound(poses, state_p01, state_p99)


def forward_world_model(model, wm_args, action_cond, current_latent, his_latent, text, device, dtype):
    """
    Run the Ctrl-World world model forward.

    Args:
        action_cond: (num_history+num_frames, 7) numpy, already normalized to [-1, 1]
        current_latent: (1, 4, 72, 40) tensor, current frame latent
        his_latent: (1, num_history, 4, 72, 40) tensor, history latents
        text: instruction string or None
    Returns:
        predict_latents: raw latents from the pipeline, rearranged per-view
        decoded_videos: (3, num_frames, H, W, 3) uint8 numpy, per-view decoded frames
    """
    action_tensor = torch.tensor(action_cond, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        if text is not None:
            text_token = model.action_encoder(action_tensor, text, model.tokenizer, model.text_encoder)
        else:
            text_token = model.action_encoder(action_tensor)

        pipeline = model.pipeline
        _, latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=text_token,
            width=wm_args.width,
            height=int(wm_args.height * 3),
            num_frames=wm_args.num_frames,
            history=his_latent,
            num_inference_steps=wm_args.num_inference_steps,
            decode_chunk_size=wm_args.decode_chunk_size,
            max_guidance_scale=wm_args.guidance_scale,
            fps=wm_args.fps,
            motion_bucket_id=wm_args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )

    # split stacked 3-camera latents into per-view: (3, F, C, H, W)
    predict_latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

    decoded_videos = decode_latents_to_pixels(
        predict_latents, pipeline, wm_args.decode_chunk_size, dtype
    )  # (3, F, H, W, 3)

    return predict_latents, decoded_videos


def main():
    parser = argparse.ArgumentParser(description="Roll out Ctrl-World with robomimic BC-Transformer policy")
    parser.add_argument("--robomimic_ckpt", type=str, required=True, help="path to BC-Transformer checkpoint")
    parser.add_argument("--ctrl_world_ckpt", type=str, required=True, help="path to Ctrl-World checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="path to extracted Ctrl-World dataset (from extract_latent_robomimic.py)")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for rollout videos")
    parser.add_argument("--num_rollouts", type=int, default=5, help="number of rollouts to generate")
    parser.add_argument("--interact_num", type=int, default=INTERACT_NUM, help="number of interaction steps per rollout")
    parser.add_argument("--svd_model_path", type=str, required=True, help="path to stable-video-diffusion-img2vid")
    parser.add_argument("--clip_model_path", type=str, required=True, help="path to clip-vit-base-patch32")
    parser.add_argument("--data_stat_path", type=str, required=True, help="path to stat.json for normalization bounds")
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    dtype = torch.bfloat16

    # load normalization stats
    with open(args.data_stat_path, "r") as f:
        data_stat = json.load(f)
    state_p01 = np.array(data_stat["state_01"])[None, :]
    state_p99 = np.array(data_stat["state_99"])[None, :]

    # load models
    policy, ckpt_dict, action_norm_stats = load_robomimic_policy(args.robomimic_ckpt, device)
    model, wm_args = load_ctrl_world(args, device, dtype)

    # discover available trajectories from dataset annotation
    traj_ids = []
    for split in ["val", "train"]:
        ann_dir = os.path.join(args.dataset_dir, "annotation", split)
        if os.path.isdir(ann_dir):
            for fn in sorted(os.listdir(ann_dir)):
                if fn.endswith(".json"):
                    traj_ids.append((split, fn[:-5]))  # (split, episode_id)
    if not traj_ids:
        raise RuntimeError(f"No annotation files found in {args.dataset_dir}/annotation/")

    os.makedirs(args.output_dir, exist_ok=True)

    for rollout_idx in range(args.num_rollouts):
        split, episode_id = traj_ids[rollout_idx % len(traj_ids)]
        start_idx = 0
        print(f"\n{'='*60}")
        print(f"Rollout {rollout_idx+1}/{args.num_rollouts}: episode={episode_id}, split={split}")
        print(f"{'='*60}")

        traj = get_traj_initial_state(
            args.dataset_dir, episode_id, start_idx, state_p01, state_p99,
            device, dtype, model.pipeline.vae,
        )

        # initialize history buffers
        his_cond = []
        his_eef = []
        for _ in range(NUM_HISTORY * 4):
            his_cond.append(traj["first_latent"])
            his_eef.append(traj["initial_state_norm"].copy())

        # initialize observation frame buffer (5 frames for context_length=5)
        current_raw_state = traj["initial_state"].copy()
        obs_buffer = []
        init_obs = build_obs_dict_from_pixels(traj["initial_pixel_obs"], current_raw_state)
        for _ in range(5):
            obs_buffer.append(init_obs)

        video_frames_to_save = []
        policy.start_episode()

        for step_i in range(args.interact_num):
            print(f"  Step {step_i+1}/{args.interact_num}")

            # -- Policy forward: get 5 raw delta actions --
            deltas = get_delta_chunk(policy, obs_buffer, action_norm_stats)  # (5, 7)

            # -- Convert deltas to normalized absolute poses for WM --
            abs_poses_norm = deltas_to_absolute_poses_norm(
                deltas, current_raw_state, state_p01, state_p99
            )  # (5, 7) normalized

            # -- Build action conditioning for world model --
            action_cond = np.concatenate(
                [his_eef[idx] for idx in HISTORY_IDX], axis=0
            )  # (6, 7)
            action_cond = np.concatenate([action_cond, abs_poses_norm], axis=0)  # (11, 7)

            his_latent = torch.cat(
                [his_cond[idx] for idx in HISTORY_IDX], dim=0
            ).unsqueeze(0)  # (1, 6, 4, 72, 40)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)

            # -- World model forward --
            predict_latents, decoded_videos = forward_world_model(
                model, wm_args, action_cond, current_latent, his_latent,
                traj["instruction"], device, dtype,
            )
            # decoded_videos: (3, 5, H, W, 3) -- 3 cameras, 5 frames each

            # -- Update history buffers --
            last_pose_norm = abs_poses_norm[PRED_STEP - 1:PRED_STEP]  # (1, 7)
            his_eef.append(last_pose_norm)
            # Stack 3 camera latents along H (dim=1), same as get_traj_initial_state / first_latent.
            last_latent = torch.cat(
                [predict_latents[cam_i, PRED_STEP - 1] for cam_i in range(3)], dim=1
            ).unsqueeze(0)  # each cam (4,24,40) -> (4,72,40) -> (1,4,72,40)
            his_cond.append(last_latent)

            # -- Update current_raw_state for next step's delta accumulation --
            current_raw_state = denormalize_bound(
                abs_poses_norm[PRED_STEP - 1], state_p01[0], state_p99[0]
            )

            # -- Update observation buffer for next policy call --
            obs_buffer = []
            for frame_i in range(PRED_STEP):
                frame_pixels = [decoded_videos[cam_i, frame_i] for cam_i in range(3)]
                raw_state_at_frame = denormalize_bound(
                    abs_poses_norm[frame_i], state_p01[0], state_p99[0]
                )
                obs_buffer.append(build_obs_dict_from_pixels(frame_pixels, raw_state_at_frame))

            # -- Collect frames for video (concatenate 3 camera views horizontally) --
            for frame_i in range(PRED_STEP - 1):
                concat_frame = np.concatenate(
                    [decoded_videos[cam_i, frame_i] for cam_i in range(3)], axis=1
                )  # (H, W*3, 3)
                video_frames_to_save.append(concat_frame)

        # -- Save rollout video --
        if video_frames_to_save:
            video = np.stack(video_frames_to_save, axis=0)
            uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                args.output_dir,
                f"rollout_{rollout_idx}_{episode_id}_{uuid}.mp4",
            )
            mediapy.write_video(filename, video, fps=4)
            print(f"  Saved video ({video.shape[0]} frames) to {filename}")


if __name__ == "__main__":
    main()
