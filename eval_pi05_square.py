"""
Evaluate pi0.5 (pi05_droid or fine-tuned pi05_robosuite_square) on robosuite NutAssemblySquare.

Uses JOINT_VELOCITY controller and DROID-format observations (matches Ctrl-World / openpi).
For pi05_robosuite_square, norm stats normally live under checkpoint ``assets/local/robomimic_square_jvel/``;
use ``--checkpoint-norm-stats-asset-id droid`` only for older checkpoints saved with DROID base stats.
"""

from __future__ import annotations

import csv
import logging
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro

# Allow importing openpi / openpi_client: OPENPI_ROOT env (e.g. Slurm), else sibling checkout.
def _resolve_openpi_root() -> pathlib.Path | None:
    env = os.environ.get("OPENPI_ROOT")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    sibling = pathlib.Path(__file__).resolve().parent.parent / "openpi"
    return sibling if sibling.is_dir() else None


_OPENPI_ROOT = _resolve_openpi_root()
if _OPENPI_ROOT is not None:
    sys.path.insert(0, str(_OPENPI_ROOT / "src"))
    _client = _OPENPI_ROOT / "packages" / "openpi-client" / "src"
    if _client.is_dir():
        sys.path.insert(0, str(_client))


def _resize_hwc_uint8(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize (H,W,3) uint8 with PIL for stable bicubic."""
    from PIL import Image

    pil = Image.fromarray(img)
    pil = pil.resize((width, height), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def _build_policy_obs(
    obs: dict[str, Any],
    *,
    resize_size: int,
    prompt: str,
    joint_key: str,
    gripper_key: str,
    agentview_key: str,
    wrist_key: str,
) -> dict[str, Any]:
    agentview = np.asarray(obs[agentview_key])
    wrist = np.asarray(obs[wrist_key])
    # Match Ctrl-World: native 192x320 then policy resize_with_pad to square
    if agentview.shape[0] != 192 or agentview.shape[1] != 320:
        agentview = _resize_hwc_uint8(agentview, 320, 192)
        wrist = _resize_hwc_uint8(wrist, 320, 192)

    from openpi_client import image_tools

    return {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(agentview, resize_size, resize_size)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, resize_size, resize_size)
        ),
        "observation/joint_position": np.asarray(obs[joint_key], dtype=np.float32).reshape(-1)[:7],
        "observation/gripper_position": np.asarray(obs[gripper_key], dtype=np.float32).reshape(-1)[:1],
        "prompt": prompt,
    }


def _postprocess_action_droid_to_robosuite(action: np.ndarray) -> np.ndarray:
    """Binarize gripper like openpi droid example; map gripper to [-1, 1] for robosuite."""
    a = np.asarray(action, dtype=np.float64).copy()
    if a[-1] > 0.5:
        a[-1] = 1.0
    else:
        a[-1] = 0.0
    a[:7] = np.clip(a[:7], -1.0, 1.0)
    # robosuite Panda gripper command is typically in [-1, 1]
    a[-1] = 2.0 * a[-1] - 1.0
    return a.astype(np.float32)


def _make_env(
    *,
    controller_config_path: pathlib.Path,
    control_freq: int,
    horizon: int,
    camera_height: int,
    camera_width: int,
):
    import sys

    import robosuite as rs

    _rd = pathlib.Path(__file__).resolve().parent
    if str(_rd) not in sys.path:
        sys.path.insert(0, str(_rd))

    from robosuite_controller_config import load_pi05_panda_composite_config

    controller_configs = load_pi05_panda_composite_config(controller_config_path)

    return rs.make(
        "NutAssemblySquare",
        robots="Panda",
        controller_configs=controller_configs,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=True,
        control_freq=control_freq,
        horizon=horizon,
    )


def _check_success(env) -> bool:
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def _sim_rgb_video_frame(env, *, camera_names: list[str], height: int, width: int) -> np.ndarray:
    """
    Off-screen render for video, matching robomimic EnvRobosuite.render(mode='rgb_array'):
    MuJoCo/robosuite images use a flipped vertical axis, so we apply im[::-1]. Multiple
    cameras are concatenated horizontally like run_trained_agent.rollout.

    See robomimic/envs/env_robosuite.py (render, get_observation).
    """
    parts: list[np.ndarray] = []
    for cam_name in camera_names:
        im = env.sim.render(width=width, height=height, camera_name=cam_name)
        parts.append(im[::-1])
    return np.concatenate(parts, axis=1)


@dataclass
class Args:
    """Roll out pi0.5 on NutAssemblySquare."""

    # Policy: either WebSocket server or local checkpoint
    host: str = "localhost"
    port: int = 8000
    use_local_policy: bool = False
    """If True, load policy locally (no server). Requires --policy-config and --checkpoint-dir."""

    policy_config: str = "pi05_droid"
    """openpi TrainConfig name: pi05_droid or pi05_robosuite_square."""

    checkpoint_dir: str | None = None
    """Path to checkpoint (local or gs://). Required when use_local_policy is True."""

    checkpoint_norm_stats_asset_id: str | None = None
    """If set, load normalization stats from ``<checkpoint>/assets/<this>/`` instead of the TrainConfig default.
    Use ``droid`` only for legacy fine-tunes saved before robosuite-specific norm stats (under assets/droid/)."""

    # Env
    controller_config: pathlib.Path = pathlib.Path(__file__).resolve().parent / "controller_configs" / "joint_velocity_panda.json"
    """JSON file: a full `BASIC` composite (type + body_parts), or arm-only keys merged onto robosuite's default `JOINT_VELOCITY` part config for Panda."""
    control_freq: int = 20
    horizon: int = 400
    camera_height: int = 192
    camera_width: int = 320

    # Rollout
    num_rollouts: int = 10
    max_steps: int = 400
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 0

    prompt: str = "fit the square nut onto the square peg"

    video_out_path: str | None = None
    """If set, save rollouts as MP4. Uses env.sim.render (like robomimic run_trained_agent), not policy input."""

    video_fps: float = 20.0
    """Frames per second for saved video (robomimic default for imageio writers is 20)."""

    video_skip: int = 1
    """Write one video frame every n environment steps (1 = every step; larger = smaller files)."""

    video_width: int = 512
    video_height: int = 512
    """Off-screen render resolution per camera (robomimic run_trained_agent defaults)."""

    video_camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    """Cameras to render, concatenated left-to-right (same pattern as robomimic --camera_names)."""

    video_crf: int = 18
    """libx264 CRF for ffmpeg (lower = better quality; 18 is visually near-lossless for most content)."""

    results_csv: pathlib.Path | None = None

    # Observation keys (robosuite v1 Panda)
    joint_key: str = "robot0_joint_pos"
    gripper_key: str = "robot0_gripper_qpos"
    agentview_key: str = "agentview_image"
    wrist_key: str = "robot0_eye_in_hand_image"


def _disable_hf_flash_attn_for_conda_abi_mismatch() -> None:
    """Skip Flash-Attention-2/3 in HuggingFace when flash_attn is installed but its CUDA extension
    does not match the active PyTorch (common in mixed conda/cluster setups).

    HF only imports flash_attn when ``is_flash_attn_*_available()`` is true; otherwise it uses
    SDPA/eager. The websocket eval path avoids this entirely (policy runs under openpi ``uv run``).

    Set ``EVAL_PI05_ALLOW_FLASH_ATTN=1`` to leave HF's default detection enabled.
    """
    if os.environ.get("EVAL_PI05_ALLOW_FLASH_ATTN", "").lower() in ("1", "true", "yes"):
        return
    import transformers.utils as _tu
    import transformers.utils.import_utils as _iu

    def _false() -> bool:
        return False

    _iu.is_flash_attn_2_available = _false  # type: ignore[method-assign]
    _iu.is_flash_attn_3_available = _false  # type: ignore[method-assign]
    _tu.is_flash_attn_2_available = _false  # type: ignore[method-assign]
    _tu.is_flash_attn_3_available = _false  # type: ignore[method-assign]


def eval_square(args: Args) -> None:
    np.random.seed(args.seed)

    env = _make_env(
        controller_config_path=args.controller_config,
        control_freq=args.control_freq,
        horizon=args.horizon,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )

    if args.use_local_policy:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required when --use-local-policy is set")
        _disable_hf_flash_attn_for_conda_abi_mismatch()
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config

        train_cfg = openpi_config.get_config(args.policy_config)
        policy_kwargs: dict[str, Any] = {}
        if args.checkpoint_norm_stats_asset_id is not None:
            from openpi.training import checkpoints as openpi_checkpoints

            policy_kwargs["norm_stats"] = openpi_checkpoints.load_norm_stats(
                pathlib.Path(args.checkpoint_dir).resolve() / "assets",
                args.checkpoint_norm_stats_asset_id,
            )
        policy = policy_config.create_trained_policy(train_cfg, args.checkpoint_dir, **policy_kwargs)

        def infer_fn(obs_dict: dict) -> dict:
            return policy.infer(obs_dict)

    else:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        def infer_fn(obs_dict: dict) -> dict:
            return client.infer(obs_dict)

    if args.video_out_path:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    rows: list[dict[str, Any]] = []
    successes = 0

    for ep in range(args.num_rollouts):
        ep_seed = args.seed + ep
        np.random.seed(ep_seed)
        # Gymnasium envs may define env.seed = None; only call if actually implemented.
        if callable(getattr(env, "seed", None)):
            env.seed(ep_seed)
        obs = env.reset()

        action_plan: list[np.ndarray] = []
        total_r = 0.0
        success = False

        video_writer: Any = None
        video_tmp_path: pathlib.Path | None = None
        video_cameras = list(args.video_camera_names)
        if args.video_out_path and args.max_steps > 0:
            fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=args.video_out_path)
            os.close(fd)
            video_tmp_path = pathlib.Path(tmp_name)
            video_writer = imageio.get_writer(
                video_tmp_path,
                fps=args.video_fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=[
                    "-crf",
                    str(args.video_crf),
                    "-preset",
                    "medium",
                ],
            )
        video_count = 0
        last_step = 0

        for t in range(args.max_steps):
            last_step = t + 1
            if not action_plan:
                element = _build_policy_obs(
                    obs,
                    resize_size=args.resize_size,
                    prompt=args.prompt,
                    joint_key=args.joint_key,
                    gripper_key=args.gripper_key,
                    agentview_key=args.agentview_key,
                    wrist_key=args.wrist_key,
                )
                out = infer_fn(element)
                chunk = np.asarray(out["actions"])
                if chunk.ndim != 2:
                    raise ValueError(f"Expected actions (T, 8), got shape {chunk.shape}")
                if chunk.shape[0] < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {chunk.shape[0]} steps but replan_steps={args.replan_steps}"
                    )
                n = min(args.replan_steps, chunk.shape[0])
                action_plan = [chunk[i] for i in range(n)]

            raw_action = action_plan.pop(0)
            step_action = _postprocess_action_droid_to_robosuite(raw_action)
            obs, r, done, _ = env.step(step_action)
            total_r += float(r)
            if _check_success(env):
                success = True

            if video_writer is not None:
                if video_count % args.video_skip == 0:
                    video_writer.append_data(
                        _sim_rgb_video_frame(
                            env,
                            camera_names=video_cameras,
                            height=args.video_height,
                            width=args.video_width,
                        )
                    )
                video_count += 1

            if done:
                break

        if video_writer is not None:
            video_writer.close()
            assert video_tmp_path is not None
            final_mp4 = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{ep:04d}_{'success' if success else 'fail'}.mp4"
            )
            if final_mp4.exists():
                final_mp4.unlink()
            video_tmp_path.replace(final_mp4)
            logging.info("Wrote %s", final_mp4)

        successes += int(success)
        logging.info(
            "Episode %s/%s: success=%s return=%.4f",
            ep + 1,
            args.num_rollouts,
            success,
            total_r,
        )

        rows.append(
            {
                "episode": ep,
                "success": int(success),
                "return": total_r,
                "steps": last_step,
            }
        )

    rate = successes / max(args.num_rollouts, 1)
    logging.info("Success rate: %s / %s = %.4f", successes, args.num_rollouts, rate)

    if args.results_csv and rows:
        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logging.info("Wrote results to %s", args.results_csv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    # Parse Args directly so flags are --host, --port, ... not --args.host (tyro nests on the parameter name).
    cfg = tyro.cli(Args)
    eval_square(cfg)


if __name__ == "__main__":
    main()
