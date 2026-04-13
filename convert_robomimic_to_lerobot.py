"""
Convert robomimic PH HDF5 (e.g. NutAssemblySquare) to LeRobot format for openpi fine-tuning (pi05_robosuite_square).

Uses joint velocities from obs/robot0_joint_vel and gripper from obs/robot0_gripper_qpos.
Images are resized to 180x320 (H,W) to match DROID LeRobot convention in openpi.

After conversion, from the openpi repo compute normalization stats on your LeRobot dataset (required for training/eval):
  uv run scripts/compute_norm_stats.py --config-name pi05_robosuite_square
This writes ./assets/pi05_robosuite_square/local/robomimic_square_jvel/norm_stats.json (repo_id must match).
The pi05_robosuite_square TrainConfig loads those stats (not the base DROID pi05_droid assets).

Run this script with openpi's environment (lerobot is pinned in openpi/pyproject.toml), e.g.:
  cd /path/to/openpi && uv run python /path/to/reward-model-design/convert_robomimic_to_lerobot.py /path/to/data.hdf5

Do not use a generic conda env unless you install the same lerobot revision openpi uses; otherwise you get
ModuleNotFoundError: No module named 'lerobot.common'.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    raise ImportError(
        "Missing Hugging Face LeRobot (package layout lerobot.common.datasets). "
        "Run this script via openpi: `cd openpi && uv run python /path/to/convert_robomimic_to_lerobot.py ...` "
        "or install openpi's lerobot pin (see openpi/pyproject.toml [tool.uv.sources] lerobot)."
    ) from e


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """image: (H,W,3) uint8 -> resized"""
    pil = Image.fromarray(image)
    pil = pil.resize((width, height), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def _get_obs(demo_grp, t: int, candidates: list[str]) -> np.ndarray:
    obs = demo_grp["obs"]
    for name in candidates:
        key = f"{name}"
        if key in obs:
            return np.asarray(obs[key][t])
    raise KeyError(f"None of {candidates} found under obs/ in this demo")


def convert(
    hdf5_path: Path,
    *,
    repo_id: str = "local/robomimic_square_jvel",
    out_width: int = 320,
    out_height: int = 180,
    task_text: str = "fit the square nut onto the square peg",
    overwrite: bool = True,
) -> None:
    hdf5_path = hdf5_path.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=20,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (out_height, out_width, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (out_height, out_width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (out_height, out_width, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )

        for demo_key in tqdm(demos, desc="demos"):
            demo = f["data"][demo_key]
            n = int(demo.attrs["num_samples"])

            for t in range(n):
                agent = _get_obs(demo, t, ["agentview_image", "agentview_rgb"])
                wrist = _get_obs(demo, t, ["robot0_eye_in_hand_image", "eye_in_hand_image"])

                agent = _resize_image(agent, out_width, out_height)
                wrist = _resize_image(wrist, out_width, out_height)
                # Second exterior: duplicate agent (DROID stereo slot; DroidInputs ignores or uses zeros at inference)
                ext2 = agent.copy()

                jp = np.asarray(demo["obs"]["robot0_joint_pos"][t], dtype=np.float32).reshape(-1)[:7]
                jv = np.asarray(demo["obs"]["robot0_joint_vel"][t], dtype=np.float32).reshape(-1)[:7]
                gq = np.asarray(demo["obs"]["robot0_gripper_qpos"][t], dtype=np.float32).reshape(-1)
                g_scalar = float(np.clip(gq[0], 0.0, 1.0))
                gripper_pos = np.array([g_scalar], dtype=np.float32)
                actions = np.concatenate([jv, gripper_pos]).astype(np.float32)

                dataset.add_frame(
                    {
                        "exterior_image_1_left": agent,
                        "exterior_image_2_left": ext2,
                        "wrist_image_left": wrist,
                        "joint_position": jp,
                        "gripper_position": gripper_pos,
                        "actions": actions,
                        "task": task_text,
                    }
                )
            dataset.save_episode()

    print(f"Wrote LeRobot dataset to {output_path}")


def main(
    hdf5_path: str,
    repo_id: str = "local/robomimic_square_jvel",
    task_text: str = "fit the square nut onto the square peg",
):
    """Convert a single robomimic HDF5 file to LeRobot format."""
    convert(Path(hdf5_path), repo_id=repo_id, task_text=task_text)


if __name__ == "__main__":
    tyro.cli(main)
