"""Lazy-loaded reward backends (robometer baselines + local success detector)."""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("reward_eval.backends")


def _ensure_robometer() -> None:
    root = os.environ.get("ROBOMETER_ROOT", "/u/asunesara/robometer")
    if root not in sys.path:
        sys.path.insert(0, root)


class RewardBackend(ABC):
    name: str

    @abstractmethod
    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return per-frame predictions (T,) float and sidecar meta (warnings, timing)."""


class TopRewardBackend(RewardBackend):
    def __init__(self, model_path: str, **kwargs: Any):
        _ensure_robometer()
        from robometer.evals.baselines.topreward import TopReward

        self.name = "topreward_qwen"
        self._m = TopReward(model_path=model_path, **kwargs)
        logger.info("TopReward loaded: %s", model_path)

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = self._m.compute_progress(frames_hwc, task_description=task_text)
        pred = np.asarray(out, dtype=np.float64).ravel()
        return pred, {"backend": self.name}


class RoboDopamineBackend(RewardBackend):
    def __init__(self, model_path: str, **kwargs: Any):
        _ensure_robometer()
        from robometer.evals.baselines.robodopamine import RoboDopamine

        self.name = "robodopamine"
        self._m = RoboDopamine(model_path=model_path, **kwargs)
        logger.info("RoboDopamine loaded: %s", model_path)

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = self._m.compute_progress(frames_hwc, task_description=task_text)
        pred = np.asarray(out, dtype=np.float64).ravel()
        return pred, {"backend": self.name}


class RoboRewardBackend(RewardBackend):
    """Discrete end-of-episode score (1–5) broadcast to each frame; normalized to ~[0,1] in robometer."""

    def __init__(self, model_path: str, **kwargs: Any):
        _ensure_robometer()
        from robometer.evals.baselines.roboreward import RoboReward

        self.name = "roboreward"
        self._m = RoboReward(
            model_path=model_path,
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            use_unsloth=bool(kwargs.get("use_unsloth", True)),
        )
        logger.info("RoboReward loaded: %s", model_path)

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = self._m.compute_progress(frames_hwc, task_description=task_text)
        pred = np.asarray(out, dtype=np.float64).ravel()
        # Matches robometer RoboReward: one discrete 1–5 judgment, normalized ~[0,1], replicated each frame.
        return pred, {
            "backend": self.name,
            "prediction_mode": "end_of_episode_broadcast",
            "note": "RoboReward judges full episode once; per-frame values are constant (not dense progress).",
        }


class RBMFamilyBackend(RewardBackend):
    """RBM and ReWiND (reward-fm) checkpoints via robometer RBMModel — same loader."""

    def __init__(self, checkpoint_path: str, name: str = "rbm", max_frames: int | None = 48):
        _ensure_robometer()
        from robometer.evals.baselines.rbm_model import RBMModel

        self.name = name
        self._m = RBMModel(checkpoint_path=checkpoint_path)
        # Qwen3-VL attention memory grows sharply with frame count; full-horizon rollouts OOM even on H200.
        self.max_frames = max_frames
        logger.info("%s loaded from %s (max_frames=%s)", name, checkpoint_path, max_frames)

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        hwc = np.asarray(frames_hwc)
        if hwc.ndim != 4:
            raise ValueError(f"Expected (T,H,W,C) frames, got {hwc.shape}")
        t_full = int(hwc.shape[0])
        cap = self.max_frames
        if cap is not None and cap > 0 and t_full > cap:
            idx = np.linspace(0, t_full - 1, cap, dtype=np.int64)
            small = hwc[idx]
            prog = self._m.compute_progress(small, task_description=task_text)
            pred_sub = np.asarray(prog, dtype=np.float64).ravel()
            if pred_sub.size != small.shape[0]:
                logger.warning(
                    "RBM progress length %d != subsampled T %d; trimming",
                    pred_sub.size,
                    small.shape[0],
                )
                m = min(pred_sub.size, small.shape[0])
                pred_sub = pred_sub[:m]
                small = small[:m]
                idx = idx[:m]
            x_old = np.linspace(0.0, 1.0, num=pred_sub.size)
            x_new = np.linspace(0.0, 1.0, num=t_full)
            pred = np.interp(x_new, x_old, pred_sub).astype(np.float64)
            extra: dict[str, Any] = {
                "backend": self.name,
                "frame_subsample_cap": cap,
                "original_T": t_full,
                "interpolated_to_dense": True,
            }
            return pred, extra

        prog = self._m.compute_progress(hwc, task_description=task_text)
        pred = np.asarray(prog, dtype=np.float64).ravel()
        return pred, {"backend": self.name}


class SuccessDetectorBackend(RewardBackend):
    """Loads SVD latents (.pt per camera) and SuccessPredictor checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        latent_root: str,
        *,
        latent_root_expert: str = "",
        latent_root_rollout: str = "",
        device: str | None = None,
    ):
        sd_root = Path(__file__).resolve().parent.parent / "success_detector"
        if str(sd_root) not in sys.path:
            sys.path.insert(0, str(sd_root))
        from success_model import SuccessPredictor  # type: ignore

        self.name = "success_detector"
        self._latent_default = Path(latent_root).expanduser() if latent_root else None
        self._latent_expert = Path(latent_root_expert).expanduser() if latent_root_expert else None
        self._latent_rollout = Path(latent_root_rollout).expanduser() if latent_root_rollout else None
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        if not isinstance(args, dict):
            nc = int(getattr(args, "num_cameras", 3))
            ed = int(getattr(args, "encoder_dim", 128))
        else:
            nc = int(args.get("num_cameras", 3))
            ed = int(args.get("encoder_dim", 128))
        self.model = SuccessPredictor(num_cameras=nc, encoder_dim=ed, dropout=0.2).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(
            "SuccessDetector loaded from %s (latent default=%s expert=%s rollout=%s)",
            checkpoint_path,
            self._latent_default,
            self._latent_expert,
            self._latent_rollout,
        )

    def _latent_tree_root(self, dataset_name: str) -> Path | None:
        """extract_latent_robomimic --output_path; manifest dataset_name is expert_ph | rollout_mixed."""
        if dataset_name == "expert_ph" and self._latent_expert is not None:
            return self._latent_expert
        if dataset_name == "rollout_mixed" and self._latent_rollout is not None:
            return self._latent_rollout
        return self._latent_default

    def _find_latent_dir(self, traj_id: int, dataset_name: str) -> Path | None:
        root = self._latent_tree_root(dataset_name)
        if root is None:
            return None
        lv = root / "latent_videos"
        if not lv.is_dir():
            return None
        for split_dir in sorted(lv.iterdir()):
            if not split_dir.is_dir():
                continue
            cand = split_dir / str(traj_id)
            if cand.is_dir() and any(cand.glob("*.pt")):
                return cand
        return None

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        traj_id = int(meta.get("demo_id", -1))
        dataset_name = str(meta.get("dataset_name", "") or "")
        latent_dir = self._find_latent_dir(traj_id, dataset_name)
        root_used = self._latent_tree_root(dataset_name)
        if latent_dir is None:
            raise FileNotFoundError(
                f"No latent dir for dataset_name={dataset_name!r} demo_id={traj_id} under "
                f"{root_used}/latent_videos/*/{traj_id} — run extract_latent_robomimic.py on the same HDF5 as preprocess, "
                f"or set --latent-root-expert / --latent-root-rollout."
            )
        # Load camera tensors [T, C, H, W] sorted by 0.pt, 1.pt, 2.pt
        pts = sorted(latent_dir.glob("*.pt"), key=lambda p: p.stem)
        if len(pts) < 1:
            raise FileNotFoundError(f"No .pt in {latent_dir}")
        cams = [torch.load(p, map_location="cpu", weights_only=True) for p in pts]
        t_per_cam = [int(x.shape[0]) for x in cams]
        t_lat = min(t_per_cam)
        t_frm = int(frames_hwc.shape[0])
        if len(set(t_per_cam)) > 1:
            logger.warning(
                "Latent length differs across cameras demo_id=%d: %s — using min T=%d",
                traj_id,
                t_per_cam,
                t_lat,
            )

        # preprocess_manifest aligns trajectories to min(len(rewards), len(rgb)); extract_latent_robomimic
        # encodes obs/*_image[::rgb_skip] without that trim. When rewards are shorter than RGB, latent T can
        # exceed manifest frame T — truncate latents to the manifest (evaluation) horizon.
        if t_lat > t_frm:
            logger.info(
                "Latent T=%d > manifest frame T=%d demo_id=%d — truncating latents to match preprocess "
                "(full RGB stack vs min(rewards,rgb) in manifest)",
                t_lat,
                t_frm,
                traj_id,
            )
            cams = [c[:t_frm].contiguous() for c in cams]
            t_lat = min(int(c.shape[0]) for c in cams)
        t_use = t_lat
        if t_lat < t_frm:
            logger.warning(
                "Latent T=%d < manifest frame T=%d demo_id=%d — scoring %d steps; run_reward_inference will "
                "interpolate preds to GT length (check rgb_skip / HDF5 match for extract_latent_robomimic)",
                t_lat,
                t_frm,
                traj_id,
                t_use,
            )
        probs: list[float] = []
        warn: dict[str, Any] = {
            "latent_T_before_align": t_per_cam,
            "latent_T_used": t_use,
            "frame_T": t_frm,
        }
        with torch.no_grad():
            for t in range(t_use):
                views = []
                for cam in cams:
                    x = cam[t : t + 1].to(self.device, dtype=torch.float32)
                    views.append(x)
                logits = self.model(views)
                p = torch.sigmoid(logits).item()
                probs.append(float(p))
        pred = np.asarray(probs, dtype=np.float64)
        warn["latent_dir"] = str(latent_dir)
        return pred, warn


def build_backend(kind: str, **kwargs: Any) -> RewardBackend:
    k = kind.lower().strip()
    if k in ("topreward", "topreward_qwen"):
        return TopRewardBackend(
            model_path=kwargs["model_path"],
            max_frames=int(kwargs.get("max_frames", 64)),
            num_prefix_samples=int(kwargs.get("num_prefix_samples", 15)),
        )
    if k == "robodopamine":
        return RoboDopamineBackend(
            model_path=kwargs["model_path"],
            frame_interval=int(kwargs.get("frame_interval", 1)),
            eval_mode=str(kwargs.get("eval_mode", "incremental")),
        )
    if k == "rbm":
        mf = kwargs.get("max_frames")
        cap: int | None
        if mf is None:
            cap = 48
        else:
            cap = int(mf) if int(mf) > 0 else None
        return RBMFamilyBackend(checkpoint_path=kwargs["checkpoint"], name="rbm", max_frames=cap)
    if k == "roboreward":
        return RoboRewardBackend(
            model_path=kwargs["model_path"],
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            use_unsloth=bool(kwargs.get("use_unsloth", True)),
        )
    if k in ("success_detector", "success"):
        return SuccessDetectorBackend(
            checkpoint_path=kwargs["checkpoint"],
            latent_root=str(kwargs.get("latent_root", "") or ""),
            latent_root_expert=str(kwargs.get("latent_root_expert", "") or ""),
            latent_root_rollout=str(kwargs.get("latent_root_rollout", "") or ""),
            device=kwargs.get("device"),
        )
    raise ValueError(f"Unknown backend kind: {kind}")
