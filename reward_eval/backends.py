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


def _roboreward_discrete_score_from_normalized(pred_broadcast: np.ndarray) -> int:
    """Invert RoboReward normalization ``pred = score/4 - 0.25`` to integer 1..5."""
    pv = float(np.asarray(pred_broadcast, dtype=np.float64).ravel()[0])
    disc = int(np.clip(np.round((pv + 0.25) * 4.0), 1, 5))
    return disc


def _first_positive_frame_1d(
    mask: np.ndarray,
) -> float:
    """First index where boolean mask is True; NaN if none."""
    m = np.asarray(mask, dtype=bool).ravel()
    hit = np.flatnonzero(m)
    return float(hit[0]) if hit.size else float("nan")


def _topreward_sliding_window_success(
    progress_dense: np.ndarray, *, threshold: float, window: int = 3
) -> tuple[float, np.ndarray, float]:
    """TOPReward: success if any length-``window`` mean of min–max normalized prefix scores exceeds ``threshold``.

    Returns:
        traj_success, rolling-start window means (NaN padded), first window **start** index (pred timeline)
        where the mean crosses ``threshold``, else NaN.
    """
    p = np.asarray(progress_dense, dtype=np.float64).ravel()
    t = int(p.size)
    if t == 0:
        return 0.0, p, float("nan")
    if t < window:
        m = float(np.mean(p))
        dense = np.full(t, m, dtype=np.float64)
        ok = m > threshold
        first_start = 0.0 if ok else float("nan")
        return ((1.0 if ok else 0.0), dense, first_start)
    dense = np.full(t, np.nan, dtype=np.float64)
    best = 0.0
    first_start: float | None = None
    for i in range(t - window + 1):
        wmean = float(p[i : i + window].mean())
        dense[i] = wmean
        best = max(best, wmean)
        if first_start is None and wmean > threshold:
            first_start = float(i)
    fs = float(first_start) if first_start is not None else float("nan")
    return (1.0 if best > threshold else 0.0), dense, fs


def _interp_01(y: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    if n_from == n_to:
        return y.copy()
    if n_from == 0:
        return np.zeros(n_to, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, num=n_from)
    x_new = np.linspace(0.0, 1.0, num=n_to)
    return np.interp(x_new, x_old, y).astype(np.float64)


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
        self._success_threshold = float(kwargs.pop("success_threshold", 0.95))
        # Defaults match ``--rbm-max-frames`` / dense per-frame scoring (see run_reward_inference).
        self._num_prefix_samples = int(kwargs.get("num_prefix_samples", 0))
        self._max_frames = int(kwargs.get("max_frames", 48))
        self._m = TopReward(model_path=model_path, **kwargs)
        logger.info(
            "TopReward loaded: %s (success_threshold=%s max_frames=%s num_prefix_samples=%s)",
            model_path,
            self._success_threshold,
            self._max_frames,
            self._num_prefix_samples,
        )

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = self._m.compute_progress(frames_hwc, task_description=task_text)
        pred = np.asarray(out, dtype=np.float64).ravel()
        traj_s, succ_dense, first_win_start = _topreward_sliding_window_success(
            pred, threshold=self._success_threshold, window=3
        )
        t_native = int(pred.size)
        timing_native = float(t_native - 1) if traj_s >= 0.5 and t_native > 0 else float("nan")
        return pred, {
            "backend": self.name,
            "topreward_max_frames": self._max_frames,
            "topreward_num_prefix_samples": self._num_prefix_samples,
            "topreward_dense_prefixes": self._num_prefix_samples <= 0,
            "success_metric": "topreward_sliding_mean3",
            "topreward_success_threshold": self._success_threshold,
            "traj_success_pred": float(traj_s),
            "pred_success_dense": succ_dense,
            # Frame indices on the same timeline as ``pred`` (before manifest alignment); map in run_reward_inference.
            "success_pred_first_idx_native": float(first_win_start),
            "success_pred_timing_idx_native": timing_native,
            "success_pred_native_len": int(t_native),
        }


class RoboDopamineBackend(RewardBackend):
    def __init__(self, model_path: str, **kwargs: Any):
        _ensure_robometer()
        from robometer.evals.baselines.robodopamine import RoboDopamine

        self.name = "robodopamine"
        eg = kwargs.pop("expert_goal_frames_npz", None) or ""
        self._expert_goal_rgb: np.ndarray | None = None
        self._expert_goal_source: str = ""
        if eg:
            p = Path(str(eg)).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"RoboDopamine expert goal frames_npz not found: {p}")
            z = np.load(p)
            rgb = z["rgb"]
            if rgb.ndim != 4 or int(rgb.shape[-1]) < 3:
                raise ValueError(f"Bad rgb in expert goal npz {p}: shape={getattr(rgb, 'shape', None)}")
            self._expert_goal_rgb = np.asarray(rgb[-1, ..., :3], dtype=np.uint8)
            self._expert_goal_source = str(p)
            logger.info("RoboDopamine REFERENCE END = last frame of %s", p)
        self._m = RoboDopamine(model_path=model_path, **kwargs)
        logger.info("RoboDopamine loaded: %s", model_path)

    def predict_dense(
        self,
        frames_hwc: np.ndarray,
        task_text: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        out = self._m.compute_progress(
            frames_hwc,
            task_description=task_text,
            reference_goal_rgb=self._expert_goal_rgb,
        )
        pred = np.asarray(out, dtype=np.float64).ravel()
        extra: dict[str, Any] = {
            "backend": self.name,
            "success_metric": "skipped",
            "traj_success_pred": float("nan"),
        }
        if self._expert_goal_source:
            extra["expert_goal_frames_npz"] = self._expert_goal_source
        return pred, extra


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
        if pred.size == 0:
            return pred, {
                "backend": self.name,
                "success_metric": "roboreward_max_bin5",
                "traj_success_pred": 0.0,
                "pred_success_dense": pred.astype(np.float64),
            }
        disc = _roboreward_discrete_score_from_normalized(pred)
        traj_s = 1.0 if disc >= 5 else 0.0
        succ_dense = np.full(pred.shape, traj_s, dtype=np.float64)
        return pred, {
            "backend": self.name,
            "prediction_mode": "end_of_episode_broadcast",
            "note": "RoboReward judges full episode once; per-frame values are constant (not dense progress).",
            "success_metric": "roboreward_max_bin5",
            "roboreward_discrete_score": disc,
            "traj_success_pred": traj_s,
            "pred_success_dense": succ_dense,
        }


class RBMFamilyBackend(RewardBackend):
    """RBM / ReWiND via ``RBMModel``. Progress in ``[0, 1]`` from head (Sigmoid or bin expectation) + clip."""

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
            prog, succ_sub = self._m.compute_progress_with_success(small, task_description=task_text)
            pred_sub = np.asarray(prog, dtype=np.float64).ravel()
            succ_sub = np.asarray(succ_sub, dtype=np.float64).ravel()
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
                if succ_sub.size:
                    succ_sub = succ_sub[:m]
            pred = np.clip(_interp_01(pred_sub, pred_sub.size, t_full), 0.0, 1.0)
            extra: dict[str, Any] = {
                "backend": self.name,
                "frame_subsample_cap": cap,
                "original_T": t_full,
                "interpolated_to_dense": True,
                "success_metric": "rbm_success_head_sigmoid",
            }
            if succ_sub.size:
                succ_full = np.clip(_interp_01(succ_sub, succ_sub.size, t_full), 0.0, 1.0)
                extra["traj_success_pred"] = float(np.any(succ_full > 0.5))
                extra["pred_success_dense"] = succ_full
                hit = np.flatnonzero(succ_full > 0.5)
                ff = float(hit[0]) if hit.size else float("nan")
                extra["success_pred_first_idx_native"] = ff
                extra["success_pred_timing_idx_native"] = ff
                extra["success_pred_native_len"] = int(t_full)
            else:
                extra["success_head_missing"] = True
                extra["traj_success_pred"] = float("nan")
            return pred, extra

        prog, succ_sub = self._m.compute_progress_with_success(hwc, task_description=task_text)
        pred = np.clip(np.asarray(prog, dtype=np.float64).ravel(), 0.0, 1.0)
        succ_sub = np.asarray(succ_sub, dtype=np.float64).ravel()
        extra = {"backend": self.name, "success_metric": "rbm_success_head_sigmoid"}
        if succ_sub.size:
            extra["traj_success_pred"] = float(np.any(succ_sub > 0.5))
            extra["pred_success_dense"] = np.clip(succ_sub, 0.0, 1.0)
            hit = np.flatnonzero(succ_sub > 0.5)
            ff = float(hit[0]) if hit.size else float("nan")
            extra["success_pred_first_idx_native"] = ff
            extra["success_pred_timing_idx_native"] = ff
            extra["success_pred_native_len"] = int(succ_sub.size)
        else:
            extra["success_head_missing"] = True
            extra["traj_success_pred"] = float("nan")
        return pred, extra


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
        logits_list: list[float] = []
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
                logit_t = self.model(views)
                lv = float(logit_t.reshape(-1)[0].item())
                logits_list.append(lv)
                p = float(torch.sigmoid(logit_t).reshape(-1)[0].item())
                probs.append(p)
        logits_arr = np.asarray(logits_list, dtype=np.float64)
        pred = np.asarray(probs, dtype=np.float64)
        mask = (logits_arr > 0.0) | (pred > 0.5)
        traj_success = float(np.any(mask))
        ff = _first_positive_frame_1d(mask)
        warn["latent_dir"] = str(latent_dir)
        warn["success_metric"] = "success_detector_logit_or_prob"
        warn["traj_success_pred"] = traj_success
        warn["pred_success_dense"] = pred
        warn["pred_success_logit"] = logits_arr
        warn["success_pred_first_idx_native"] = ff
        warn["success_pred_timing_idx_native"] = ff
        warn["success_pred_native_len"] = int(t_use)
        return pred, warn


# Backends that broadcast one scalar to every frame (no dense time-varying progress). Pearson vs per-frame
# GT in ``reward_eval.compute_metrics`` is skipped when preds are flat; RoboReward is flat by design.
FLAT_BROADCAST_BACKEND_KINDS: frozenset[str] = frozenset({"roboreward"})


def build_backend(kind: str, **kwargs: Any) -> RewardBackend:
    k = kind.lower().strip()
    if k in ("topreward", "topreward_qwen"):
        return TopRewardBackend(
            model_path=kwargs["model_path"],
            max_frames=int(kwargs.get("max_frames", 48)),
            num_prefix_samples=int(kwargs.get("num_prefix_samples", 0)),
            success_threshold=float(kwargs.get("success_threshold", 0.95)),
        )
    if k == "robodopamine":
        return RoboDopamineBackend(
            model_path=kwargs["model_path"],
            frame_interval=int(kwargs.get("frame_interval", 5)),
            eval_mode=str(kwargs.get("eval_mode", "incremental")),
            expert_goal_frames_npz=kwargs.get("expert_goal_frames_npz") or "",
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
