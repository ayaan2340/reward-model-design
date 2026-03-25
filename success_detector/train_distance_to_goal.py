import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from robomimic_dataset import RobomimicLatentDataset
from distance_to_goal_model import DistanceToGoalPredictor

logger = logging.getLogger("train_dtg")


# ------------------------------------------------------------------ #
#  Data splitting (success-only train/val, mixed test)
# ------------------------------------------------------------------ #

def _success_only_traj_split(dataset: RobomimicLatentDataset,
                             train_frac: float, val_frac: float,
                             seed: int):
    """Split successful trajectories into train/val/test.

    Failed trajectories are collected separately and only used for the
    success-detection evaluation at test time.
    """
    rng = np.random.RandomState(seed)

    pos_trajs = []
    neg_trajs = []
    for i, anno in enumerate(dataset.annotations):
        (pos_trajs if anno["success"] else neg_trajs).append(i)

    perm = rng.permutation(len(pos_trajs))
    n_tr = int(len(pos_trajs) * train_frac)
    n_va = int(len(pos_trajs) * val_frac)
    pos_tr = [pos_trajs[j] for j in perm[:n_tr]]
    pos_va = [pos_trajs[j] for j in perm[n_tr:n_tr + n_va]]
    pos_te = [pos_trajs[j] for j in perm[n_tr + n_va:]]

    traj_sets = {
        "train": set(pos_tr),
        "val": set(pos_va),
        "test_success": set(pos_te),
        "test_fail": set(neg_trajs),
    }

    indices: dict[str, list[int]] = {k: [] for k in traj_sets}
    for frame_idx, (traj_idx, _) in enumerate(dataset._index):
        for k, traj_set in traj_sets.items():
            if traj_idx in traj_set:
                indices[k].append(frame_idx)
                break

    traj_indices: dict[str, list[int]] = {
        "train": sorted(pos_tr),
        "val": sorted(pos_va),
        "test_success": sorted(pos_te),
        "test_fail": sorted(neg_trajs),
    }

    logger.info("  trajectories: %d success (%d tr / %d va / %d te), %d fail",
                len(pos_trajs), len(pos_tr), len(pos_va), len(pos_te),
                len(neg_trajs))

    return indices, traj_indices


# ------------------------------------------------------------------ #
#  Training
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_ae = 0.0
    total = 0

    for batch in loader:
        views = [v.to(device) for v in batch["latent_views"]]
        target = (batch["num_frames"].float() - batch["timestep"].float() - 1).to(device).unsqueeze(1)

        pred = model(views)
        loss = F.smooth_l1_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = target.size(0)
        total_loss += loss.item() * n
        total_ae += (pred - target).abs().sum().item()
        total += n

    return {"loss": total_loss / total, "mae": total_ae / total}


# ------------------------------------------------------------------ #
#  Regression evaluation (success trajectories only)
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_ae = 0.0
    total_se = 0.0
    total = 0

    ae_first = []
    ae_mid = []
    ae_last = []

    for batch in loader:
        views = [v.to(device) for v in batch["latent_views"]]
        target = (batch["num_frames"].float() - batch["timestep"].float() - 1).to(device).unsqueeze(1)

        pred = model(views)
        loss = F.smooth_l1_loss(pred, target)
        errors = (pred - target).abs()

        n = target.size(0)
        total_loss += loss.item() * n
        total_ae += errors.sum().item()
        total_se += ((pred - target) ** 2).sum().item()
        total += n

        timesteps = batch["timestep"].float()
        num_frames = batch["num_frames"].float()
        frac = timesteps / (num_frames - 1).clamp(min=1)

        for i in range(n):
            err = errors[i].item()
            f = frac[i].item()
            if f < 0.1:
                ae_first.append(err)
            elif f > 0.9:
                ae_last.append(err)
            elif 0.4 <= f <= 0.6:
                ae_mid.append(err)

    return {
        "loss": total_loss / total,
        "mae": total_ae / total,
        "rmse": math.sqrt(total_se / total),
        "mae_first10": float(np.mean(ae_first)) if ae_first else float("nan"),
        "mae_mid": float(np.mean(ae_mid)) if ae_mid else float("nan"),
        "mae_last10": float(np.mean(ae_last)) if ae_last else float("nan"),
        "n": total,
    }


# ------------------------------------------------------------------ #
#  Success-detection evaluation (trajectory-level, all trajectories)
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_success_detection(model, dataset, traj_indices_success,
                               traj_indices_fail, thresholds, device):
    """Run the model on the *last frame* of every test trajectory and
    classify success/failure by thresholding the predicted distance."""
    model.eval()

    records: list[dict] = []

    for traj_idx in traj_indices_success + traj_indices_fail:
        traj = dataset.get_trajectory(traj_idx)
        anno = dataset.annotations[traj_idx]
        T = anno["video_length"]
        is_success = bool(anno["success"])

        last_frame_idx = T - 1
        last_views = [cam[last_frame_idx].unsqueeze(0).to(device)
                      for cam in traj["latent_views"]]
        pred_last = model(last_views).item()

        first_views = [cam[0].unsqueeze(0).to(device)
                       for cam in traj["latent_views"]]
        pred_first = model(first_views).item()

        all_preds = []
        for f in range(T):
            frame_views = [cam[f].unsqueeze(0).to(device)
                           for cam in traj["latent_views"]]
            all_preds.append(model(frame_views).item())

        records.append({
            "traj_idx": traj_idx,
            "traj_id": anno["episode_id"],
            "is_success": is_success,
            "num_frames": T,
            "pred_last": pred_last,
            "pred_first": pred_first,
            "pred_mean": float(np.mean(all_preds)),
            "pred_std": float(np.std(all_preds)),
            "all_preds": all_preds,
        })

    results_per_threshold: dict[float, dict] = {}
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for r in records:
            predicted_success = r["pred_last"] < thr
            if r["is_success"] and predicted_success:
                tp += 1
            elif not r["is_success"] and predicted_success:
                fp += 1
            elif r["is_success"] and not predicted_success:
                fn += 1
            else:
                tn += 1
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results_per_threshold[thr] = {
            "acc": (tp + tn) / total if total > 0 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    return records, results_per_threshold


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log"), mode="w"),
        ],
    )
    logger.info("Args: %s", vars(args))

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading dataset from %s", args.latent_root)
    dataset = RobomimicLatentDataset(
        latent_root=args.latent_root,
        hdf5_path=args.hdf5_path,
        split="train",
        rgb_skip=args.rgb_skip,
        preload=args.preload,
    )
    logger.info("  %s", dataset)

    frame_indices, traj_indices = _success_only_traj_split(
        dataset, args.train_frac, args.val_frac, args.seed
    )

    train_ds = Subset(dataset, frame_indices["train"])
    val_ds = Subset(dataset, frame_indices["val"])
    test_succ_ds = Subset(dataset, frame_indices["test_success"])

    for name, ds in [("train", train_ds), ("val", val_ds),
                     ("test_success", test_succ_ds)]:
        logger.info("  %s: %d frames", name, len(ds))
    logger.info("  test_fail: %d frames (held out for detection eval)",
                len(frame_indices["test_fail"]))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_succ_loader = DataLoader(test_succ_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True)

    model = DistanceToGoalPredictor(
        num_cameras=args.num_cameras, encoder_dim=args.encoder_dim, dropout=0.2,
    ).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device)
        val_m = evaluate_regression(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            "epoch %3d/%d  %.1fs  lr=%.1e  "
            "train_loss=%.4f train_mae=%.2f  "
            "val_loss=%.4f val_mae=%.2f val_rmse=%.2f  "
            "val_mae[first10%%]=%.2f val_mae[mid]=%.2f val_mae[last10%%]=%.2f",
            epoch, args.epochs, elapsed, args.lr,
            train_m["loss"], train_m["mae"],
            val_m["loss"], val_m["mae"], val_m["rmse"],
            val_m["mae_first10"], val_m["mae_mid"], val_m["mae_last10"],
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }, os.path.join(args.output_dir, "best.pt"))

    logger.info("Best val_loss=%.4f at epoch %d", best_val_loss, best_epoch)

    # ---- Load best model for final evaluation ---- #
    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---- Steps-to-go regression on held-out successful trajectories ---- #
    test_reg = evaluate_regression(model, test_succ_loader, device)
    logger.info(
        "Test (steps-to-go)  loss=%.4f  mae=%.2f  rmse=%.2f  "
        "mae[first10%%]=%.2f  mae[mid]=%.2f  mae[last10%%]=%.2f  n=%d",
        test_reg["loss"], test_reg["mae"], test_reg["rmse"],
        test_reg["mae_first10"], test_reg["mae_mid"], test_reg["mae_last10"],
        test_reg["n"],
    )

    # ---- Success detection via last-frame thresholding ---- #
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, args.success_threshold]
    thresholds = sorted(set(thresholds))

    records, det_results = evaluate_success_detection(
        model, dataset,
        traj_indices["test_success"],
        traj_indices["test_fail"],
        thresholds, device,
    )

    logger.info("--- Success detection (threshold sweep) ---")
    for thr in thresholds:
        m = det_results[thr]
        logger.info(
            "  thr=%.1f  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  "
            "[TP=%d FP=%d FN=%d TN=%d]",
            thr, m["acc"], m["precision"], m["recall"], m["f1"],
            m["tp"], m["fp"], m["fn"], m["tn"],
        )

    logger.info("--- Per-trajectory predictions (last frame) ---")
    for r in records:
        tag = "SUCCESS" if r["is_success"] else "FAIL   "
        logger.info(
            "  [%s] traj=%s  T=%d  pred_last=%.2f  pred_first=%.2f  "
            "pred_mean=%.2f  pred_std=%.2f",
            tag, r["traj_id"], r["num_frames"],
            r["pred_last"], r["pred_first"],
            r["pred_mean"], r["pred_std"],
        )

    # ---- Save final checkpoint ---- #
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": ckpt["val_metrics"],
        "test_regression": test_reg,
        "test_detection": {str(k): v for k, v in det_results.items()},
        "args": vars(args),
    }, os.path.join(args.output_dir, "final.pt"))
    logger.info("Saved final checkpoint to %s", args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DistanceToGoalPredictor on SVD-VAE latents")

    p.add_argument("--latent_root", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True,
                   help="Original HDF5 for corrected success labels")

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--rgb_skip", type=int, default=1)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--num_cameras", type=int, default=3)
    p.add_argument("--encoder_dim", type=int, default=128)

    p.add_argument("--output_dir", type=str, default="runs/distance_to_goal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--preload", action="store_true",
                   help="Preload all latent tensors into RAM")
    p.add_argument("--success_threshold", type=float, default=2.0,
                   help="Predicted-distance threshold below which a frame "
                        "is classified as 'success' for detection evaluation")

    return p.parse_args()


if __name__ == "__main__":
    main()
