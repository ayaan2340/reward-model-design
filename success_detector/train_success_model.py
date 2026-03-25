import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from robomimic_dataset import RobomimicLatentDataset
from success_model import SuccessPredictor

logger = logging.getLogger("train_success")



class BalancedBCEWithLogitsLoss(nn.Module):
    # pos_weight: weights success frames more heavily to handle less success frames
    # fp_weight:  weights failure errors to penalize false positives

    def __init__(self, pos_weight: float = 1.0, fp_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.fp_weight = fp_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.pos_weight),
            torch.full_like(targets, self.fp_weight),
        )
        return (weights * bce).mean()


def _stratified_traj_split(dataset: RobomimicLatentDataset,
                           train_frac: float, val_frac: float,
                           seed: int):
    rng = np.random.RandomState(seed)

    pos_trajs = []
    neg_trajs = []
    for i, anno in enumerate(dataset.annotations):
        (pos_trajs if anno["success"] else neg_trajs).append(i)

    def _split_list(lst):
        perm = rng.permutation(len(lst))
        n_tr = int(len(lst) * train_frac)
        n_va = int(len(lst) * val_frac)
        return (
            [lst[j] for j in perm[:n_tr]],
            [lst[j] for j in perm[n_tr:n_tr + n_va]],
            [lst[j] for j in perm[n_tr + n_va:]],
        )

    pos_tr, pos_va, pos_te = _split_list(pos_trajs)
    neg_tr, neg_va, neg_te = _split_list(neg_trajs)

    split_sets = {
        "train": set(pos_tr + neg_tr),
        "val": set(pos_va + neg_va),
        "test": set(pos_te + neg_te),
    }

    indices = {k: [] for k in split_sets}
    for frame_idx, (traj_idx, _) in enumerate(dataset._index):
        for k, traj_set in split_sets.items():
            if traj_idx in traj_set:
                indices[k].append(frame_idx)
                break

    return indices


def _balance_indices(
    dataset: RobomimicLatentDataset,
    indices: list,
    max_neg_ratio: int,
    seed: int,
    split: str = "",
) -> list:
    rng = np.random.RandomState(seed)

    last_success_frames = {
        (traj_idx, anno["video_length"] - 1)
        for traj_idx, anno in enumerate(dataset.annotations)
        if anno["success"]
    }

    pos_indices, neg_indices = [], []
    for flat_idx in indices:
        traj_idx, frame_idx = dataset._index[flat_idx]
        label = dataset.frame_label(traj_idx, frame_idx)
        if label > 0.5 or (traj_idx, frame_idx) in last_success_frames:
            pos_indices.append(flat_idx)
        else:
            neg_indices.append(flat_idx)

    max_neg = len(pos_indices) * max_neg_ratio
    if len(neg_indices) > max_neg:
        neg_indices = rng.choice(neg_indices, size=max_neg, replace=False).tolist()

    label = f"  {split} balance" if split else "  balance"
    logger.info(
        "%s: kept %d pos, sampled %d/%d neg (ratio %.1f:1)",
        label, len(pos_indices), len(neg_indices),
        len(pos_indices) * max_neg_ratio + len(neg_indices),
        len(neg_indices) / max(len(pos_indices), 1),
    )
    return sorted(pos_indices + neg_indices)


def _label_stats_fast(dataset) -> dict:
    pos = neg = 0
    if isinstance(dataset, Subset):
        base = dataset.dataset
        for idx in dataset.indices:
            traj_idx, frame_idx = base._index[idx]
            if base.frame_label(traj_idx, frame_idx) > 0.5:
                pos += 1
            else:
                neg += 1
    elif isinstance(dataset, RobomimicLatentDataset):
        for traj_idx, frame_idx in dataset._index:
            if dataset.frame_label(traj_idx, frame_idx) > 0.5:
                pos += 1
            else:
                neg += 1
    return {"pos": pos, "neg": neg, "total": pos + neg}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        views = [v.to(device) for v in batch["latent_views"]]
        labels = batch["success"].to(device).unsqueeze(1)

        logits = model(views)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    fp_succ_dists: list[float] = []
    fp_fail_fracs: list[float] = []

    for batch in loader:
        views = [v.to(device) for v in batch["latent_views"]]
        labels = batch["success"].to(device).unsqueeze(1)

        logits = model(views)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (logits > 0).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

        fp_mask = (preds.squeeze(1).cpu() == 1) & (labels.squeeze(1).cpu() == 0)
        for i in fp_mask.nonzero(as_tuple=False).squeeze(1).tolist():
            t = batch["timestep"][i].item()
            n = batch["num_frames"][i].item()
            if batch["traj_success"][i].item() > 0.5:
                fp_succ_dists.append(float((n - 1) - t))
            else:
                fp_fail_fracs.append(float(t) / max(n - 1, 1))

    total = tp + fp + fn + tn
    n_pos = tp + fn  
    n_neg = fp + tn  
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / n_pos if n_pos > 0 else 0.0   
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall                                    
    fpr = fp / n_neg if n_neg > 0 else 0.0         
    fnr = fn / n_pos if n_pos > 0 else 0.0         
    tnr = tn / n_neg if n_neg > 0 else 0.0         

    return {
        "loss": total_loss / total,
        "acc": (tp + tn) / total,
        "success_acc": tpr,   
        "failure_acc": tnr,   
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr, "fpr": fpr, "fnr": fnr, "tnr": tnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fp_succ_n": len(fp_succ_dists),
        "fp_succ_avg_dist": float(np.mean(fp_succ_dists)) if fp_succ_dists else float("nan"),
        "fp_fail_n": len(fp_fail_fracs),
        "fp_fail_avg_frac": float(np.mean(fp_fail_fracs)) if fp_fail_fracs else float("nan"),
    }


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

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
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

    split_idx = _stratified_traj_split(dataset, args.train_frac, args.val_frac, args.seed)
    train_ds = Subset(dataset, _balance_indices(
        dataset, split_idx["train"], args.max_neg_ratio, args.seed, split="train"
    ))
    val_ds = Subset(dataset, _balance_indices(
        dataset, split_idx["val"], args.max_neg_ratio, args.seed, split="val"
    ))
    test_ds = Subset(dataset, _balance_indices(
        dataset, split_idx["test"], args.max_neg_ratio, args.seed, split="test"
    ))

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        stats = _label_stats_fast(ds)
        logger.info("  %s: %d frames (pos=%d, neg=%d, %.1f%% pos)",
                     name, stats["total"], stats["pos"], stats["neg"],
                     100 * stats["pos"] / max(stats["total"], 1))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    train_stats = _label_stats_fast(train_ds)
    pos_weight = train_stats["neg"] / train_stats["pos"] if train_stats["pos"] > 0 else 1.0
    logger.info("  pos_weight=%.2f  fp_weight=%.2f", pos_weight, args.fp_weight)

    model = SuccessPredictor(
        num_cameras=args.num_cameras, encoder_dim=args.encoder_dim, dropout=0.2
    ).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    criterion = BalancedBCEWithLogitsLoss(pos_weight=pos_weight, fp_weight=args.fp_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "epoch %3d/%d  %.1fs  lr=%.1e  "
            "train_loss=%.4f train_acc=%.3f  "
            "val_loss=%.4f val_acc=%.3f val_f1=%.3f  "
            "val_succ_acc=%.3f val_fail_acc=%.3f  "
            "TPR=%.3f FPR=%.3f FNR=%.3f TNR=%.3f  "
            "[TP=%d FP=%d FN=%d TN=%d]  "
            "FP@succ_traj: n=%d avg_dist_to_end=%.1f frames  "
            "FP@fail_traj: n=%d avg_pos=%.2f (frac of traj)",
            epoch, args.epochs, elapsed, args.lr,
            train_m["loss"], train_m["acc"],
            val_m["loss"], val_m["acc"], val_m["f1"],
            val_m["success_acc"], val_m["failure_acc"],
            val_m["tpr"], val_m["fpr"], val_m["fnr"], val_m["tnr"],
            val_m["tp"], val_m["fp"], val_m["fn"], val_m["tn"],
            val_m["fp_succ_n"], val_m["fp_succ_avg_dist"],
            val_m["fp_fail_n"], val_m["fp_fail_avg_frac"],
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

    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Test  loss=%.4f  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  "
        "succ_acc=%.3f  fail_acc=%.3f  "
        "TPR=%.3f FPR=%.3f FNR=%.3f TNR=%.3f  "
        "[TP=%d FP=%d FN=%d TN=%d]  "
        "FP@succ_traj: n=%d avg_dist_to_end=%.1f frames  "
        "FP@fail_traj: n=%d avg_pos=%.2f (frac of traj)",
        test_m["loss"], test_m["acc"], test_m["precision"], test_m["recall"], test_m["f1"],
        test_m["success_acc"], test_m["failure_acc"],
        test_m["tpr"], test_m["fpr"], test_m["fnr"], test_m["tnr"],
        test_m["tp"], test_m["fp"], test_m["fn"], test_m["tn"],
        test_m["fp_succ_n"], test_m["fp_succ_avg_dist"],
        test_m["fp_fail_n"], test_m["fp_fail_avg_frac"],
    )

    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": ckpt["val_metrics"],
        "test_metrics": test_m,
        "args": vars(args),
    }, os.path.join(args.output_dir, "final.pt"))
    logger.info("Saved final checkpoint to %s", args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Train SuccessPredictor on SVD-VAE latents")

    p.add_argument("--latent_root", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True,
                   help="Original HDF5 for corrected success labels and dones/rewards")

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--rgb_skip", type=int, default=1)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--num_cameras", type=int, default=3)
    p.add_argument("--encoder_dim", type=int, default=128)

    p.add_argument("--output_dir", type=str, default="runs/success_detector")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--preload", action="store_true",
                   help="Preload all latent tensors into RAM")
    p.add_argument("--max_neg_ratio", type=int, default=10,
                   help="Max ratio of failure to success frames kept in training set")
    p.add_argument("--fp_weight", type=float, default=2.0,
                   help="Extra loss weight on failure-class samples to penalize false positives")

    return p.parse_args()


if __name__ == "__main__":
    main()
