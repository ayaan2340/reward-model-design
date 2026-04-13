import argparse
import logging
import os
import sys
import time
import traceback

import h5py
import numpy as np
import torch

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.utils.file_utils import create_hdf5_filter_key, load_dict_from_checkpoint

log = logging.getLogger(__name__)

def ensure_hdf5_train_val_masks(dataset_path, val_ratio=0.1, seed=1):
    path = os.path.expanduser(dataset_path)

    def read_masks():
        with h5py.File(path, "r") as f:
            if "mask" not in f or "train" not in f["mask"] or "valid" not in f["mask"]:
                return None, None
            tr = [x.decode("utf-8") for x in np.array(f["mask/train"][:])]
            va = [x.decode("utf-8") for x in np.array(f["mask/valid"][:])]
            return tr, va

    train_demos, valid_demos = read_masks()
    if (
        train_demos is not None
        and len(train_demos) > 0
        and len(valid_demos) > 0
        and set(train_demos).isdisjoint(set(valid_demos))
    ):
        return True

    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    n = len(demos)
    if n < 2:
        log.warning(
            "Dataset has fewer than 2 demos; disabling validation (train on all trajectories)."
        )
        return False

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, min(int(round(n * val_ratio)), n - 1))
    valid_keys = [demos[perm[i]] for i in range(n_val)]
    train_keys = [demos[perm[i]] for i in range(n_val, n)]
    create_hdf5_filter_key(path, train_keys, "train")
    create_hdf5_filter_key(path, valid_keys, "valid")
    log.info(
        "Created mask/train (%d demos) and mask/valid (%d demos) in %s",
        len(train_keys),
        len(valid_keys),
        path,
    )
    return True


def install_epoch_logging():
    _orig = TrainUtils.run_epoch

    def _wrapped(*args, **kwargs):
        out = _orig(*args, **kwargs)
        epoch = kwargs.get("epoch")
        is_val = kwargs.get("validate", False)
        prefix = "val" if is_val else "train"

        loss = out.get("Loss")
        ll = out.get("Log_Likelihood")
        lr = out.get("Optimizer/policy0_lr")
        grad = out.get("Policy_Grad_Norms")

        parts = [f"[epoch {epoch:4d}] {prefix:>5s}"]
        if loss is not None:
            parts.append(f"loss={loss:+.4f}")
        if ll is not None:
            parts.append(f"ll={ll:+.4f}")
        if not is_val:
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            if grad is not None:
                parts.append(f"grad={grad:.1f}")
        t_epoch = out.get("Time_Epoch")
        if t_epoch is not None:
            parts.append(f"t={t_epoch:.2f}m")
        log.info("  ".join(parts))
        return out

    TrainUtils.run_epoch = _wrapped


def print_training_summary(output_dir, experiment_name):
    base = os.path.join(os.path.expanduser(output_dir), experiment_name)
    if not os.path.isdir(base):
        return
    subdirs = sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )
    if not subdirs:
        return
    run_dir = os.path.join(base, subdirs[-1])
    last_pth = os.path.join(run_dir, "last.pth")
    if not os.path.isfile(last_pth):
        return
    ckpt = load_dict_from_checkpoint(last_pth)
    vs = ckpt.get("variable_state") or {}

    log.info("")
    log.info("=" * 60)
    log.info("Training summary (from %s)", last_pth)
    log.info("=" * 60)
    bvl = vs.get("best_valid_loss")
    log.info("  best_valid_loss : %s", f"{bvl:.6f}" if bvl is not None else "N/A")
    bsr = vs.get("best_success_rate")
    if isinstance(bsr, dict) and bsr:
        printable = {k: v for k, v in bsr.items() if v is not None and v > -0.5}
        log.info("  best_success_rate: %s", printable if printable else "N/A (no rollout improvement)")
    else:
        log.info("  best_success_rate: N/A (rollouts disabled)")
    last_epoch = vs.get("epoch")
    if last_epoch is not None:
        log.info("  last_epoch      : %d", last_epoch)
    log.info("=" * 60)


def build_config(dataset_path, output_dir, experiment_name, use_validation=True, seed=1):
    config = config_factory("bc")

    with config.values_unlocked():
        config.experiment.name = experiment_name
        config.experiment.validate = use_validation
        config.experiment.save.on_best_validation = True
        config.experiment.save.every_n_epochs = 20
        config.experiment.rollout.enabled = False
        config.experiment.logging.log_wandb = True
        config.experiment.logging.wandb_proj_name = "ctrl-world-bc-5"

        config.experiment.env_meta_update_dict = {"env_kwargs": {"control_freq": 5}}

        config.train.data = [{"path": dataset_path}]
        config.train.output_dir = output_dir
        config.train.seed = seed
        config.train.num_data_workers = 2
        config.train.num_epochs = 400
        config.train.max_grad_norm = 1.0
        config.train.frame_stack = 5
        config.train.seq_length = 5
        config.train.dataset_keys = ["actions"]
        config.train.action_config = {"actions": {"normalization": "min_max"}}
        if use_validation:
            config.train.hdf5_filter_key = "train"
            config.train.hdf5_validation_filter_key = "valid"

        config.algo.optim_params.policy.optimizer_type = "adamw"
        config.algo.optim_params.policy.learning_rate.epoch_schedule = [200]
        config.algo.optim_params.policy.learning_rate.scheduler_type = "linear"
        config.algo.optim_params.policy.regularization.L2 = 0.01

        config.algo.actor_layer_dims = []
        config.algo.gmm.enabled = True
        config.algo.transformer.enabled = True
        config.algo.transformer.context_length = 5
        config.algo.transformer.supervise_all_steps = True
        config.algo.transformer.pred_future_acs = True
        config.algo.transformer.use_action_queue = True

        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        config.observation.modalities.obs.rgb = [
            "agentview_image",
            "frontview_image",
            "robot0_eye_in_hand_image",
        ]
        config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"

    config.lock()
    return config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Train BC-Transformer-GMM on 5Hz accumulated-delta robosuite data")
    parser.add_argument("--dataset", type=str, required=True, help="path to 5Hz HDF5 with accumulated delta actions")
    parser.add_argument("--output_dir", type=str, default="/projects/bggq/asunesara/bc_transformer_trained_models",
                        help="output directory for checkpoints and logs")
    parser.add_argument("--name", type=str, default="square_bc_transformer_gmm",
                        help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="random seed (data split + training)")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="fraction of demos for validation when creating mask/valid (if missing)",
    )
    parser.add_argument("--debug", action="store_true", help="short training run for debugging")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    args = parser.parse_args()

    use_val = ensure_hdf5_train_val_masks(
        args.dataset, val_ratio=args.val_ratio, seed=args.seed
    )
    config = build_config(
        args.dataset,
        args.output_dir,
        args.name,
        use_validation=use_val,
        seed=args.seed,
    )

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.train.output_dir = "/tmp/bc_transformer_debug"

    config.lock()

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    t_start = time.time()
    res_str = "finished run successfully!"
    try:
        install_epoch_logging()
        train(config, device=device, resume=args.resume)
        print_training_summary(config.train.output_dir, config.experiment.name)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    elapsed = time.time() - t_start
    log.info("Total wall time: %.1f min (%.1f h)", elapsed / 60, elapsed / 3600)
    print(res_str)


if __name__ == "__main__":
    main()
