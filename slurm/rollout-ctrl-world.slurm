#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=rollout-ctrl-world
#SBATCH --time=00:30:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

echo "job is starting on `hostname`"
module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world

# python /u/asunesara/reward-model-design/run_ctrl_world.py \
#     --robomimic_ckpt /projects/bggq/asunesara/bc_transformer_trained_models/square_bc_transformer_gmm_4/20260401184240/models/model_epoch_61_best_validation_-11.562271785736083.pth \
#     --ctrl_world_ckpt /projects/bggq/asunesara/ctrl_world_robomimic_ckpt_2/checkpoint-10000.pt \
#     --dataset_dir /projects/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents_skip \
#     --output_dir /projects/bggq/asunesara/ctrl_world_rollouts \
#     --num_rollouts 2 \
#     --svd_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
#     --clip_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/ \
#     --data_stat_path /u/asunesara/Ctrl-World/dataset_meta_info/robomimic_policy_square_latents_skip/stat.json

# echo "Rollout complete"

python /u/asunesara/reward-model-design/run_ctrl_world.py \
    --robomimic_ckpt /projects/bggq/asunesara/bc_transformer_trained_models/square_bc_transformer_gmm_4/20260401184240/models/model_epoch_61_best_validation_-11.562271785736083.pth \
    --ctrl_world_ckpt /projects/bggq/asunesara/ctrl_world_robomimic_ckpt/checkpoint-30000.pt \
    --dataset_dir /projects/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents_skip \
    --output_dir /projects/bggq/asunesara/ctrl_world_rollouts \
    --num_rollouts 2 \
    --svd_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
    --clip_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/ \
    --data_stat_path /u/asunesara/Ctrl-World/dataset_meta_info/robomimic_square/stat.json