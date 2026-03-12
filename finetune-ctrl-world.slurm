#!/bin/bash
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=finetune-ctrl-world
#SBATCH --time=08:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=8
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"
 
echo "job is starting on `hostname`"
module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world
WANDB_MODE=offline accelerate launch --main_process_port 29501 \
            --num_processes 8 \
            --num_machines 1 \
            --mixed_precision fp16 \
            /u/asunesara/Ctrl-World/scripts/train_wm.py \
            --dataset_root_path /scratch/bggq/asunesara                                \
            --dataset_meta_info_path /u/asunesara/Ctrl-World/dataset_meta_info         \
            --dataset_names robomimic_square                                            \
            --dataset_cfgs robomimic_square                                             \
            --down_sample 4                                                             \
            --svd_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
            --clip_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/ \
            --ckpt_path /projects/bggq/asunesara/.cache/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt \
            --max_train_steps 30000                                                     \
            --learning_rate 5e-6                                                        \
            --checkpointing_steps 5000                                                  \
            --validation_steps 1000                                                      \
            --train_batch_size 1                                                        \
            --video_num 5                                                               \
            --output_dir /projects/bggq/asunesara/ctrl_world_robomimic_ckpt             \
            --wandb_project_name robomimic_square_finetune

echo "Training complete"


# srun --mem=64g --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --partition=gpuA100x4-interactive,gpuA40x4-interactive --account=bggq-delta-gpu --gpus-per-node=1 --time=00:15:00 --pty /bin/bash

# WANDB_MODE=offline accelerate launch --main_process_port 29501 \
#             --num_processes 1 \
#             --num_machines 1 \
#             --mixed_precision fp16 \
#             /u/asunesara/Ctrl-World/scripts/train_wm.py \
#             --dataset_root_path /scratch/bggq/asunesara                                \
#             --dataset_meta_info_path /u/asunesara/Ctrl-World/dataset_meta_info         \
#             --dataset_names robomimic_square                                            \
#             --dataset_cfgs robomimic_square                                             \
#             --down_sample 4                                                             \
#             --svd_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
#             --clip_model_path /projects/bggq/asunesara/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/ \
#             --ckpt_path /projects/bggq/asunesara/.cache/huggingface/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt \
#             --max_train_steps 15000                                                     \
#             --learning_rate 5e-6                                                        \
#             --checkpointing_steps 3000                                                  \
#             --validation_steps 500                                                      \
#             --train_batch_size 1                                                        \
#             --video_num 5                                                               \
#             --output_dir /projects/bggq/asunesara/ctrl_world_robomimic_ckpt             \
#             --wandb_project_name robomimic_square_finetune
