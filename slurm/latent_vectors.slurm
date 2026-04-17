#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=latent_vectors
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH -e slurm-latent-vectors-%j.err
#SBATCH -o slurm-latent-vectors-%j.out
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"
 
echo "job is starting on `hostname`"
module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision no \
    /u/asunesara/reward-model-design/extract_latent_robomimic.py \
        --hdf5_path /scratch/bggq/asunesara/rollout_dataset/output_dense_shaping.hdf5 \
        --output_path /scratch/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents \
        --svd_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
        --rgb_skip 1

# accelerate launch \
#     --num_processes 2 \
#     --num_machines 1 \
#     --mixed_precision no \
#     /u/asunesara/reward-model-design/extract_latent_robomimic.py \
#         --hdf5_path /projects/bggq/asunesara/rollout_dataset/output.hdf5 \
#         --output_path /projects/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents_skip \
#         --svd_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
#         --rgb_skip 4

echo "Encoding complete. Copying output to /projects..."
echo "Copy complete."