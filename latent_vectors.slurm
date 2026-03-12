#!/bin/bash
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=latent_vectors
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"
 
echo "job is starting on `hostname`"
module load pytorch-conda/2.8
module load ffmpeg
conda activate ctrl-world
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision no \
    /u/asunesara/reward-model-design/extract_latent_robomimic.py \
        --hdf5_path /scratch/bggq/asunesara/square_image_dense_three.hdf5 \
        --output_path /scratch/bggq/asunesara/robomimic_square_latents \
        --svd_path /projects/bggq/asunesara/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/

echo "Encoding complete. Copying output to /projects..."
cp -r /scratch/bggq/asunesara/robomimic_square_latents \
      /projects/bggq/asunesara/robomimic_square_latents
echo "Copy complete."