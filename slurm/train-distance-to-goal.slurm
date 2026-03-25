#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=train-dtg
#SBATCH --time=00:30:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

echo "job is starting on `hostname`"
module load pytorch-conda/2.8
conda activate robomimic

python /u/asunesara/reward-model-design/success_detector/train_distance_to_goal.py \
    --latent_root /projects/bggq/asunesara/rollout_dataset/robomimic_policy_square_latents \
    --hdf5_path /projects/bggq/asunesara/rollout_dataset/output.hdf5 \
    --output_dir /projects/bggq/asunesara/distance_to_goal_runs/run1 \
    --epochs 100 \
    --batch_size 256 \
    --lr 3e-4 \
    --num_workers 2 \
    --preload \
    --success_threshold 2.0

echo "Training complete"


# srun --account=bggq-delta-gpu --partition=gpuA40x4-interactive,gpuA100x4-interactive   --nodes=1 --gpus-per-node=1 --tasks=1   --tasks-per-node=2 --cpus-per-task=2 --mem=64g   --pty /bin/bash
