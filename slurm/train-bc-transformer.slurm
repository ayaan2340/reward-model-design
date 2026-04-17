#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=train-bc-transformer-4
#SBATCH --time=6:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-train-bc-transformer-%j.err
#SBATCH -o slurm-train-bc-transformer-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"

echo "job is starting on `hostname`"
module load pytorch-conda/2.8
conda activate robomimic

python /u/asunesara/reward-model-design/train_bc_transformer.py \
    --dataset /projects/bggq/asunesara/subsampled_deltas.hdf5 \
    --output_dir /projects/bggq/asunesara/bc_transformer_trained_models \
    --name square_bc_transformer_gmm_4 

echo "Training complete"
