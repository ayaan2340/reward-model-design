#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x8
#SBATCH --account=bggq-delta-gpu
#SBATCH --job-name=collect-policy-demos
#SBATCH --time=04:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-collect-policy-demos-%j.err
#SBATCH -o slurm-collect-policy-demos-%j.out
#SBATCH --gpus-per-node=1 
#SBATCH --mail-user=ayaans@utexas.edu
#SBATCH --mail-type="BEGIN,END"
 
echo "job is starting on `hostname`"
module load pytorch-conda/2.8
conda activate robomimic
# Writes mask/train + mask/valid; attrs task_success, done_mode; dones match dataset_states_to_obs (default --done-mode 0).
python /u/asunesara/robomimic/robomimic/scripts/run_trained_agent.py \
  --agent /u/asunesara/robomimic/robomimic/models/square_ph_image_epoch_540_succ_78.pth \
  --obs_camera_names agentview frontview robot0_eye_in_hand \
  --obs_camera_height 192 --obs_camera_width 320 \
  --dataset_path /scratch/bggq/asunesara/rollout_dataset/output_dense_shaping.hdf5 \
  --dataset_obs \
  --reward-shaping true \
  --done-mode 0 \
  --val-split-ratio 0.1 \
  --split-seed 0 \
  --seed 0 \
  --horizon 400 \
  --n_rollouts 300

echo "Demo collection complete"
