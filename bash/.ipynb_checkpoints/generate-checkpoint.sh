#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=logs/generate/GFP_hard_smoothed_%j.out
#SBATCH --error=logs/generate/GFP_hard_smoothed_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=32GB
#SBATCH --array=1-5


SEED=$SLURM_ARRAY_TASK_ID
python biggs/BiG.py experiment=generate/GFP-hard-smoothed run.seed=$SEED
