#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=logs/generate/Diagonal-unsmoothed_%j.out
#SBATCH --error=logs/generate/Diagonal-unsmoothed_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=32GB
#SBATCH --array=1-5

python ggs/GWG.py experiment=generate/Diagonal-unsmoothed run.seed=$SLURM_ARRAY_TASK_ID 
