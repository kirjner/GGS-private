#!/bin/bash

#SBATCH --job-name=train_smooth
#SBATCH --output=logs/train/smoothed/Diagonal-smoothed_%j.out
#SBATCH --error=logs/train/smoothed/Diagonal-smoothed_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1


python ggs/train_predictor.py experiment=train/Diagonal_smoothed
