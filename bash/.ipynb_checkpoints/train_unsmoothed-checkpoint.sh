#!/bin/bash

#SBATCH --job-name=train_unsmooth
#SBATCH --output=logs/train/unsmoothed/Recombination-oracle_%j.out
#SBATCH --error=logs/train/unsmoothed/Recombination-oracle_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1


python ggs/train_predictor.py experiment=train/Recombination-oracle