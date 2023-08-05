#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate/GFP-hard-smoothed_%j.out
#SBATCH --error=logs/evaluate/GFP-hard-smoothed_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1


python biggs/evaluate.py experiment=evaluate/GFP-hard-smoothed
