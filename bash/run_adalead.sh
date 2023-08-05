#!/bin/bash

#SBATCH --job-name=run_adalead
#SBATCH --output=logs/run_adalead_%j.out
#SBATCH --error=logs/run_adalead_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:volta:1


python biggs/run_flexs_baselines.py 
