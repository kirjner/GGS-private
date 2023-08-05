#!/bin/bash

#SBATCH --job-name=smooth
#SBATCH --output=logs/smooth/AAV-hard_%j.out
#SBATCH --error=logs/smooth/AAV-hard_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:volta:1

python biggs/GS.py experiment=smooth/AAV-hard
