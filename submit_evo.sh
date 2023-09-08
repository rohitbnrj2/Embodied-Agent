#!/bin/bash
#SBATCH -c 25
#SBATCH --gres=gpu:volta:1
python -u ./cambrian/reinforce/basic_evo_pixels.py