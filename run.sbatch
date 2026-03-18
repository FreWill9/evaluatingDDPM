#!/bin/bash
#SBATCH --job-name=main
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=main.out
#SBATCH --error=main.err

source /opt/csg/anaconda/etc/profile.d/conda.sh
conda activate eml

which python
python --version

python train.py
python sampling_utils.py
python utils.py
