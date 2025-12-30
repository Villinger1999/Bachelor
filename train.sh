#!/bin/bash

#BSUB -J train
#BSUB -q hpc
#BSUB -W 300
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed (depends on DTU module system)

module load python/3.10.21 

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python train.py b1_e50_01 1 50 0.01 true
python train.py b1_e100_01 1 100 0.01 true
python train.py b64_e150_01 64 150 0.01 true
python train.py b64_e100_01 64 100 0.01 true
python train.py b64_e100_1 64 100 0.1 true
python train.py b64_e150_1 64 150 0.1 true