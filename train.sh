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

# python train_FL.py b1_e1_c1_FL
python train_FL.py b64_e15_c10_2