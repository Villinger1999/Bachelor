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
# python train.py b64_e100_relu relu
# python train.py b64_e100_leaky leaky_relu
# python train.py b64_e100_tanh tanh
# python train.py b64_e100_sig sigmoid
python train.py b64_e100_soft softmax
# python train.py b64_e100_lin linear

# python train.py b64_e100_resnet