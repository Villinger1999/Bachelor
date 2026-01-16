#!/bin/bash

#BSUB -J train
#BSUB -q hpc
#BSUB -W 600
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
# python train_FL.py b64_e10_c5_clipping_4 clipping 0.4
# python train_FL.py b64_e10_c5_sgp_3 sgp 0.3
python train_FL.py b64_e10_c5_relu relu 
python train_FL.py b64_e10_c5_relu leaky_relu
python train_FL.py b64_e10_c5_tanh tanh
python train_FL.py b64_e10_c5_sig sigmoid
python train_FL.py b64_e10_c5_soft softmax
python train_FL.py b64_e10_c5_lin linear

# python train.py b64_e100_resnet