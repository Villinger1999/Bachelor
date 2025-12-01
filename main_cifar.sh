#!/bin/bash

#BSUB -J cifar_b2_lenet
#BSUB -q hpc 
#BSUB -W 200
#BSUB -R "rusage[mem=20G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o cifar_b2_lenet_%J.out
#BSUB -e cifar_b2_lenet_%J.err

# Load Python
module load python3/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python main_cifar.py 1 1 1 1 1 c1_b1_e1_pretrained