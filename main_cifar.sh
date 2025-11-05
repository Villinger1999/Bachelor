#!/bin/bash

#BSUB -J cifar_batch2
#BSUB -q gpua10 
#BSUB -W 200
#BSUB -R "rusage[mem=20G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o cifar_batch2_%J.out
#BSUB -e cifar_batch2_%J.err

# Load Python
module load python3/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python main_cifar.py 5 10 3 2 1 5c_10r_batch2