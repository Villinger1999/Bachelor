#!/bin/bash

#BSUB -J brisque
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o brisque_%J.out
#BSUB -e brisque_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python brisque_imnet_hpc.py