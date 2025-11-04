#!/bin/bash

#BSUB -J train
#BSUB -q gpua10 
#BSUB -W 200
#BSUB -R "rusage[mem=15G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed (depends on DTU module system)
module load python/3.10.12  

# Activate your virtual environment
source ~/torch-env/bin/activate

python main_imnet.py 5 20 5 64 1