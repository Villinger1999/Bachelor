#!/bin/bash

#BSUB -J batch1
#BSUB -q gpua10 
#BSUB -W 260
#BSUB -R "rusage[mem=30G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 3
#BSUB -o batch1_%J.out
#BSUB -e batch1_%J.err

# Load Python 
module load python/3.9.21  

# Activate your virtual environment
source ~/torch-env/bin/activate

python batchloader.py 2 10 5 1 1