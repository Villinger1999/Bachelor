#!/bin/bash

#BSUB -J resnet_comparison
#BSUB -q hpc
#BSUB -W 180
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed (depends on DTU module system)

module load python3/3.12.11 

# Activate your virtual environment
# source ~/bachelor-env/bin/activate
source /zhome/8e/8/187047/Documents/Bachelor/bachelor-env/bin/activate

python models/numpy_torch_com.py