#!/bin/bash

#BSUB -J bris_02_03
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed (depends on DTU module system)

module load python3/3.12.11

# Activate your virtual environment
# source ~/bachelor-env/Scripts/activate
source /zhome/8e/8/187047/Documents/Bachelor/bachelor-env/bin/activate

# variance 0.0 and 0.0001, reslutions used every forth number in the interval [32,120], using 1000 images, saving the plots locally
python train_ImageNet.py test
