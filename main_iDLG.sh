#!/bin/bash

#BSUB -J train
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed 
module load python/3.10.12  

# Activate your virtual environment
source /zhome/8e/8/187047/Documents/Bachelor/bachelor/bin/activate

python main_iDLG.py