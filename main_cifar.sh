#!/bin/bash

#BSUB -J train
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=20G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

<<<<<<< HEAD:main_cifar.sh
# # Load Python 
module load python/3.10.12  
=======
# # Load Python if needed (depends on DTU module system)
module load python/3.9.21  
>>>>>>> main:main.sh

# Activate your virtual environment
source /zhome/8e/8/187047/Documents/Bachelor/bachelor/bin/activate

python main.py 3 3 4 1 1