#!/bin/bash

#BSUB -J brisque_std_01
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
# source ~/bachelor-env/Scripts/activate
source /zhome/8e/8/187047/Documents/Bachelor/bachelor-env/bin/activate

# variance 0.0 and 0.01, reslution inverval [32,96], every forth number in the interval, using 100 images, not saving the plots locally
python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0001

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0004

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0009

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0016

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0025

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0036

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0049

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0064

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.0081

python brisque_value.py --plot --image_count 1000 --res_ub 120 --variance 0.01