#!/bin/bash

#BSUB -J bris_02_03
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

# variance 0.0 and 0.0001, reslutions used every forth number in the interval [32,120], using 1000 images, saving the plots locally
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0001
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0004
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0009
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0016
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0025
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0036
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0049
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0064
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0081
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.01
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0121
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.0144
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.02
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.03
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.04
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.05
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.06
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.07
python brisque_value.py --image_count 10 --res_ub 32 --variance 0.08


