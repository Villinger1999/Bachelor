#!/bin/bash

#BSUB -J tommy0
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o tommy0_%J.out
#BSUB -e tommy0_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python main_Tommy.py data/noise/noisy0_0.jpg noise_0_LeNet_N
python main_Tommy.py data/noise/noisy0_0.01.jpg noise_01_LeNet_N
python main_Tommy.py data/noise/noisy0_0.1.jpg noise_1_LeNet_N
python main_Tommy.py data/noise/noisy0_0.2.jpg noise_2_LeNet_N
python main_Tommy.py data/noise/noisy0_0.3.jpg noise_3_LeNet_N
python main_Tommy.py data/noise/noisy0_0.4.jpg noise_4_LeNet_N
python main_Tommy.py data/noise/noisy0_0.5.jpg noise_5_LeNet_N