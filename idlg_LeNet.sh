#!/bin/bash

#BSUB -J tommy
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o tommy_%J.out
#BSUB -e tommy_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python idlg_LeNet.py data/noise/noisy0_0.jpg noise_0_FL1
python idlg_LeNet.py data/noise/noisy0_0.01.jpg noise_01_FL1
python idlg_LeNet.py data/noise/noisy0_0.1.jpg noise_1_FL1
python idlg_LeNet.py data/noise/noisy0_0.2.jpg noise_2_FL1
python idlg_LeNet.py data/noise/noisy0_0.3.jpg noise_3_FL1
python idlg_LeNet.py data/noise/noisy0_0.4.jpg noise_4_FL1
python idlg_LeNet.py data/noise/noisy0_0.5.jpg noise_5_FL1