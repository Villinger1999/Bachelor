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

python idlg_LeNet.py emptyDummy
# python idlg_LeNet.py noise_01_FL2
# python idlg_LeNet.py noise_1_FL2
# python idlg_LeNet.py noise_2_FL2
# python idlg_LeNet.py noise_3_FL2
# python idlg_LeNet.py noise_4_FL2
# python idlg_LeNet.py noise_5_FL2