#!/bin/bash

#BSUB -J iDLGdefense
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o iDLGdefense_%J.out
#BSUB -e iDLGdefense_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

# python main_Tommy.py None True 0.0 0 SGP e150_img0_SGP_0.0002
# python main_Tommy.py None True 0.0 1 SGP e150_img1_SGP_0.0002
python main_Tommy.py None True 0.0 1 PLGP e150_img1_PLGP_0.1_a0.8
python main_Tommy.py None True 0.0 0 PLGP e150_img0_PLGP_0.1_a0.8
python main_Tommy.py None True 0.0 0 Clipping e150_img0_Clipping_0.00005
python main_Tommy.py None True 0.0 1 Clipping e150_img1_Clipping_0.00005