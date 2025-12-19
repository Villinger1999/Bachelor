#!/bin/bash

#BSUB -J iDLG_FL
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o iDLG_FL_%J.out
#BSUB -e iDLG_FL_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python main_Tommy.py None True 0.0 0 N c6_b64_e10_img0_FL_seedNone_ex2
python main_Tommy.py None True 0.0 1 N c6_b64_e10_img1_FL_seedNone_ex2
python main_Tommy.py None True 0.0 2 N c6_b64_e10_img2_FL_seedNone_ex2
python main_Tommy.py None True 0.0 3 N c6_b64_e10_img3_FL_seedNone_ex2
python main_Tommy.py None True 0.0 4 N c6_b64_e10_img4_FL_seedNone_ex2