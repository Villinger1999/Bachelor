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

python main_Tommy.py None True 0.0 0 img0_origGrads
python main_Tommy.py None True 0.0 1 img1_origGrads
python main_Tommy.py None True 0.0 4 img4_origGrads
python main_Tommy.py None True 0.0 5 img5_origGrads
python main_Tommy.py None True 0.0 7 img7_origGrads
python main_Tommy.py None True 0.0 13 img13_origGrads