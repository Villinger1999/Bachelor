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

python main_Tommy.py None True 0 0 img0_thres_c11 clipping 0.99996
python main_Tommy.py None True 0 0 img0_thres_c12 clipping 0.99996
python main_Tommy.py None True 0 0 img0_thres_c13 clipping 0.99996
python main_Tommy.py None True 0 0 img0_thres_c14 clipping 0.99995
python main_Tommy.py None True 0 0 img0_thres_c15 clipping 0.99995
python main_Tommy.py None True 0 0 img0_thres_c16 clipping 0.99995
python main_Tommy.py None True 0 0 img0_thres_c17 clipping 0.99994
python main_Tommy.py None True 0 0 img0_thres_c18 clipping 0.99994
python main_Tommy.py None True 0 0 img0_thres_c19 clipping 0.99994


python main_Tommy.py None True 0 0 img0_thres_s11 sgp 0.94
python main_Tommy.py None True 0 0 img0_thres_s12 sgp 0.94
python main_Tommy.py None True 0 0 img0_thres_s13 sgp 0.94
python main_Tommy.py None True 0 0 img0_thres_s14 sgp 0.93
python main_Tommy.py None True 0 0 img0_thres_s15 sgp 0.93
python main_Tommy.py None True 0 0 img0_thres_s16 sgp 0.93
python main_Tommy.py None True 0 0 img0_thres_s17 sgp 0.92
python main_Tommy.py None True 0 0 img0_thres_s18 sgp 0.92
python main_Tommy.py None True 0 0 img0_thres_s19 sgp 0.92












