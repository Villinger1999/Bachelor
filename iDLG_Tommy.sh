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

python main_Tommy.py None True 0.0 0 normal_img0_lr11_TV6
python main_Tommy.py None True 0.0 1 normal_img1_lr11_TV6
python main_Tommy.py None True 0.0 2 normal_img2_lr11_TV6
python main_Tommy.py None True 0.0 3 normal_img3_lr11_TV6
python main_Tommy.py None True 0.0 4 normal_img4_lr11_TV6
python main_Tommy.py None True 0.0 10 normal_img10_lr11_TV6
python main_Tommy.py None True 0.0 7 normal_img_7_lr11_TV6
python main_Tommy.py None True 0.0 33 normal_img_33_lr11_TV6
python main_Tommy.py None True 0.0 45 normal_img_45_lr11_TV6
python main_Tommy.py None True 0.0 12 normal_img_12_lr11_TV6
python main_Tommy.py None True 0.0 11 normal_img_11_lr11_TV6
python main_Tommy.py None True 0.0 68 normal_img_68_lr11_TV6