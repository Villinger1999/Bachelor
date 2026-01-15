#!/bin/bash

#BSUB -J iDLG_def
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o iDLG_def_%J.out
#BSUB -e iDLG_def_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python main_Tommy.py None True 0.0 0 Clipping 0.03 b64_e150_img0_Clip_03_1
python main_Tommy.py None True 0.0 0 Cliiping 0.03 b64_e150_img0_Clip_03_2
python main_Tommy.py None True 0.0 0 Clipping 0.03 b64_e150_img0_Clip_03_3
python main_Tommy.py None True 0.0 0 Clipping 0.03 b64_e150_img0_Clip_03_4
python main_Tommy.py None True 0.0 0 Clipping 0.03 b64_e150_img0_Clip_03_5
python main_Tommy.py None True 0.0 0 Clipping 0.04 b64_e150_img0_Clip_04_1
python main_Tommy.py None True 0.0 0 Clipping 0.04 b64_e150_img0_Clip_04_2
python main_Tommy.py None True 0.0 0 Clipping 0.04 b64_e150_img0_Clip_04_3
python main_Tommy.py None True 0.0 0 Clipping 0.04 b64_e150_img0_Clip_04_4
python main_Tommy.py None True 0.0 0 Clipping 0.04 b64_e150_img0_Clip_04_5
# python main_Tommy.py None True 0.0 0 SGP 0.7 b64_e150_img0_SGP_7
# python main_Tommy.py None True 0.0 0 SGP 0.65 b64_e150_img0_SGP_65
# python main_Tommy.py None True 0.0 0 SGP 0.6 b64_e150_img0_SGP_6
# python main_Tommy.py None True 0.0 0 SGP 0.55 b64_e150_img0_SGP_55
# python main_Tommy.py None True 0.0 0 SGP 0.5 b64_e150_img0_SGP_5
# python main_Tommy.py None True 0.0 0 SGP 0.45 b64_e150_img0_SGP_45
# python main_Tommy.py None True 0.0 0 SGP 0.4 b64_e150_img0_SGP_4
# python main_Tommy.py None True 0.0 0 SGP 0.35 b64_e150_img0_SGP_35
# python main_Tommy.py None True 0.0 0 SGP 0.3 b64_e150_img0_SGP_3
# python main_Tommy.py None True 0.0 0 SGP 0.25 b64_e150_img0_SGP_25


# python main_Tommy.py None True 0.0 0 N 0.95 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.9 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.85 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.8 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.75 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.7 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.65 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.6 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.55 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.5 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.45 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.4 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.35 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.3 c6_b64_e10_img0_SGP
# python main_Tommy.py None True 0.0 0 N 0.25 c6_b64_e10_img0_SGP
