#!/bin/bash
#BSUB -J idlg_fl
#BSUB -q hpc
#BSUB -W 1000
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o idlg_fl_%J.out
#BSUB -e idlg_fl_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_balanced.py \
  --scenario fl_model_orig_grads \
  --normal_model global_model_state_exp2_b64_e15_c10.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --seed 123 \
  --repeats 25 \
  --iterations 100 \
  --balanced \
  --num_images 50 \
  --save_indices balanced50_seed123.txt \
  --tvr 3e-7 \
  --defense none \
  --out_csv results_FL_orig_1_b.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --balanced \
#   --num_images 20 \
#   --save_indices balanced50_seed123_2.txt \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_2_b.csv