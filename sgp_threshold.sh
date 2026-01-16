#!/bin/bash
#BSUB -J sgp
#BSUB -q hpc
#BSUB -W 1100
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o sgp_%J.out
#BSUB -e sgp_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model global_model_state_exp2_b64_e15_c10.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 49 \
  --repeats 25 \
  --iterations 100 \
  --tvr 3e-7 \
  --defense sgp \
  --def_params 0.9 \
  --out_csv results_fl_orig_sgp_9.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 38-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.8 \
#   --out_csv results_fl_orig_sgp_8.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.7 \
#   --out_csv results_fl_orig_sgp_7.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.6 \
#   --out_csv results_fl_orig_sgp_6.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt\
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.5 \
#   --out_csv results_fl_orig_sgp_5.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.4 \
#   --out_csv results_fl_orig_sgp_4.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.3 \
#   --out_csv results_fl_orig_sgp_3.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.2 \
#   --out_csv results_fl_orig_sgp_2.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense sgp \
#   --def_params 0.1 \
#   --out_csv results_fl_orig_sgp_1.csv