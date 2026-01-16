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

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model global_model_state_exp2_b64_e15_c10.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 0-24 \
  --seed 123 \
  --repeats 25 \
  --iterations 100 \
  --tvr 0 \
  --defense none \
  --out_csv results_FL_orig_none_2.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 8-9 \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_1.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 10-19 \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_2.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 27-29 \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_3.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 39 \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_4.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 40-49 \
#   --seed 123 \
#   --repeats 50 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense none \
#   --out_csv results_FL_orig_5.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_none.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_tv4.csv

  # python run_idlg_ex.py \
  # --scenario fl_model_orig_grads \
  # --normal_model global_model_state_exp2_b64_e15_c10.pt \
  # --fl_model global_model_state_exp2_b64_e15_c10.pt \
  # --images 0-9 \
  # --seed 123 \
  # --repeats 5 \
  # --iterations 300 \
  # --tvr 1e-5 \
  # --defense none \
  # --out_csv results_FL_orig_tv5_100.csv

  # python run_idlg_ex.py \
  # --scenario fl_model_orig_grads \
  # --normal_model global_model_state_exp2_b64_e15_c10.pt \
  # --fl_model global_model_state_exp2_b64_e15_c10.pt \
  # --images 0-9 \
  # --seed 123 \
  # --repeats 5 \
  # --iterations 100 \
  # --tvr 1e-6 \
  # --defense none \
  # --out_csv results_FL_orig_tv6_100.csv

  # python run_idlg_ex.py \
  # --scenario fl_model_orig_grads \
  # --normal_model global_model_state_exp2_b64_e15_c10.pt \
  # --fl_model global_model_state_exp2_b64_e15_c10.pt \
  # --images 0-9 \
  # --seed 123 \
  # --repeats 5 \
  # --iterations 100 \
  # --defense none \
  # --out_csv results_FL_orig_tv7_100.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --tvr 2e-7 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_tv27_100.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --tvr 3e-7 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_tv37_100.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --tvr 5e-7 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_tv57_100.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --tvr 4e-7 \
#   --iterations 300 \
#   --defense none \
#   --out_csv results_FL_orig_tv47_300.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-9 \
#   --seed 123 \
#   --repeats 5 \
#   --tvr 4e-7 \
#   --iterations 100 \
#   --defense none \
#   --out_csv results_FL_orig_tv47_100.csv