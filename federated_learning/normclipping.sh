#!/bin/bash
#BSUB -J clip
#BSUB -q hpc
#BSUB -W 1100
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o clip_%J.out
#BSUB -e clip_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.999 \
#   --out_csv results_fl_orig_normclip_999.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.998 \
#   --out_csv results_fl_orig_normclip_998.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.997 \
#   --out_csv results_fl_orig_normclip_997.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.996 \
#   --out_csv results_fl_orig_normclip_996.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.995 \
#   --out_csv results_fl_orig_normclip_995_2.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.994 \
#   --out_csv results_fl_orig_normclip_994.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.993 \
#   --out_csv results_fl_orig_normclip_993.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.85 \
#   --out_csv results_fl_orig_normclip_85.csv

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model global_model_state_exp2_b64_e15_c10.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 44-49 \
  --repeats 25 \
  --iterations 100 \
  --tvr 3e-7 \
  --defense normclipping \
  --def_params 0.88 \
  --out_csv results_fl_orig_normclip_88.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.93 \
#   --out_csv results_fl_orig_normclip_93.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 45-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.95 \
#   --out_csv results_fl_orig_normclip_95.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.98 \
#   --out_csv results_fl_orig_normclip_98.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.7 \
#   --out_csv results_fl_orig_normclip_7.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense normclipping \
#   --def_params 0.6 \
#   --out_csv results_fl_orig_normclip_6.csv