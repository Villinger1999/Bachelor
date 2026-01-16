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

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model global_model_state_exp2_b64_e15_c10.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 0-49 \
  --repeats 25 \
  --iterations 100 \
  --tvr 3e-7 \
  --defense clipping \
  --def_params 0.9999 \
  --out_csv results_fl_orig_clip_9999.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.9995 \
#   --out_csv results_fl_orig_clip_9995.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.9999 \
#   --out_csv results_fl_orig_clip_9999.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.998 \
#   --out_csv results_fl_orig_clip_998.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.997 \
#   --out_csv results_fl_orig_clip_997.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 0-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.996 \
#   --out_csv results_fl_orig_clip_996.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 44-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.995 \
#   --out_csv results_fl_orig_clip_995.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.994 \
#   --out_csv results_fl_orig_clip_994.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 47-49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.993 \
#   --out_csv results_fl_orig_clip_993.csv

# python run_idlg_ex.py \
#   --scenario fl_model_orig_grads \
#   --normal_model global_model_state_exp2_b64_e15_c10.pt \
#   --fl_model global_model_state_exp2_b64_e15_c10.pt \
#   --images 49 \
#   --repeats 25 \
#   --iterations 100 \
#   --tvr 3e-7 \
#   --defense clipping \
#   --def_params 0.99 \
#   --out_csv results_fl_orig_clip_99.csv