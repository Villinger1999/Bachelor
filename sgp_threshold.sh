#!/bin/bash
#BSUB -J sgp
#BSUB -q hpc
#BSUB -W 900
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o sgp_%J.out
#BSUB -e sgp_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 0-49 \
  --repeats 25 \
  --iterations 100 \
  --defense sgp \
  --def_params 0.1 \
  --out_csv results_fl_orig_sgp_multi_1.csv