#!/bin/bash
#BSUB -J idlg_fl
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o idlg_fl_%J.out
#BSUB -e idlg_fl_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 100 \
  --iterations 100 \
  --defense none \
  --out_csv results_FL_orig_3.csv