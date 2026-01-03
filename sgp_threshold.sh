#!/bin/bash
#BSUB -J sgp
#BSUB -q hpc
#BSUB -W 600
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o sgp_%J.out
#BSUB -e sgp_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model state_dict_b64_e150_sig2.pt \
  --images 0 \
  --repeats 100 \
  --iterations 100 \
  --defense sgp \
  --def_params 0.85,82,0.80,0.75 \
  --out_csv results_normal_orig_sgp.csv