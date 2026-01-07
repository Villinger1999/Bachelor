#!/bin/bash
#BSUB -J clip
#BSUB -q hpc
#BSUB -W 800
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o clip_%J.out
#BSUB -e clip_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario fl_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --images 0-10 \
  --repeats 5 \
  --iterations 100 \
  --defense clipping \
  --def_params 0.7,0.6,0.5,0.4 \
  --out_csv results_fl_orig_clip_multi2.csv