#!/bin/bash
#BSUB -J clip
#BSUB -q hpc
#BSUB -W 600
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o clip_%J.out
#BSUB -e clip_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model state_dict_b64_e150_sig2.pt \
  --images 0 \
  --repeats 300 \
  --seed 323 \
  --iterations 100 \
  --defense clipping \
  --def_params 0.7 \
  --out_csv results_normal_orig_clip.csv