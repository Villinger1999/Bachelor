#!/bin/bash
#BSUB -J idlg
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o idlg_%J.out
#BSUB -e idlg_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model state_dict_b64_e150_sig2.pt \
  --images 0 \
  --repeats 1000 \
  --iterations 100 \
  --defense clipping \
  --def_params 0.99993,0.99992,0.99991,0.9999,0.9998,0.9995,0.999 \
  --out_csv results_normal_orig_clip.csv