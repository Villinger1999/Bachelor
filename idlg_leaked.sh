#!/bin/bash
#BSUB -J idlg_leaked
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o idlg_leaked_%J.out
#BSUB -e idlg_leaked_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python run_idlg_ex.py \
  --scenario normal_model_leaked_grads \
  --normal_model state_dict_b64_e150_sig2.pt \
  --fl_model global_model_state_exp2_b64_e15_c10.pt \
  --leaked_grads local_grads_client0_exp4.pt\
  --images 1-9 \
  --repeats 10 \
  --iterations 100 \
  --defense none \
  --out_csv results_normal_leaked_multi.csv