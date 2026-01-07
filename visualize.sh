#!/bin/bash
#BSUB -J viz
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o viz_%J.out
#BSUB -e viz_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python visualize.py \
  --csv  results_FL_orig.csv \
  --model state_dict_b64_e150_sig2.pt \
  --defense none \
  --out_dir vis_FL_orig.csv