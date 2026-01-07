#!/bin/bash
#BSUB -J iqa
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o iqa_%J.out
#BSUB -e iqa_%J.err

# # Load Python if needed 
module load python/3.12.11  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python iqa_score_assesment.py \
  --csv results_normal_orig.csv \
  --model state_dict_b64_e150_sig2.pt \
  --img_idx 0 \
  --base_seed 123 \
  --iterations 100 \
  --defense none \
  --out_dir viz_img0_bins_lpips
