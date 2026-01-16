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
  --model global_model_state_exp2_b64_e15_c10.pt \
  --img_idx 0 \
  --seed 123\
  --defense none \
  --tvr 0 \
  --percentile 0 \
  --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 0 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 1 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 2 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 2 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 3 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 3 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 4 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 4 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 5 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 5 \
#   --seed 123 \
#   --defense normclipping \
#   --tvr 3e-7 \
#   --percentile 0.9 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 6 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 0 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 1 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 2 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 3 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 4 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 5 \
#   --seed 123 \
#   --defense sgp \
#   --tvr 3e-7 \
#   --percentile 0.4 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 7 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 7 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0.995 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 8 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 8 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0.995 \
#   --iterations 100

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 9 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 300

# python visualize.py \
#   --model global_model_state_exp2_b64_e15_c10.pt \
#   --img_idx 9 \
#   --seed 123 \
#   --defense none \
#   --tvr 3e-7 \
#   --percentile 0 \
#   --iterations 100