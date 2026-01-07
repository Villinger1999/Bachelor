#!/bin/bash
#BSUB -J model
#BSUB -q hpc
#BSUB -W 400
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o model_%J.out
#BSUB -e model_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_sig.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_sig.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_sig_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation sigmoid \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_sig.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_sig_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation sigmoid \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_sig.csv


python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_leaky.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_leaky.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_leaky_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation leaky_relu \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_leaky.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_leaky_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation leaky_relu \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_leaky.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_lin.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_lin.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_lin_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation linear\
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_lin.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_lin_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation linear \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_lin.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_relu.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_relu.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_relu_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation relu \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_relu.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_relu_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation relu \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_relu.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_soft.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_soft.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_soft_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation softmax\
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_soft.csv

    python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_soft_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation softmax \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_soft.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_tanh.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_tanh.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_tanh_avg.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation tanh \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_tanh.csv

python model_ex.py \
  --scenario normal_model_orig_grads \
  --normal_model state_dict_b64_e100_tanh_max.pt \
  --fl_model global_model_state_exp6_b64_e15_c10_3.pt \
  --images 0 \
  --activation tanh \
  --repeats 50 \
  --iterations 100 \
  --defense none \
  --out_csv results_tanh.csv