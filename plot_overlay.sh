#!/bin/bash
#BSUB -J model
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o model_%J.out
#BSUB -e model_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python plot_overlay.py \
  --csvs results_normal_orig.csv results_FL_orig.csv results_normal_orig2.csv results_FL_orig_2.csv results_normal_orig3.csv results_FL_orig_3.csv \
  --labels normal_orig fl_orig normal_orig2 fl_orig2 normal_orig3 fl_orig3 \
  --out compare_scatter_all.png \
  --show