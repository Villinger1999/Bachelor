#!/bin/bash
#BSUB -J get_score
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o get_score_%J.out
#BSUB -e get_score_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python get_scores_def.py \
  --csvs results_fl_orig_sgp_multi_1.csv results_fl_orig_sgp_multi_2.csv results_fl_orig_sgp_multi_3.csv results_fl_orig_sgp_multi_4.csv results_fl_orig_sgp_multi_5.csv results_fl_orig_sgp_multi_6.csv results_fl_orig_sgp_multi_7.csv results_fl_orig_sgp_multi_8.csv results_fl_orig_sgp_multi_9.csv \
  --outdir scores/scores_sgp