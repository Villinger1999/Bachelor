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

# python get_scores.py \
#   --csvs results_fl_orig_clip_multi_995.csv \
#   --outdir scores/scores_clip

  python get_scores.py \
  --csvs results_fl_orig_sgp_multi_44.csv \
  --outdir scores/scores_sgp

# python get_scores.py \
#   --csvs results_FL_orig_1.csv \
#   --outdir scores