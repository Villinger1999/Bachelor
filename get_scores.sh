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

python get_scores.py \
  --csvs results_fl_orig_clip_995.csv \
  --outdir scores/scores_clip

python get_scores.py \
  --csvs results_fl_orig_sgp_4.csv \
  --outdir scores/scores_sgp

python get_scores.py \
  --csvs results_FL_orig_1.csv results_FL_orig_2.csv results_FL_orig_3.csv results_FL_orig_4.csv results_FL_orig_5.csv\
  --outdir scores/scores_FL

python get_scores.py \
  --csvs results_FL_orig_1_b.csv\
  --outdir scores

python get_scores.py \
  --csvs results_fl_orig_normclip_9.csv \
  --outdir scores/norm_clip