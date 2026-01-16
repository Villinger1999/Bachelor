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

python get_results/get_scores.py \
  --csvs results/scores/results_fl_orig_clip_995.csv \
  --outdir results/cum_scores/scores_clip

python get_results/get_scores.py \
  --csvs results/scores/results_fl_orig_sgp_4.csv \
  --outdir results/cum_scores/scores_sgp

python get_results/get_scores.py \
  --csvs results/scores/results_FL_orig_1.csv results/scores/results_FL_orig_2.csv results/scores/results_FL_orig_3.csv results/scores/results_FL_orig_4.csv results/scores/results_FL_orig_5.csv\
  --outdir results/cum_scores/scores_FL

python get_results/get_scores.py \
  --csvs results/scores/results_FL_orig_1_b.csv\
  --outdir results/cum_scores

python get_results/get_scores.py \
  --csvs results/scores/results_fl_orig_normclip_9.csv \
  --outdir results/cum_scores/norm_clip