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

python get_results/get_scores_def.py \
  --csvs results/scores/results_fl_orig_sgp_9.csv results/scores/results_fl_orig_sgp_8.csv results/scores/results_fl_orig_sgp_7.csv results/scores/results_fl_orig_sgp_6.csv results/scores/results_fl_orig_sgp_5.csv results/scores/results_fl_orig_sgp_4.csv results/scores/results_fl_orig_sgp_3.csv results/scores/results_fl_orig_sgp_2.csv\
  --out results/cum_scores/scores_sgp/sgp.csv

python get_results/get_scores_def.py \
  --csvs results/scores/results_fl_orig_clip_999.csv results/scores/results_fl_orig_clip_998.csv results/scores/results_fl_orig_clip_997.csv results/scores/results_fl_orig_clip_996.csv results/scores/results_fl_orig_clip_995.csv results/scores/results_fl_orig_clip_994.csv results/scores/results_fl_orig_clip_993.csv results/scores/results_fl_orig_clip_99.csv \
  --out results/cum_scores/scores_clip/clip.csv

python get_results/get_scores_def.py \
  --csvs results/scores/results_fl_orig_normclip_995_2.csv results/scores/results_fl_orig_normclip_99_2.csv results/scores/results_fl_orig_normclip_98.csv results/scores/results_fl_orig_normclip_95.csv results/scores/results_fl_orig_normclip_93.csv results/scores/results_fl_orig_normclip_9.csv results/scores/results_fl_orig_normclip_88.csv results/scores/results_fl_orig_normclip_85.csv\
  --out results/cum_scores/norm_clip/normclip.csv

# python get_results/get_scores_def.py \
#   --csvs results/scores/results_FL_orig_none.csv results/scores/results_FL_orig_tv5_300.csv results/scores/results_FL_orig_tv6_300.csv results/scores/results_FL_orig_tv57_300.csv results/scores/results_FL_orig_tv47_300.csv results/scores/results_FL_orig_tv37_300.csv results/scores/results_FL_orig_tv27_300.csv results/scores/results_FL_orig_tv7_300.csv \
#   --out results/cum_scores/tvr/tvr_300.csv

python get_results/get_scores_def.py \
  --csvs results/scores/results_FL_orig_none.csv results/scores/results_FL_orig_tv5.csv results/scores/results_FL_orig_tv6.csv results/scores/results_FL_orig_tv57.csv results/scores/results_FL_orig_tv47_100.csv results/scores/results_FL_orig_tv37.csv results/scores/results_FL_orig_tv27.csv results/scores/results_FL_orig_tv7.csv \
  --out results/cum_scores/tvr/tvr.csv