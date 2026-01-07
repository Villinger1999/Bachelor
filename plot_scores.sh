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

python plot_scores.py --csv results_FL_orig_3.csv --out scatter_FL3.png --show
