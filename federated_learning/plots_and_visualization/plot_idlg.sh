#!/bin/bash
#BSUB -J tvr
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o plot_tvr_%J.out
#BSUB -e plot_tvr_%J.err

module load python/3.12.11
source ~/bachelor-env/bin/activate

python plots_and_visualization/plot_idlg.py 0
# python plots_and_visualization/plot_idlg.py 1
# python plots_and_visualization/plot_idlg.py 2
# python plots_and_visualization/plot_idlg.py 3
# python plots_and_visualization/plot_idlg.py 4
# python plots_and_visualization/plot_idlg.py 5
# python plots_and_visualization/plot_idlg.py 6