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

python plot_tvr.py 0
# python plot_tvr.py 1
# python plot_tvr.py 2
# python plot_tvr.py 3
# python plot_tvr.py 4
# python plot_tvr.py 5
# python plot_tvr.py 6
# python plot_tvr.py 7
# python plot_tvr.py 8
# python plot_tvr.py 9