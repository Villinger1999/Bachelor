#!/bin/bash

#BSUB -J ks_test_on_brisque_scores
#BSUB -q hpc
#BSUB -W 180
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# # Load Python if needed (depends on DTU module system)

module load python3/3.12.11

# Activate your virtual environment
# source ~/bachelor-env/Scripts/activate
source /zhome/8e/8/187047/Documents/Bachelor/bachelor-env/bin/activate

# variance 0.0 and 0.01, reslution inverval [32,96], every forth number in the interval, using 100 images, not saving the plots locally
python brisque_value.py /dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/ 0.01 32 96 4 100 plot

python brisque_value.py /dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/ 0.05 32 96 4 100 no_plot

python brisque_value.py /dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/ 0.1 32 96 4 100 no_plot

python brisque_value.py /dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/ 0.2 32 96 4 100 no_plot

python brisque_value.py /dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/ 0.3 32 96 4 100 no_plot

