#!/bin/bash

#BSUB -J noise
#BSUB -q hpc 
#BSUB -W 200
#BSUB -R "rusage[mem=20G]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o noise_%J.out
#BSUB -e noise_%J.err

# Load Python 
module load python/3.9.21  

# Activate your virtual environment
source ~/bachelor-env/bin/activate

python test_train_noiseadd.py data/noise/noisy0_0.jpg noise_0
python test_train_noiseadd.py data/noise/noisy0_0.01.jpg noise_01
python test_train_noiseadd.py data/noise/noisy0_0.1.jpg noise_1
python test_train_noiseadd.py data/noise/noisy0_0.2.jpg noise_2
python test_train_noiseadd.py data/noise/noisy0_0.3.jpg noise_3
python test_train_noiseadd.py data/noise/noisy0_0.4.jpg noise_4
python test_train_noiseadd.py data/noise/noisy0_0.5.jpg noise_5