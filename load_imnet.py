import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os

# Load data directory
LOAD_DIR = '/dtu/datasets1/imagenet_object_localization_patched2019/'

import os, glob
VAL_IMG_DIR   = '/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val'
TRAIN_IMG_DIR = '/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train'

n_val   = len(glob.glob(os.path.join(VAL_IMG_DIR, '*.JPEG')))
n_train = len(glob.glob(os.path.join(TRAIN_IMG_DIR, '**', '*.JPEG'), recursive=True))

print('val images:', n_val)       # should be 50000
print('train images:', n_train)   # ~1.2M if present


# Save directory
path = os.getcwd() + "/"   # always points to the folder you are in
SAVE_DIR = path + 'data/'

# Load CSV files
submission = pd.read_csv(join(LOAD_DIR, 'LOC_sample_submission.csv'))
train = pd.read_csv(join(LOAD_DIR, 'LOC_train_solution.csv'))
val = pd.read_csv(join(LOAD_DIR, 'LOC_val_solution.csv'))

# Load synset mapping 
mapping = pd.read_csv(join(LOAD_DIR, 'LOC_synset_mapping.txt'), sep='\t', header=None, names=['synset', 'label'])

# Create destination folder if it doesn’t exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Save copies of each file as CSVs in your chosen folder
submission.to_csv(join(SAVE_DIR, 'submission.csv'), index=False)
train.to_csv(join(SAVE_DIR, 'train.csv'), index=False)
val.to_csv(join(SAVE_DIR, 'val.csv'), index=False)
mapping.to_csv(join(SAVE_DIR, 'synset_mapping.csv'), index=False)