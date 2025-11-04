import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join
import os, glob

# Load data directory
VAL_IMG_DIR   = '/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val/'
TRAIN_IMG_DIR = '/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train/'

# Build list with validation and train set paths
val = glob.glob(os.path.join(VAL_IMG_DIR, '*.JPEG')) 
train = glob.glob(os.path.join(TRAIN_IMG_DIR, '**', '*.JPEG'), recursive=True)