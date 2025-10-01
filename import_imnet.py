import zipfile
import os
import random
from pathlib import Path

# Make path the folder path
path = path = os.getcwd() + "/"   

zip_path = "data/imnet.zip"                 # path to downloaded Kaggle file
extract_path = "data/imnet_subset_1000"     # folder to extract subset into
subset_size = 1000                          # number of random images to extract
random_seed = 42                            # seed for reproducibility


os.makedirs(path + extract_path, exist_ok=True)
random.seed(random_seed)

with zipfile.ZipFile(path + zip_path, 'r') as zip_ref:
    # Get all image file paths inside ZIP
    all_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total images in ZIP: {len(all_files):,}")

    # Randomly sample subset
    if subset_size > len(all_files):
        raise ValueError("Subset size larger than total available images!")

    subset_files = random.sample(all_files, subset_size)
    print(f"Sampling {subset_size} random images...")

    # Extract only selected images
    for f in subset_files:
        zip_ref.extract(f, path + extract_path)

print(f"Done! Extracted {len(subset_files)} images to '{path + extract_path}'")

