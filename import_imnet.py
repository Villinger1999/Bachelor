import zipfile
import os
import random
from pathlib import Path
from PIL import Image
from io import BytesIO

# Make path the folder path
path = path = os.getcwd() + "/"   

zip_path = "data/imnet.zip"                 # path to downloaded Kaggle file
extract_path = "data/imnet_subset_1000"     # folder to extract subset into
subset_size = 1000                          # number of random images to extract
random_seed = 42                            # for reproducibility


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
        # Read file from zip directly
        with zip_ref.open(f) as file:
            img = Image.open(BytesIO(file.read()))
            img = img.convert("RGB")                 # ensure RGB
            img = img.resize((224, 224), Image.LANCZOS) 
            
            # Save resized image to extract_path
            out_path = Path(path + extract_path) / Path(f).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=95)

print(f"Extracted {len(subset_files)} images to '{path + extract_path}'")
