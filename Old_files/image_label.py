import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from pathlib import Path

# Make path the folder path
path = path = ""   
extract_path = "/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_10026.JPEG"     # folder to extract subset into

# Base path to extracted images
subset_dir = Path(path + extract_path)

# Collect all extracted images
image_paths = list(subset_dir.rglob("*.[jp][pn]g"))  # match .jpg/.png

print(f"Found {len(image_paths)} extracted images")

# Pick one random image
sample_path = random.choice(image_paths)

# Load the image
img = Image.open(sample_path).convert("RGB")

# Get label = parent folder name
label = sample_path.parent.name

# Display
plt.figure(figsize=(4,4))
plt.imshow(img)
plt.axis("off")
plt.title(f"Label: {label}", fontsize=12)
plt.show()