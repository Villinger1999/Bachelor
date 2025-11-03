import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("dimensi0n/imagenet-256")

# Move to a custom folder
p = os.getcwd() + "/"   
target_path = p + "data/imnet"
shutil.move(path, target_path)

print("Moved dataset to:", target_path)