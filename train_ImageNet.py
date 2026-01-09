from classes.federated_learning import Client
from classes.models import LeNet5
from torch.utils.data import DataLoader, TensorDataset
import torch
# import tensorflow as tf
import sys
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def get_images_from_folders(base_dir, label_folders, image_count):
    x = []
    y = []
    for label_idx, folder in enumerate(label_folders):
        folder_path = join(base_dir, folder)
        all_files = [f for f in listdir(folder_path) if f.endswith('.JPEG')]
        all_files.sort()
        selected_files = all_files[:image_count]
        x.extend([join(folder_path, fname) for fname in selected_files])
        y.extend([label_idx] * len(selected_files))
    return x, y

# Helper function to load and preprocess images
def load_and_preprocess_images(image_paths, resize):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize, resize))
        img_array = np.array(img)
        images.append(img_array)
    images_np = np.stack(images, axis=0)
    # Convert to (N, C, H, W)
    images_np = images_np.transpose((0, 3, 1, 2))
    return torch.tensor(images_np, dtype=torch.float32) / 255.0

data_path = "/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train/"
label_folders = ["n01440764", "n01775062",  "n02086079",  "n02106030",  "n02190166",  "n02504013",  "n02906734",  "n03223299",  "n03627232",  "n03873416"]

# Example usage:
base_dir = data_path  # or set to your ImageNet subset root
image_count = 50  # number of images per class
x_all, y_all = get_images_from_folders(base_dir, label_folders, image_count)

# Split into train/test (80/20 split)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all, random_state=42)


# Convert x_train/x_test to float tensors and normalize to [0, 1]
x_train_torch = load_and_preprocess_images(x_train,32)
x_test_torch  = load_and_preprocess_images(x_test,32)

# Convert labels to long tensors and flatten
y_train_torch = torch.tensor(np.array(y_train).squeeze(), dtype=torch.long)
y_test_torch  = torch.tensor(np.array(y_test).squeeze(), dtype=torch.long)

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

# Create DataLoader for the smaller test subset
testloader = DataLoader(testset, batch_size=64, shuffle=False)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

model = LeNet5()

client = Client(client_id=0, dataset=trainset, batch_size=64, device="cpu")
state_dict, _ = client.train_local(global_model=model, testloader=testloader, epochs=100, lr=0.01)

torch.save(state_dict, f"state_dict_{model._get_name()}_{str(sys.argv[1])}.pt")