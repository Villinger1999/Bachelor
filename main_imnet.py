from models.central_model import *
from load_imnet import train, val
from torch.utils.data import random_split
import sys
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
from PIL import Image
import os
import pandas as pd
import numpy as np

num_clients = int(sys.argv[1]) # e.g 3
num_rounds = int(sys.argv[2]) # e.g 2
local_epochs = int(sys.argv[3]) # e.g 1
batch_size = int(sys.argv[4]) # e.g 24
C = int(sys.argv[5]) # e.g 1

# make label map from train folder names
synsets = sorted({os.path.basename(os.path.dirname(p)) for p in train})
class_to_idx = {s: i for i, s in enumerate(synsets)}

# Take 10% of the total dataset
total_size = len(train)
subset_size = int(0.0005 * total_size)  # % of data
subset = random.sample(train, subset_size)
print(subset_size)

# Assume testset is a PyTorch Dataset (e.g., CIFAR10(test=True))
total_size = len(train)
subset_size = int(0.00005 * total_size)     # % subset
test_subset = random.sample(train, subset_size)
print(subset_size)

# Transform
img_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),              # now we have (C,H,W) float in [0,1]
])

x_train = []
y_train = []

for path in subset:
    img = Image.open(path).convert('RGB')
    img = img_tf(img) 
    x_train.append(img)
    
    # Get synset label from parent directory name
    synset = os.path.basename(os.path.dirname(path))
    label = class_to_idx[synset]
    y_train.append(label)
    
# stack to tensors
x_train_torch = torch.stack(x_train)                         # (N, C, H, W)
y_train_torch = torch.tensor(y_train, dtype=torch.long)      # (N,)
    
x_test = []
y_test = []
for path in test_subset:
    img = Image.open(path).convert('RGB')
    img = img_tf(img)
    x_test.append(img)
    
    # Get synset label from parent directory name
    synset = os.path.basename(os.path.dirname(path))
    label = class_to_idx[synset]
    y_test.append(label)

x_test_torch = torch.stack(x_test)                           # (M, C, H, W)
y_test_torch = torch.tensor(y_test, dtype=torch.long) # change to label mapping from xml

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

num_samples = len(trainset) 

client_sizes = [num_samples // num_clients] * num_clients
client_sizes[-1] += num_samples - sum(client_sizes)

client_datasets = random_split(trainset, client_sizes)

local_states, global_model = fl_training(num_rounds, local_epochs, batch_size, client_datasets, C, defense_function=None, fedtype=fedavg)

model = get_model()

# Create DataLoader for the smaller test subset
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

acc_model = evaluate_global(model, testloader)
acc_global_model = evaluate_global(global_model, testloader)

print(acc_model, acc_global_model)

# Save accuracies
acc_df = pd.DataFrame({
    f"Model{num_clients},{num_rounds},{local_epochs},{batch_size}:": ["Initial Model", "Global Model"],
    "Accuracy": [acc_model, acc_global_model]
})
acc_df.to_csv("model_accuracies.csv", mode='a', index=False)