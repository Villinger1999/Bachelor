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

# Take 10% of the total dataset
total_size = len(train)
subset_size = int(0.0001 * total_size)  # % of data
remaining_size = total_size - subset_size

# Randomly split 10% subset and discard the rest
subset, _ = random.sample(train, subset_size)

# Assume testset is a PyTorch Dataset (e.g., CIFAR10(test=True))
total_size = len(val)
subset_size = int(0.001 * total_size)     # % subset
remaining_size = total_size - subset_size

# Split: keep 10%, discard 90%
test_subset, _ = random.sample(val, subset_size)

x_train = []
y_train = []

for i in range(len(subset)):
    path = subset[i]
    img = Image.open(path).convert('RGB')
    x_train.append(img)
    
    # Get synset label from parent directory name
    synsets = sorted({os.path.basename(os.path.dirname(p)) for p in train})
    class_to_idx = {s: i for i, s in enumerate(synsets)}
    y_train.append(class_to_idx)
    
x_test = []
y_test = np.zeros(len(test_subset))

for i in range(len(test_subset)):
    path = test_subset[i]
    img = Image.open(path).convert('RGB')
    
    x_test.append(img)

    # synset = os.path.basename(os.path.dirname(path))
    # label = val[i].class_to_idx[synset]
    # y_test.append(label)
    
# Convert x_train/x_test to float tensors and normalize to [0, 1]
x_train_torch = torch.tensor(x_train.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0
x_test_torch  = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0

# Convert labels to long tensors and flatten
y_train_torch = torch.tensor(y_train.squeeze(), dtype=torch.long)
y_test_torch  = torch.tensor(y_test.squeeze(), dtype=torch.long)

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

# Now split that 10% subset among 3 clients
client_sizes = [subset_size // num_clients] * num_clients
client_sizes[-1] += subset_size - sum(client_sizes)

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