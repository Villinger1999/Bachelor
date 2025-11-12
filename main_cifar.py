from models.central_model import *
from test_train_noiseadd import x_train, y_train, x_test, y_test
from torch.utils.data import random_split
import sys
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import pandas as pd
from models.lenet import LeNet
from collections import defaultdict
from models.central_model import get_model

num_clients = int(sys.argv[1]) # e.g 3
num_rounds = int(sys.argv[2]) # e.g 2
local_epochs = int(sys.argv[3]) # e.g 1
batch_size = int(sys.argv[4]) # e.g 24
C = int(sys.argv[5]) # e.g 1

# Convert x_train/x_test to float tensors and normalize to [0, 1]
x_train_torch = torch.tensor(x_train.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0
x_test_torch  = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0

# Convert labels to long tensors and flatten
y_train_torch = torch.tensor(y_train.squeeze(), dtype=torch.long)
y_test_torch  = torch.tensor(y_test.squeeze(), dtype=torch.long)

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

# Take 10% of the total dataset
total_size = len(trainset)
subset_size = int(0.5 * total_size)  # % of data
remaining_size = total_size - subset_size

# Randomly split 10% subset and discard the rest
subset, _ = random_split(trainset, [subset_size, remaining_size])

# Get subset indices and labels
subset_indices = subset.indices
subset_labels = y_train_torch[subset_indices].cpu().numpy()

# Group indices by label
label_to_indices = defaultdict(list)
for idx, label in zip(subset_indices, subset_labels):
        label_to_indices[int(label)].append(int(idx))

# Distribute: each client gets 2 images per label (round-robin)
client_indices = [[] for _ in range(num_clients)]
for label, indices in label_to_indices.items():
    random.shuffle(indices)
    for i, global_idx in enumerate(indices): 
        client_indices[i % num_clients].append(global_idx)

# Create Subset datasets for each client
client_datasets = [torch.utils.data.Subset(trainset, inds) for inds in client_indices]
print(f"Client dataset sizes: {[len(c) for c in client_datasets]}")

# Assume testset is a PyTorch Dataset (e.g., CIFAR10(test=True))
total_size = len(testset)
subset_size = int(20)     # % subset
remaining_size = total_size - subset_size

# Split: keep 10%, discard 90%
test_subset, _ = random_split(testset, [subset_size, remaining_size])

# Create DataLoader for the smaller test subset
testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

model = LeNet()
# model = get_model()

local_states, global_model = fl_training(
    model,
    num_rounds, 
    local_epochs, 
    batch_size, 
    testloader, 
    C, 
    client_datasets, 
    client_loader=None,
    defense_function=None, 
    fedtype=fedavg
)

model = LeNet()
# model = get_model()

acc_model = evaluate_global(model, testloader)
acc_global_model = evaluate_global(global_model, testloader)

# Save accuracies
acc_df = pd.DataFrame({
    f"Model{num_clients},{num_rounds},{local_epochs},{batch_size}:": ["Initial Model", "Global Model"],
    "Accuracy": [acc_model, acc_global_model]
})
acc_df.to_csv("model_accuracies.csv", mode='a', index=False)

# state = local_train(model, trainloader, testloader, epochs=local_epochs*num_rounds, device=device, defense_function=None) 
# model.load_state_dict(state)
# acc_resnet = evaluate_global(model, testloader, device=device)

print(f"Accuracy before training: {acc_model}, Accurracy after FL: {acc_global_model}") # Accuracy ResNet: {acc_resnet}