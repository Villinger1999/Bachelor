import os
import sys
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import pandas as pd
from models.central_model import *
from load_imnet import train, val  # assuming `train` is a list of image paths

num_clients = int(sys.argv[1])   # e.g. 3
num_rounds = int(sys.argv[2])    # e.g. 2
local_epochs = int(sys.argv[3])  # e.g. 1
batch_size = int(sys.argv[4])    # e.g. 24
C = float(sys.argv[5])           # e.g. 1.0

# Build label map from folder names
synsets = sorted({os.path.basename(os.path.dirname(p)) for p in train})
class_to_idx = {s: i for i, s in enumerate(synsets)}

# Take % of total dataset 
total_size = len(train)
subset_size = int(0.01 * total_size)  # 10% for train
train_subset_paths = random.sample(train, subset_size)
print("Train subset size:", len(train_subset_paths))

# test subset from same pool (1%)
test_subset_size = int(0.001 * total_size)
test_subset_paths = random.sample(train, test_subset_size)
print("Test subset size:", len(test_subset_paths))

# Transforms
img_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Lazy dataset
class ImagePathDataset(Dataset):
    def __init__(self, paths, class_to_idx, transform=None):
        self.paths = paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        synset = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[synset]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

# build full lazy datasets
train_dataset = ImagePathDataset(train_subset_paths, class_to_idx, img_tf)
test_dataset = ImagePathDataset(test_subset_paths, class_to_idx, img_tf)

# Split into client datasets
num_samples = len(train_dataset)

# simple equal split -> homogeneous in size
client_sizes = [num_samples // num_clients] * num_clients
client_sizes[-1] += num_samples - sum(client_sizes)

client_datasets = random_split(train_dataset, client_sizes)

# each client gets its own DataLoader that loads images on demand
client_loaders = [
    DataLoader(cd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    for cd in client_datasets
]

testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# FL training 
local_states, global_model = fl_training(
    num_rounds,
    local_epochs,
    batch_size,
    testloader,
    C,
    client_datasets,
    client_loader=client_loaders,
    defense_function=None,
    fedtype=fedavg
)

# Evaluate models
model = get_model()

acc_model = evaluate_global(model, testloader, device=device)
acc_global_model = evaluate_global(global_model, testloader, device=device)

# Save accuracies
acc_df = pd.DataFrame({
    f"Model{num_clients},{num_rounds},{local_epochs},{batch_size}:": ["Initial Model", "Global Model"],
    "Accuracy": [acc_model, acc_global_model]
})
acc_df.to_csv("model_accuracies.csv", mode='a', index=False)

# local fine-tune on whole train dataset, but still lazy/batched
full_trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
state = local_train(model, full_trainloader, testloader, epochs=local_epochs * num_clients, device=device, defense_function=None)
model.load_state_dict(state)
acc_resnet = evaluate_global(model, testloader, device=device)

print(f"Accuracy before training: {acc_model}, Accuracy after FL: {acc_global_model}, Accuracy ResNet: {acc_resnet}")
