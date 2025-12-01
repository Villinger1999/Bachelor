from models.central_model import *
from test_train_noiseadd import x_train, y_train, x_test, y_test
import sys
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import torch
import pandas as pd
from models.lenet import LeNet
from collections import defaultdict
from models.central_model import get_model
import os
import torchvision.transforms.functional as TF

path = os.getcwd() + "/"  

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

# img_pil = TF.to_pil_image(x_train_torch[0])
# img_pil.save("test.png")

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

# Use only the first sample in trainset
single_trainset = Subset(trainset, [0])
single_testset = Subset(testset, [0])

# For FL: one client with that single-sample dataset
client_datasets = [single_trainset]

# Now split that 10% subset among 3 clients
# client_sizes = [len(trainset) // num_clients] * num_clients
# client_sizes[-1] += len(trainset) - sum(client_sizes)

# Create Subset datasets for each client
# client_datasets = random_split(trainset, client_sizes)

# Create DataLoader for the smaller test subset
testloader = DataLoader(single_testset, batch_size=batch_size, shuffle=False)
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

model = LeNet()
model.load_state_dict(torch.load("state_dict_2_b64_e2.pt", map_location="cpu", weights_only= True))

acc_model = evaluate_global(model, testloader)

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