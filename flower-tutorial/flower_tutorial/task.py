# to use flower open terminal
# cd into flower-tutorial 
# pip install -e .

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=500):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(model, trainloader, epochs=1, device=device, lr=0.01, defense_function=None):
    """
    Performs local training on a client using its local dataset.

    Args:
        model: global model (to be copied for local training)
        trainloader: DataLoader with the client's local dataset
        epochs: number of local epochs, default is set to 1
        device: 'cuda' or 'cpu'
        lr: local learning rate, default is set to 0.01
        defense_function: optional; for gradient manipulation, default is None

    Returns:
        state_dict: the updated model weights after local training
    """
    # Create a copy so the global model isn’t modified in-place
    local_model = copy.deepcopy(model).to(device)                                               # ensure each client trains on it's own model
    local_model.train() 

    criterion = torch.nn.CrossEntropyLoss()                                                     # objective function
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)                  # set stochastic gradient decent as optimizer function

    # Local training loop
    for epoch in range(epochs):
        running_loss = 0.0
        # for batch_idx, (images, labels) in enumerate(trainloader):
        #     images, labels = images.to(device), labels.to(device)                               # Move both inputs and labels to GPU (or CPU), matching model device
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)


            optimizer.zero_grad()                                                               # Clear old gradients from the previous step
            outputs = local_model(images)                                                       # send the images through the model
            loss = criterion(outputs, labels)                                                   # Compare predictions to labels using CrossEntropyLoss
            loss.backward()                                                                     # Compute gradients of the loss for each weight using autograd and store them in param.grad for each parameter
            optimizer.step()                                                                    # optimize the weights using SGD

            running_loss += loss.item()                                                         # Accumulates total loss to compute the average later

        avg_loss = running_loss / len(trainloader)                                              # calculate average loss
        print(f"Local Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Return the trained model weights
    return avg_loss, local_model.state_dict()

def evaluate(model, dataloader, device='cpu'):
    """
    Args:
        model: the global model, that needs to be evaluated
        dataloader: Dataloader with the testset and batchsize (i.e. testloader = DataLoader(testset, batch_size=64, shuffle=False))
        device: Defaults to 'cpu' can also be GPU

    Returns:
        The accuracy of the global model
    """
    model.eval()                                                                                # Switch model to evaluation mode
    correct, total, running_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # for images, labels in dataloader:
        #     images, labels = images.to(device), labels.to(device)
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)                                                             # Get model predictions
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)                                                # finds the maximum value along dimension 1 (the class dimension)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                                       # creates a tensor of True/False values for correct predictions, sums and converts that count from a tensor to a Python number.
    avg_loss = running_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc                                                                     # calculates and returns accuracy


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

## to run $ flwr run . --run-config "num-server-rounds=5 local-epochs=3"