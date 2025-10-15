# to use flower open terminal and write:
# flwr new flower-tutorial --framework pytorch --username flwrlabs
# then cd into flower-tutorial & pip install -e .

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=500):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def local_train(model, trainloader, epochs=1, device=device, lr=0.01, defense_function=None):
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
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)                               # Move both inputs and labels to GPU (or CPU), matching model device

            optimizer.zero_grad()                                                               # Clear old gradients from the previous step
            outputs = local_model(images)                                                       # send the images through the model
            loss = criterion(outputs, labels)                                                   # Compare predictions to labels using CrossEntropyLoss
            loss.backward()                                                                     # Compute gradients of the loss for each weight using autograd and store them in param.grad for each parameter
            optimizer.step()                                                                    # optimize the weights using SGD

            running_loss += loss.item()                                                         # Accumulates total loss to compute the average later

        avg_loss = running_loss / len(trainloader)                                              # calculate average loss
        print(f"Local Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Return the trained model weights
    return local_model.state_dict()

def evaluate_global(model, dataloader, device='cpu'):
    """
    Args:
        model: the global model, that needs to be evaluated
        dataloader: Dataloader with the testset and batchsize (i.e. testloader = DataLoader(testset, batch_size=64, shuffle=False))
        device: Defaults to 'cpu' can also be GPU

    Returns:
        The accuracy of the global model
    """
    model.eval()                                                                                # Switch model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                                                             # Get model predictions
            _, predicted = torch.max(outputs, 1)                                                # finds the maximum value along dimension 1 (the class dimension)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                                       # creates a tensor of True/False values for correct predictions, sums and converts that count from a tensor to a Python number.
    return correct / total                                                                      # calculates and returns accuracy

# turn the models state_dict tensors into numpy arrays
def parameters_to_list(model):                                                                  
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# turn numpy arrays from flower into tensors to use as state_dict
def list_to_parameters(model, params_list):
    state_dict = model.state_dict()
    new = {}
    for (k, v), arr in zip(state_dict.items(), params_list):
        new[k] = torch.from_numpy(arr).to(v.device)
    model.load_state_dict(new)
