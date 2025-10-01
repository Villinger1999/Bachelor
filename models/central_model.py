import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=500):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Fucntion for federated Averaging
def fedavg(states):
    avg = 0
    return avg

# Fucntion for federated SGD
def fedavg(states):
    sgd = 0
    return sgd

# Model for local training proces on each client
def local_train(model):
    """
    Args:
        num_rounds: number of training rounds
    
    Returns:
        Training weights for a FedAVG or gradients for FedSGD
    """
    model = copy.deepcopy(model) # copy of model that ensures that the original model is not changed
    grads = {name: torch.zeros_like(param, device=device) 
             for name, param in model.named_parameters()}
    return model.state_dict() # returns the training weights for FedAVG - can also be "return grads" for returning the gradients for FedSGD

# Function for the full Federated Learning proces 
def fl_training(num_rounds, local_epochs, batch_size, client_datasets, defense_function=None, fedtype=fedavg):
    """
    Args:
        num_rounds: number of training rounds
        local_epochs: number of local epochs
        batch_size: the number of training samples processed together
        client_datasets: an list of subsets with training data for each client
        defense_function: the defense function that should be used, defualt is None
        fedtype: type of federated learning update, defaults to federated averaging 
    
    Returns:
        The updated global model
    """
    global_model = get_model().to(device)

    for round in range(num_rounds): # for loop for number of training rounds
        print(f"Round {round+1}/{num_rounds}")

        local_states = []
        for i, client_dataset in enumerate(client_datasets): # for loop to simulate the training of each local client
            trainloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True) # load the clients data
            
            # Train local model
            local_state = local_train(global_model, trainloader, epochs=local_epochs, device=device) 
            local_states.append(local_state) 
            
            if defense_function != None: 
                defended_states = defense_function(local_states) # add defense, if applied
            print(f"Client {i+1} done.")

        # Aggregate
        if defense_function != None:
            global_state = fedtype(defended_states) 
            return defended_states, global_model.load_state_dict(global_state) # update global model 
        else:
            global_state = fedtype(local_states)
            return local_states, global_model.load_state_dict(global_state) # update global model 


def evaluate_global(model, dataloader, device='cpu'):
    """
    Args:
        model: the global model, that needs to be evaluated
        dataloader: Dataloader with the testset and batchsize (i.e. testloader = DataLoader(testset, batch_size=64, shuffle=False))
        device: Defaults to 'cpu' can also be GPU

    Returns:
        The accuracy of the global model
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total