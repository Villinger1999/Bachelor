import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import copy
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=500):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Fucntion for federated Averaging
def fedavg(states, C, client_datasets):
    """
    states: a state dictionary of the local states from the trained models
    C: fraction of clients to participate
    client_dataset: a list of subsets with training data for each client
    """
    clients = list(range(len(client_datasets)))      # total number of clients
    m = max(int(C * len(client_datasets)), 1)        # number of clients to sample

    S_t = random.sample(clients, m)
    local_states = []
    samples = []
    
    for i in S_t:
        local_states.append(states[i])
        samples.append(len(client_datasets[i]))
        
    avg_state = {}
    total_samples = sum(samples)
    for key in local_states[0].keys():
        avg_state[key] = sum(
            local_states[i][key] * (samples[i] / total_samples) for i in range(len(S_t))
        )
    return avg_state

# Model for local training proces on each client
def local_train(model, trainloader, epochs=1, device=device, defense_function=None):
    """
    Args:
        model: global_model
        trainloader: DataLoader that loads the clients data 
        epochs: default = 1
        device: gpu or cpu
    
    Returns:
        Training weights for a FedAVG or gradients for FedSGD
    """
    local_model = copy.deepcopy(model) # copy of model that ensures that the original model is not changed
    return local_model.state_dict() # 

# Function for simulation of the full Federated Learning proces 
def fl_training(num_rounds, local_epochs, batch_size, client_datasets, C, defense_function=None, fedtype=fedavg):
    """
    Args:
        num_rounds: number of training rounds
        local_epochs: number of local epochs
        batch_size: the number of training samples processed together
        client_datasets: a list of subsets with training data for each client
        C: fraction of clients to participate
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
            local_state = local_train(global_model, trainloader, epochs=local_epochs, device=device, defense_function=None) 
            local_states.append(local_state) 
            print(f"Client {i+1} done.")

        # Apply defense
        if defense_function != None: 
            local_states = defense_function(local_states) # add defense, if applied

        global_state = fedtype(local_states, C, client_datasets) # aggregate the local weights using federated aver
        global_model.load_state_dict(global_state) # update global model 
    
    return local_states, global_model

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