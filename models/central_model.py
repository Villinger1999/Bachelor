import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import copy
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=1000):
    model = models.resnet18(weights='IMAGENET1K_V1') # 'IMAGENET1K_V1' or None
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Fucntion for federated Averaging
def fedavg(states, C, client_datasets):
    """
    states: a state dictionary of the local states from the trained models
    C: fraction of clients to participate
    client_dataset: a list of subsets with training data for each client
    """
    clients = list(range(len(client_datasets)))                                                 # total number of clients
    m = max(int(C * len(client_datasets)), 1)                                                   # number of clients to sample

    S_t = random.sample(clients, m)                                                             # random sampling of the clients
    local_states = []
    samples = []
    
    for i in S_t:
        local_states.append(states[i])                                                          # make list of the used clients state dicts
        samples.append(len(client_datasets[i]))                                                 # make list of number of samples in each client_dataset
        
    avg_state = {}
    total_samples = sum(samples)                                                        
    for key in local_states[0].keys():                                                          # Each key corresponds to a model parameter (weights, biases, etc.) - compute a weighted sum across clients
        avg_state[key] = sum(
            local_states[i][key] * (samples[i] / total_samples) for i in range(len(S_t))
        )
    return avg_state

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
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)                  # set stochastic gradient decent as optimizer function

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
        for i, client_dataset in enumerate(client_datasets):                                    # for loop to simulate the training of each local client
            trainloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)       # load the clients data
            
            # Train local model
            local_state = local_train(global_model, trainloader, epochs=local_epochs, device=device, defense_function=None) 
            local_states.append(local_state)                 
            
            if round == (num_rounds-1):
                # Save local_states
                try:
                    torch.save(local_state, f"state_dicts/local_state_client{i}{str(time.time)}.pt")
                except Exception as e:
                    print("Error saving local_state:", e)
                    
            print(f"Client {i+1} done.")

        # Add defense, if applied
        if defense_function != None: 
            local_states = defense_function(local_states)                                  

        global_state = fedtype(local_states, C, client_datasets)                                # update the weights using fedavg
        global_model.load_state_dict(global_state)                                              # update global model with updated weigths
    
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
