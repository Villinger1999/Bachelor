import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import copy
import random
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=10):
    model = models.resnet18(weights=None) # 'IMAGENET1K_V1' or None
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

def grad_state_dict(model):
    """
    Return a dict with the same keys as model.state_dict() but values 
    are the corresponding gradients (detached & cloned). If a parameter
    has no grad, it is skipped.
    """
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_dict[name] = param.grad.detach().clone()
    return grad_dict

def normalize_keys_strip_module(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new = {}
    for k, v in d.items():
        if k.startswith("module."):
            new_k = k.replace("module.", "", 1)
        else:
            new_k = k
        new[new_k] = v
    return new


def local_train(model, trainloader, testloader, epochs=1, device="cpu", lr=0.01, defense_function=None):
    """
    Performs local training on a client using its local dataset, but:
    - Even if trainloader.batch_size > 1, we do *per-sample* SGD steps.
    - For each sample/step we capture and store the gradient dict.

    Returns:
        state_dict: updated model weights after local training
        grads_dict: dict containing a list of per-step gradients + labels
    """
    # Copy the global model so we don't modify it in-place
    local_model = copy.deepcopy(model).to(device)
    local_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)

    # Store gradients and labels for *every* step
    all_grads = []      # list of dicts (one dict per step)
    all_labels = []     # list of tensors, each shape [1]

    for epoch in range(epochs):
        running_loss = 0.0
        num_steps = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device)      # shape [B, C, H, W]
            labels = labels.to(device)      # shape [B]

            batch_size = images.size(0)

            # Loop over samples inside the batch
            for i in range(batch_size):
                x_i = images[i:i+1]        # keep batch-dim: [1, C, H, W]
                y_i = labels[i:i+1]        # shape [1]

                optimizer.zero_grad()
                outputs = local_model(x_i)
                loss = criterion(outputs, y_i)
                loss.backward()

                # Optionally apply a defense function on gradients *before* capturing
                if defense_function is not None:
                    defense_function(local_model)

                # Capture per-step gradients
                grad_dict = grad_state_dict(local_model)
                grad_dict = normalize_keys_strip_module(grad_dict)

                all_grads.append(grad_dict)
                all_labels.append(y_i.detach().cpu().clone())

                # One SGD step per sample
                optimizer.step()

                running_loss += loss.item()
                num_steps += 1

        # Epoch summary
        avg_loss = running_loss / max(1, num_steps)
        acc = evaluate_global(local_model, testloader, device)
        local_model.train()
        print(f"Local Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Acc: {acc}")

    # Package gradients + labels
    grads_dict = {
        "grads_per_step": all_grads,          # list[dict[str, Tensor]]
        "labels_per_step": all_labels         # list[Tensor of shape [1]]
    }

    return local_model.state_dict(), grads_dict


# Function for simulation of the full Federated Learning proces 
def fl_training(model, num_rounds, local_epochs, batch_size, testloader, C, client_datasets, client_loader=None,defense_function=None, fedtype=fedavg):
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
    global_model = model.to(device)

    for round in range(num_rounds): # for loop for number of training rounds
        print(f"Round {round+1}/{num_rounds}")

        local_states = []
        if client_loader != None:
            for i, client_data in enumerate(client_loader):  # but now it's actually loaders
                trainloader = client_data
                # now client_data IS a DataLoader, don't wrap it again
                local_state, local_grads = local_train(
                    global_model,
                    trainloader,
                    testloader,
                    epochs=local_epochs,
                    device=device,
                    defense_function=None
                )
                local_states.append(local_state)
                # Add defense, if applied
                if defense_function != None: 
                    local_grads = defense_function(local_grads)
                    local_states = defense_function(local_states)                    
                
                if round == (num_rounds-1):
                    # Save local_states
                    try:
                        torch.save(local_grads, f"state_dicts/local_grads_client{i}_{str(sys.argv[6])}.pt")
                        torch.save(local_state, f"state_dicts/local_state_client{i}_{str(sys.argv[6])}.pt") # {time.time()}
                    except Exception as e:
                        print("Error saving local_state:", e)
                                                        
                print(f"Client {i+1} done.")
        else:
            for i, client_data in enumerate(client_datasets):  # but now it's actually loaders
                trainloader = DataLoader(client_data, batch_size=batch_size, shuffle=True)       # load the clients data
                # now client_data IS a DataLoader, don't wrap it again

                local_state, local_grads = local_train(
                    global_model,
                    trainloader,
                    testloader,
                    epochs=local_epochs,
                    device=device,
                    defense_function=None
                )
                local_states.append(local_state)
                
                # Add defense, if applied
                if defense_function != None: 
                    local_grads = defense_function(local_grads)
                    local_states = defense_function(local_states)     

                if round == (num_rounds-1):
                    # Save local_states
                    try:
                        torch.save(local_grads, f"state_dicts/local_grads_client{i}_{str(sys.argv[6])}.pt")
                        torch.save(local_state, f"state_dicts/local_state_client{i}_{str(sys.argv[6])}.pt")
                    except Exception as e:
                        print("Error saving local_state:", e)
                                                        
                print(f"Client {i+1} done.")                             

        global_state = fedtype(local_states, C, client_datasets)                                # update the weights using fedavg
        global_model.load_state_dict(global_state)                                              # update global model with updated weigths
        eval = evaluate_global(global_model, testloader, device)
        print(f"eval:{eval}")
    
    return local_states, global_model

def evaluate_global(model, dataloader, device=device):
    """
    Args:
        model: the global model, that needs to be evaluated
        dataloader: Dataloader with the testset and batchsize (i.e. testloader = DataLoader(testset, batch_size=64, shuffle=False))
        device: Defaults to 'cpu' can also be GPU

    Returns:
        The accuracy of the global model
    """
    model = model.to(device)
    model.eval() # Switch model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                                                             # Get model predictions
            _, predicted = torch.max(outputs, 1)                                                # finds the maximum value along dimension 1 (the class dimension)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                                       # creates a tensor of True/False values for correct predictions, sums and converts that count from a tensor to a Python number.
    return correct / total                                                                      # calculates and returns accuracy
