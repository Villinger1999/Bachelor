import random
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

def fedavg(states, C, client_datasets):
    """
    states: a state dictionary of the local states from the trained models
    C: fraction of clients to participate
    client_dataset: a list of subsets with training data for each client
    """
    clients = list(range(len(client_datasets)))
    m = max(int(C * len(client_datasets)), 1)

    S_t = random.sample(clients, m)
    local_states, samples = [], []

    for i in S_t:
        local_states.append(states[i])
        samples.append(len(client_datasets[i]))

    total_samples = sum(samples)
    avg_state = {}
    for key in local_states[0].keys():
        avg_state[key] = sum(
            local_states[i][key] * (samples[i] / total_samples)
            for i in range(len(S_t))
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


def evaluate_global(model, dataloader, device):
    """
    Args:
        model: the global model, that needs to be evaluated
        dataloader: Dataloader with the testset and batchsize (i.e. testloader = DataLoader(testset, batch_size=64, shuffle=False))
        device: Defaults to 'cpu' can also be GPU

    Returns:
        The accuracy of the global model
    """
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0

def train(model, trainloader, testloader, epochs=100, lr=0.1, device="cpu", defense = 0):
    local_model = copy.deepcopy(model).to(device)
    local_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        running_loss, num_steps = 0.0, 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            num_steps += 1

        avg_loss = running_loss / max(1, num_steps)
        acc = evaluate_global(local_model, testloader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
    state = model.state_dict()
    return state


class Client:
    def __init__(self, client_id, dataset, batch_size, device="cpu"):
        self.id = client_id
        self.dataset = dataset                # used for weighting in FedAvg
        self.batch_size = batch_size
        self.device = device

    def train_local(self, global_model, testloader, epochs=1, lr=0.01, defense=None):
        """
        Performs local training on a client using its local dataset
        
        Args:
            

        Returns:
            state_dict: updated model weights after local training
            grads_dict: dict containing a list of per-step gradients + labels
        """
        # copy global model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)

        trainloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        captured_grads = None
        captured_labels = None
        state_for_attack = None

        for epoch in range(epochs):
            running_loss, num_steps = 0.0, 0

            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # capture gradients/state from last batch
                captured_grads = grad_state_dict(local_model)
                captured_labels = labels.detach().cpu().clone()
                state_for_attack = copy.deepcopy(local_model.state_dict())

                # apply defense on grads if given (your Clipping/SGP/PLGP can go here)
                if defense is not None:
                    captured_grads = defense.apply(captured_grads)

                optimizer.step()
                running_loss += loss.item()
                num_steps += 1

            avg_loss = running_loss / max(1, num_steps)
            acc = evaluate_global(local_model, testloader, self.device)
            print(f"[Client {self.id}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

        grads_dict = {
            "grads_per_sample": captured_grads,
            "labels_per_sample": captured_labels,
            "model_state": state_for_attack,
        }
        return local_model.state_dict(), grads_dict


class FederatedTrainer:
    """
    Args:
        global_model: 
        clients: 
        testloader:
        C: fraction of clients to participate
        device: defaults to cpu (because of iDLG)
        aggregator: type of federated learning update, defaults to federated averaging 
    
    Returns:
        The updated global model
    """
    def __init__(self, global_model, clients, testloader, C,
                 device="cpu", aggregator=fedavg):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.testloader = testloader
        self.C = C
        self.device = device
        self.aggregator = aggregator

    def train(self, num_rounds, local_epochs, defense=None, save_grads=False, run_id=None):
        """
        Args:
            num_rounds: number of training rounds
            local_epochs: number of local epochs
            defense: the defense function that should be used, defualt is None
            save_grads: defaults to False, has to be True if the gradients should be saved
            run_id:
        
        Returns: 
            local_states:
            global_model:
        """
        last_local_states = None

        for rnd in range(num_rounds):
            print(f"\n=== Round {rnd+1}/{num_rounds} ===")

            local_states = []
            all_grads = []

            # you could also subsample clients here based on C instead of in fedavg
            for client in self.clients:
                print(f"Client {client.id} training...")
                local_state, local_grads = client.train_local(
                    self.global_model,
                    self.testloader,
                    epochs=local_epochs,
                    lr=0.01,
                    defense=defense,
                )
                local_states.append(local_state)
                all_grads.append(local_grads)

                if save_grads and rnd == (num_rounds - 1):
                    try:
                        torch.save(local_grads, f"state_dicts/local_grads_client{client.id}_{run_id}.pt")
                    except Exception as e:
                        print("Error saving local_grads:", e)

            # aggregation
            client_datasets = [c.dataset for c in self.clients]
            global_state = self.aggregator(local_states, self.C, client_datasets)
            self.global_model.load_state_dict(global_state)

            acc = evaluate_global(self.global_model, self.testloader, self.device)
            print(f"Global eval after round {rnd+1}: {acc:.4f}")

            last_local_states = local_states
            
        torch.save(self.global_model.state_dict(), f"state_dicts/global_model_state_{run_id}_{sys.argv[1]}.pt")
        torch.save(global_state, f"state_dicts/global_state_{run_id}_{sys.argv[1]}.pt")
        return last_local_states, self.global_model
