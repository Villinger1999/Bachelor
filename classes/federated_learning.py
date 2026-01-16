import random
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from classes.defenses import *


def fedavg(states, C, client_datasets):
    """
    states: list of state_dict from each client after local training
    C: fraction of clients to participate
    client_datasets: list of datasets/subsets (used for weighting)
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
    Return a dict keyed by parameter name with gradient tensors.
    Includes None if a parameter has no gradient.
    """
    grad_dict = {}
    for name, p in model.named_parameters():
        grad_dict[name] = None if p.grad is None else p.grad.detach().clone()
    return grad_dict


@torch.no_grad()
def load_grad_dict_into_model(model, grad_dict):
    """
    Writes gradient tensors from grad_dict back into model parameters' .grad
    so that optimizer.step() uses the defended gradients.
    """
    for name, p in model.named_parameters():
        g = grad_dict.get(name, None)
        if g is None:
            continue
        if p.grad is None:
            continue
        p.grad.copy_(g.to(device=p.grad.device, dtype=p.grad.dtype))


def apply_defense_to_grad_dict(grad_dict, defense=None, *, percentile=0.9, keep_ratio=0.9):
    """
    Returns a new dict with defended grads (same keys).
    Uses Defense classes (NormClipping/Clipping/SGP) which operate on list[tensor].
    Preserves None entries.
    """
    if defense is None:
        return grad_dict

    keys = list(grad_dict.keys())
    grads = [grad_dict[k] for k in keys]

    # indices of non-None grads
    idx = [i for i, g in enumerate(grads) if g is not None]
    if not idx:
        return grad_dict

    grads_nonnull = [grads[i] for i in idx]

    if defense == "clipping":
        thr = clipping_threshold(grads_nonnull, percentile=float(percentile))
        defended_nonnull = Clipping(threshold=thr).apply(grads_nonnull)

    elif defense == "sgp":
        thr = pruning_threshold(grads_nonnull, keep_ratio=float(keep_ratio))
        defended_nonnull = SGP(threshold=thr).apply(grads_nonnull)

    else:
        raise ValueError(f"Unknown defense: {defense}")

    grads_out = list(grads)
    for j, i in enumerate(idx):
        grads_out[i] = defended_nonnull[j]

    return {k: grads_out[i] for i, k in enumerate(keys)}


def evaluate_global(model, dataloader, device):
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


class Client:
    def __init__(
        self,
        client_id,
        dataset,
        batch_size,
        device="cpu",
        defense=None,
        percentile=0.9,
        keep_ratio=0.9,
    ):
        self.id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.defense = defense
        self.percentile = percentile
        self.keep_ratio = keep_ratio

    def train_local(self, global_model, testloader, epochs=1, lr=0.01, defense=None,
                    percentile=None, keep_ratio=None):
        """
        Local training (FedAvg style): returns updated local model weights.

        Also returns grads_dict containing the *LAST BATCH* defended gradients (and labels)
        for attack logging/saving purposes.
        """
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        trainloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Resolve defense settings: prefer call args, otherwise client defaults
        use_defense = defense if defense is not None else self.defense
        use_percentile = float(percentile) if percentile is not None else float(self.percentile)
        use_keep_ratio = float(keep_ratio) if keep_ratio is not None else float(self.keep_ratio)

        captured_grads = None
        captured_labels = None
        state_for_attack = None

        for epoch in range(epochs):
            running_loss, num_steps = 0.0, 0

            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Capture the gradients from this (last-seen) batch
                raw_grads = grad_state_dict(local_model)

                # Apply defense to gradients
                defended_grads = apply_defense_to_grad_dict(
                    raw_grads,
                    defense=use_defense,
                    percentile=use_percentile,
                    keep_ratio=use_keep_ratio,
                )

                # IMPORTANT: write defended grads back into param.grad
                load_grad_dict_into_model(local_model, defended_grads)

                # Save defended grads + labels + state for attack from last batch
                captured_grads = defended_grads
                captured_labels = labels.detach().cpu().clone()
                state_for_attack = copy.deepcopy(local_model.state_dict())

                optimizer.step()

                running_loss += loss.item()
                num_steps += 1

            avg_loss = running_loss / max(1, num_steps)
            acc = evaluate_global(local_model, testloader, self.device)
            print(f"[Client {self.id}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

        grads_dict = {
            "grads_per_sample": captured_grads,      # defended grads from last batch
            "labels_per_sample": captured_labels,    # labels from last batch
            "model_state": state_for_attack,         # state right before last optimizer.step()
            "defense": use_defense,
            "percentile": use_percentile,
            "keep_ratio": use_keep_ratio,
        }
        return local_model.state_dict(), grads_dict

class FederatedTrainer:
    """
    Federated training with FedAvg weight aggregation.
    Defense is applied on clients *before optimizer.step()* so it impacts training.
    """

    def __init__(self, global_model, clients, testloader, C,
                 device="cpu", aggregator=fedavg):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.testloader = testloader
        self.C = C
        self.device = device
        self.aggregator = aggregator

    def train(self, num_rounds, local_epochs, defense=None, save_grads=False, run_id=None,
              percentile=0.9, keep_ratio=0.9, lr=0.01):
        last_local_states = None

        for rnd in range(num_rounds):
            print(f"\n=== Round {rnd+1}/{num_rounds} ===")

            local_states = []
            all_grads = []

            for client in self.clients:
                print(f"Client {client.id} training...")
                local_state, local_grads = client.train_local(
                    self.global_model,
                    self.testloader,
                    epochs=local_epochs,
                    lr=lr,
                    defense=defense,
                    percentile=percentile,
                    keep_ratio=keep_ratio,
                )

                local_states.append(local_state)
                all_grads.append(local_grads)

                if save_grads and rnd == (num_rounds - 1):
                    try:
                        torch.save(local_grads, f"state_dicts/local_grads_client{client.id}_{run_id}.pt")
                    except Exception as e:
                        print("Error saving local_grads:", e)

            # Aggregate weights (FedAvg)
            client_datasets = [c.dataset for c in self.clients]
            global_state = self.aggregator(local_states, self.C, client_datasets)
            self.global_model.load_state_dict(global_state)

            acc = evaluate_global(self.global_model, self.testloader, self.device)
            print(f"Global eval after round {rnd+1}: {acc:.4f}")

            last_local_states = local_states

        # Save final global model
        try:
            torch.save(self.global_model.state_dict(), f"state_dicts/global_model_state_{run_id}_{sys.argv[1]}.pt")
        except Exception as e:
            print("Error saving global model:", e)

        return last_local_states, self.global_model
