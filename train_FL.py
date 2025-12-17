from classes.federated_learning import Client, FederatedTrainer, fedavg
from classes.models import LeNet
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import tensorflow as tf
import sys

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert x_train/x_test to float tensors and normalize to [0, 1]
x_train_torch = torch.tensor(x_train.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0
x_test_torch  = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0

# Convert labels to long tensors and flatten
y_train_torch = torch.tensor(y_train.squeeze(), dtype=torch.long)
y_test_torch  = torch.tensor(y_test.squeeze(), dtype=torch.long)

trainset = TensorDataset(x_train_torch, y_train_torch)
testset = TensorDataset(x_test_torch, y_test_torch)

model = LeNet()

num_clients = 6
batch_size = 64

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# split trainset across clients
num_samples = len(trainset)
client_sizes = [num_samples // num_clients] * num_clients
client_sizes[-1] += num_samples - sum(client_sizes)

client_datasets = random_split(trainset, client_sizes)

clients = []
device = "cuda" if torch.cuda.is_available() else "cpu"

for i, ds in enumerate(client_datasets):
    clients.append(
        Client(
            client_id=i,
            dataset=ds,
            batch_size=batch_size,
            device=device
        )
    )


global_model = LeNet().to(device)

trainer = FederatedTrainer(
    global_model=global_model,
    clients=clients,
    testloader=testloader,
    C=1.0,                 # fraction of clients
    device=device,
    aggregator=fedavg
)

last_states, trained_global_model = trainer.train(
    num_rounds=8,
    local_epochs=10,
    defense=None,          
    save_grads=True,       
    run_id="exp1"
)