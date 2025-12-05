from classes.federated_learning import Client
from classes.models import LeNet
from torch.utils.data import DataLoader, TensorDataset
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

# Create DataLoader for the smaller test subset
testloader = DataLoader(testset, batch_size=64, shuffle=False)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

model = LeNet()

state_dict, _ = Client.local_train(model, trainloader, testloader, epochs=100, device="cpu", lr=0.01, defense_function=None)

torch.save(state_dict, f"state_dict_{str(sys.argv[1])}.pt")