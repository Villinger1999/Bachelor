import torch
from classes.defenses import *
from classes.models import LeNet
from classes.federated_learning import evaluate_global
import tensorflow as tf
from classes.helperfunctions import *
from torch.utils.data import DataLoader, TensorDataset

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

# model = LeNet()
# model.load_state_dict(torch.load("state_dicts/state_dict_model_b64_e150.pt", map_location="cpu", weights_only=True))
# # model.load_state_dict(torch.load("state_dicts/global_state_exp1_c6_b64_e10_FL.pt", map_location=device, weights_only=True))
# model = model.to("cpu")

# acc_before = evaluate_global(model, testloader, "cpu")
# print(f"Acc of the model before defense: {acc_before}")