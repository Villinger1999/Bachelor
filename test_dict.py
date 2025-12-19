from classes.models import LeNet
import torch
from classes.federated_learning import evaluate_global
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = LeNet()
model.load_state_dict(torch.load("state_dicts/state_dict_2_b64_e2.pt", map_location="cpu", weights_only=True))
model = model.to("cpu")

x_test_torch  = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0
y_test_torch  = torch.tensor(y_test.squeeze(), dtype=torch.long)
testset = TensorDataset(x_test_torch, y_test_torch)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

acc = evaluate_global(model, testloader, "cpu")
print("e2", acc)

model = LeNet()
model.load_state_dict(torch.load("state_dicts/state_dict_2_b64_e25.pt", map_location="cpu", weights_only=True))
model = model.to("cpu")

x_test_torch  = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255.0
y_test_torch  = torch.tensor(y_test.squeeze(), dtype=torch.long)
testset = TensorDataset(x_test_torch, y_test_torch)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

acc = evaluate_global(model, testloader, "cpu")
print("e25", acc)