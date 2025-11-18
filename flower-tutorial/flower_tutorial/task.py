"""flower-tutorial: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.optim.lbfgs import LBFGS

def infer_labels_from_bias_grad(leaked_grads:dict[str, torch.Tensor], model: torch.nn.Module) -> int:
    """
    iDLG uses the fact that when cross-entropy is used with softmax, the correct label will always have a negative sign.
    iDLG: label = argmin(grad w.r.t. last-layer bias) because that bias grad equals (p - one_hot).
    
    Note: This method can only infer a SINGLE label, even for batches. The assumption is that all images 
    in the batch belong to the same class (homogeneous batch).

    Args:
        leaked_grads (dict[str, torch.Tensor]): the leaked gradients
        model (torch.nn.Module): the model used

    Returns:
        int: index of the inferred label (same for all images in batch)
    """
    
    if isinstance(leaked_grads, dict) and 'grads' in leaked_grads:
       leaked_grads = leaked_grads['grads']
    
    # find the name of the last bias
    for name, parameter in model.named_parameters(): #loop through the names and parameters in the model 
        if name.endswith(".bias") and parameter.ndim == 1: # if it's a bias parameter and it's 1 dimentional. 
            last_bias_name = name # set is as the name of the last bias term
    
    # bias_grad is the Gradient of loss w.r.t. logits (g_i in equation 3 in iDLG paper)
    bias_grad = leaked_grads[last_bias_name] # Bias gradient equals g_i
    true_label = int(torch.argmin(bias_grad).item())
    
    return true_label # True label = index of minimum gradient

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    # trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    # iDLG
    trainloader = DataLoader(partition_train_test["train"], batch_size=2)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device, return_leaked_grads=False):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    running_loss = 0.0
    num_batches = 0
    leaked_grads = None
    original_image = None
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Save original image when leaking gradients
            if return_leaked_grads and leaked_grads is None:
                leaked_grads = {
                    name: p.grad.detach().cpu().clone()
                    for name, p in net.named_parameters()
                }
                
                # Save the REAL IMAGE (still normalized)
                original_image = images[0].detach().cpu()
                print("Labels:", labels.tolist())   # prints both, e.g. [3, 5]
                print("fc3.bias grad:", leaked_grads["fc3.bias"])
                print("fc3.bias grad norm:", leaked_grads["fc3.bias"].norm().item())
                print("Inferred label (iDLG):", infer_labels_from_bias_grad(leaked_grads, net))
                print("True labels (first sample):", labels[0].item(), labels[1].item())

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    avg_trainloss = running_loss / num_batches

    if return_leaked_grads:
        return avg_trainloss, leaked_grads, original_image
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def attack(model, leaked_grads_dict, x_shape, device,
           train_ite=200, clamp=(-1.0, 1.0), seed=123):

    model = model.to(device)
    model.eval()

    # --- order gradients correctly ---
    param_names, params = zip(*model.named_parameters())

    orig_grads = []
    for name in param_names:
        assert name in leaked_grads_dict, f"Missing gradient for {name}"
        orig_grads.append(leaked_grads_dict[name].to(device))

    # --- infer label ---
    inferred_label = infer_labels_from_bias_grad(leaked_grads_dict, model)
    B = x_shape[0]
    label_pred = torch.full((B,), inferred_label, device=device, dtype=torch.long)

    # --- dummy init ---
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    param_dtype = next(model.parameters()).dtype
    dummy_data = torch.randn(x_shape, dtype=param_dtype, device=device)
    dummy_data = dummy_data.clamp(clamp[0], clamp[1]).requires_grad_(True)

    # --- optimizer ---
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = LBFGS(
        [dummy_data], lr=0.1, max_iter=50,
        tolerance_grad=1e-9, tolerance_change=1e-11,
        history_size=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        out = model(dummy_data)
        loss = criterion(out, label_pred)
        grad_list = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(grad_list, orig_grads):
            diff = diff + ((gx - gy) ** 2).sum()

        diff.backward()
        return diff

    for _ in range(train_ite):
        optimizer.step(closure)
        with torch.no_grad():
            dummy_data.clamp_(clamp[0], clamp[1])

    return dummy_data.detach(), inferred_label



























# def attack(model, leaked_grads_dict, x_shape, device,
#            train_ite: int = 200, clamp: tuple[float, float] = (-1.0, 1.0),
#            seed: int | None = 123):
#     """
#     iDLG-style reconstruction using LBFGS, adapted to your setup.
#     - model: Net() with correct weights loaded
#     - leaked_grads_dict: {name -> grad tensor} from train()
#     - x_shape: (B, C, H, W), for CIFAR-10 use (1, 3, 32, 32)
#     """

#     model = model.to(device)
#     model.eval()

#     # --- align grads to parameter order (list) ---
#     param_names, params = zip(*model.named_parameters())
#     param_names, params = zip(*model.named_parameters())
#     orig_grads = []

#     for name in param_names:
#         assert name in leaked_grads_dict, f"Missing grad for {name}"
#         orig_grads.append(leaked_grads_dict[name].to(device))

#     # --- infer label from last-layer bias grad (your helper) ---
#     inferred_label = infer_labels_from_bias_grad(leaked_grads_dict, model)
#     B = x_shape[0]   # batch size for dummy data (1 is fine, can use >1 if you want)
#     label_pred = torch.full((B,), inferred_label, device=device, dtype=torch.long)

#     # --- dummy input initialization in correct range ---
#     if seed is not None:
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     param_dtype = next(model.parameters()).dtype
#     dummy_data = torch.randn(x_shape, dtype=param_dtype, device=device)
#     dummy_data = dummy_data.clamp(clamp[0], clamp[1]).requires_grad_(True)

#     # --- optimizer: LBFGS with closure ---
#     criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
#     optimizer = LBFGS(
#         [dummy_data],
#         lr=0.1,
#         max_iter=50,
#         tolerance_grad=1e-9,
#         tolerance_change=1e-11,
#         history_size=100,
#         line_search_fn="strong_wolfe",
#     )

#     def closure():
#         optimizer.zero_grad()
#         dummy_pred = model(dummy_data)
#         dummy_loss = criterion(dummy_pred, label_pred)

#         dummy_dy_dx = torch.autograd.grad(
#             dummy_loss, model.parameters(), create_graph=True
#         )

#         grad_diff = torch.tensor(0.0, device=device)
#         for gx, gy in zip(dummy_dy_dx, orig_grads):
#             grad_diff = grad_diff + ((gx - gy) ** 2).sum()

#         grad_diff.backward()
#         return grad_diff

#     # --- outer loop over LBFGS steps ---
#     for _ in range(train_ite):
#         optimizer.step(closure)
#         if clamp is not None:
#             with torch.no_grad():
#                 dummy_data.clamp_(clamp[0], clamp[1])

#     return dummy_data.detach(), inferred_label

# def attack(model, leaked_grads, x_shape, device, train_ite=800, learning_rate=0.01):
#     model.eval()

#     # move grads once
#     for k in leaked_grads:
#         leaked_grads[k] = leaked_grads[k].detach().to(device)

#     # init dummy in [-1,1]
#     x_dummy = torch.nn.Parameter(torch.randn(x_shape, device=device).clamp_(-1, 1))

#     optimizer = torch.optim.Adam([x_dummy], lr=learning_rate)
#     loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

#     # infer label from last bias grad
#     inferred_label = infer_labels_from_bias_grad(leaked_grads=leaked_grads, model=model)
#     y_hat = torch.full((x_shape[0],), inferred_label, device=device, dtype=torch.long)

#     params = dict(model.named_parameters())
#     target_w = leaked_grads["fc3.weight"]
#     target_b = leaked_grads["fc3.bias"]

#     for _ in range(train_ite):
#         optimizer.zero_grad(set_to_none=True)

#         logits = model(x_dummy)
#         loss = loss_func(logits, y_hat)

#         # grads of last layer only
#         w = params["fc3.weight"]
#         b = params["fc3.bias"]
#         grad_w, grad_b = torch.autograd.grad(loss, (w, b), create_graph=False, retain_graph=False)

#         g_loss = ((grad_w - target_w) ** 2).sum() + ((grad_b - target_b) ** 2).sum()
#         g_loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             x_dummy.clamp_(-1, 1)

#     return x_dummy.detach(), inferred_label

# def attack(model: torch.nn.Module, leaked_grads:dict[str, torch.Tensor], x_shape:tuple[int,int,int,int], device:torch.device,
#          train_ite: int = 50, learning_rate: float = 0.1):
#     model.eval() # activating evaluation mode, no changes or updates to the model
    
#     # Extract batch size from x_shape
#     batch_size = x_shape[0]
#     # for all the leaked gradients, detach them i.e. ensures grads are treated as constant tensors, and moves them to the chosen device 
#     for k in leaked_grads:
#         leaked_grads[k] = leaked_grads[k].detach().to(device)
    
#     # initialize the dummy input and make it into a optimizable parameter
#     data_init = torch.randn(x_shape, device=device) # initialize random images of the same shape as the real ones
#     data_init.clamp_(-1, 1) # Scaled to be in pixel range of the RGB values [0, 255]
#     x_dummy = torch.nn.Parameter(data_init) # makes the dummy data into a trainable parameter
    
#     # uptimizing the dummy data using Adam
#     optimizer = torch.optim.Adam([x_dummy], lr=learning_rate)
#     # optimizer = LBFGS([x_dummy], lr = learning_rate,max_iter=train_ite, tolerance_grad=1e-5, tolerance_change=1e-5, history_size=100,line_search_fn='strong_wolfe')
    
#     # loss function
#     loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    
#     # infer the ground-truth label from the bias gradient (same label for all images in batch)
#     infered_label = infer_labels_from_bias_grad(leaked_grads=leaked_grads, model=model)
#     y_hat = torch.full((x_shape[0],), infered_label, device=device, dtype=torch.long)
    
#     for i in range(train_ite):
#         optimizer.zero_grad(set_to_none=True) # clears the optimizers buffers
        
#         # pass the image forward and compute the loss
#         logits = model(x_dummy) # get the logits by inputing the dummy image into the model
#         loss = loss_func(logits,y_hat) # compute the loss
        
#         # compute gradients w.r.t. model parameters (convert iterator to a sequence)
#         dummy_grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True, retain_graph=True)
        
#         # gradient matching loss
#         g_loss = torch.tensor(0.0, device=device)
        
#         leaked_grads_list = []
#         for name, _ in model.named_parameters():
#             leaked_grads_list.append(leaked_grads[name].detach().to(device))
        
#         for dummy_grad, leaked_grad in zip(dummy_grads, leaked_grads_list):
#             g_loss = g_loss + ((dummy_grad - leaked_grad) ** 2).sum() # compute element-wise squared difference between the gradient produced by the dummy and the leaked gradient, then sum.
         
#         total = g_loss
#         total.backward()
#         optimizer.step() 
#         with torch.no_grad():
#             x_dummy.clamp_(-1,1)
    
#     return x_dummy.detach(), infered_label
