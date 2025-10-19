from models.central_model import get_model, device
import numpy as np
import torch as torch

def cross_entropy_loss(y_c:float, y_j:list[float]) -> float:
    """cross entropy loss funtion on one hot labels
    
    l(x,c) = -log(e^(y_c)/sum_j(e^y_j))
    
    x is the input datum
    c is the corresponding ground-truth label
    y = [y1, y2,...] is the outputs (logits)
    y_i denotes the score (confidence) predicted for the ith class
    
    nn.CrossEntropyLoss(reduction='mean') in FL
    
    Args:
        y_c (float): the confidence of the ground truth label
        y_j (list): list of con
        
    Returns:
        float: _discription_
    """
    return -np.log(np.exp(y_c)/sum(np.exp(y_j)))   

def infer_labels_from_bias_grad(leaked_grads:list[torch.Tensor], model: torch.nn.Module) -> int:
    """
    iDLG uses the fact that when cross-entropy is used with softmax on a batch of size 1, the correct label will always have a negative sign, due to the way it's 
    iDLG: label = argmin(grad w.r.t. last-layer bias) because that bias grad equals (p - one_hot).
    Assumes batch_size == 1 and cross-entropy with softmax

    Args:
        leaked_grads (list[torch.Tensor]): the leaked gradients
        model (torch.nn.Module): the model used

    Returns:
        int: index of the most probable label 
    """
    
    # map gradients to names
    names = list(dict(model.named_parameters()).keys()) # makes a list of the keys, i.e the names of each parameter, using a dictionary of the 
    grad_dict = {name: grad for name, grad in zip(names, leaked_grads)} # pairs each parameter name with the corresponding gradient tensor in order.
    
    # find the name of the last bias
    for name, parameter in model.named_parameters(): #loop through the names and parameters in the model 
        if name.endswith(".bias") and parameter.ndim == 1: # if it's a bias parameter and it's 1 dimentional. 
            last_bias_name = name # set is as the name of the last bias term
    
    # bias_grad is the Gradient of loss w.r.t. logits (g_i in equation 3 in iDLG paper)
    bias_grad = grad_dict[last_bias_name] # Bias gradient equals g_i
    return int(torch.argmin(bias_grad).item())  # True label = index of minimum gradient
    
    
    
torch.no_grad()
def iDLG(model: torch.nn.Module, leaked_grads:list[torch.Tensor], infered_label: int, x_shape:tuple[int,int,int,int],
         train_ite: int = 400, learning_rate: float = 0.1, tv_weight: float = 1e-3, l2_weight: float = 1e-5,
         device:torch.device = device) -> torch.Tensor:
    """
    Improved Deep Leakage from Gradients (iDLG)
    a gradient inverion attack, meant to extract ground-truth labels and reconstruct data based on the shared gradient

    Args:
        model (torch.nn.Model): F(x,W) - differential learning model
        model_parameters: W - the model parameters 
        gradients: delta W - gradients calculated based on the extracted labels
        max_ite (int): N - maximum number of iterations
        learning_rate (float): eta - learning rate.
    """
    model.eval() # activating evaluation mode, as we changes or updates to the model
    for parameter in model.parameters():
        parameter.requires_grad_(False) # Freezes all model weights by disabling gradient computation for them.
    
    # for all the leaked gradients, detach them i.e. ensures grads are treated as constant tensors, and moves them to the chosen device 
    leaked_grads_device = [grad.detach().to(device) for grad in leaked_grads] 
    
    # initialize the dummy input and make it into a optimizable parameter
    data_init = torch.randn(x_shape, device=device) # initialize a random image of the same shape as the real one 
    data_init.clamp(0,255) # Scaled to be roughly in pixel range of the RGB values
    x_dummy = torch.nn.Parameter(data_init) # makes the dummy data into a trainable parameter
    
    # uptimizing the dummy data using Adam
    optimizer = torch.optim.Adam([x_dummy], lr = learning_rate)
    
    # loss function
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    
    # create a tensor with the ground-truth label that was infered
    infered_label = infer_labels_from_bias_grad(leaked_grads=leaked_grads, model=model)
    y_hat = torch.tensor([infered_label],device=device, dtype=torch.long)

    
    for i in range(train_ite):
        
        x=0
        
    return x_dummy
        
if __name__ == "__main__":
    print(device)