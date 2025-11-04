from models.central_model import get_model, device
import numpy as np
import torch as torch
from torch.optim.adam import Adam
from torch.optim.lbfgs import LBFGS

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
    
    # find the name of the last bias
    for name, parameter in model.named_parameters(): #loop through the names and parameters in the model 
        if name.endswith(".bias") and parameter.ndim == 1: # if it's a bias parameter and it's 1 dimentional. 
            last_bias_name = name # set is as the name of the last bias term
    
    # bias_grad is the Gradient of loss w.r.t. logits (g_i in equation 3 in iDLG paper)
    bias_grad = leaked_grads[last_bias_name] # Bias gradient equals g_i
    true_label = int(torch.argmin(bias_grad).item())
    
    return true_label # True label = index of minimum gradient
    
# torch.no_grad()

def iDLG(model: torch.nn.Module, leaked_grads:dict[str, torch.Tensor], infered_label: int, x_shape:tuple[int,int,int,int],
         train_ite: int = 50, learning_rate: float = 0.1, device:torch.device = device) -> torch.Tensor:
    """
    Improved Deep Leakage from Gradients (iDLG)
    A gradient inversion attack, meant to extract ground-truth labels and reconstruct data based on the shared gradient.
    Supports both single and batch reconstruction.
    
    Note: iDLG can only infer a SINGLE label from the bias gradient. This means it assumes all images 
    in the batch belong to the same class (homogeneous batch).

    Args:
        model (torch.nn.Module): F(x,W) - differential learning model
        leaked_grads (dict[str, torch.Tensor]): delta W - gradients leaked from the true data
        x_shape (tuple[int,int,int,int]): shape of input data (batch_size, channels, height, width)
        train_ite (int): N - maximum number of iterations for optimization
        learning_rate (float): eta - learning rate for Adam optimizer
        device (torch.device): device to run computations on
        
    Returns:
        torch.Tensor: reconstructed images with shape matching x_shape
    """
    torch.no_grad()
    model.eval() # activating evaluation mode, no changes or updates to the model
    
    # Extract batch size from x_shape
    batch_size = x_shape[0]
    
    # for all the leaked gradients, detach them i.e. ensures grads are treated as constant tensors, and moves them to the chosen device 
    for k in leaked_grads:
        leaked_grads[k] = leaked_grads[k].detach().to(device)
    
    # initialize the dummy input and make it into a optimizable parameter
    data_init = torch.randn(x_shape, device=device) # initialize random images of the same shape as the real ones
    data_init.clamp_(0, 255) # Scaled to be in pixel range of the RGB values [0, 255]
    x_dummy = torch.nn.Parameter(data_init) # makes the dummy data into a trainable parameter
    
    # uptimizing the dummy data using Adam
    optimizer = LBFGS([x_dummy], lr = learning_rate,max_iter=train_ite, tolerance_grad=1e-5, tolerance_change=1e-5, history_size=100,line_search_fn='strong_wolfe')
    
    # loss function
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    
    # infer the ground-truth label from the bias gradient (same label for all images in batch)
    infered_label = infer_labels_from_bias_grad(leaked_grads=leaked_grads, model=model)
    y_hat = torch.full((x_shape[0],), infered_label, device=device, dtype=torch.long)
    
    for i in range(train_ite):
        # optimizer.zero_grad(set_to_none=True) # clears the optimizers buffers
        
        # # pass the image forward and compute the loss
        # logits = model(x_dummy) # get the logits by inputing the dummy image into the model
        # loss = loss_func(logits,y_hat) # compute the loss
        
        # # compute gradients w.r.t. model parameters (convert iterator to a sequence)
        # dummy_grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True, retain_graph=True)
        
        # # gradient matching loss
        # g_loss = torch.tensor(0.0, device=device)
        
        # leaked_grads_list = []
        # for name, _ in model.named_parameters():
        #     leaked_grads_list.append(leaked_grads[name].detach().to(device))
        
        # for dummy_grad, leaked_grad in zip(dummy_grads, leaked_grads_list):
        #     g_loss = g_loss + ((dummy_grad - leaked_grad) ** 2).sum() # compute element-wise squared difference between the gradient produced by the dummy and the leaked gradient, then sum.
         
        # total = g_loss
        # total.backward()
        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_dummy)
            loss = loss_func(logits, y_hat)
            dummy_grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True, retain_graph=True)
            g_loss = torch.tensor(0.0, device=device)
            leaked_grads_list = []
            for name, _ in model.named_parameters():
                leaked_grads_list.append(leaked_grads[name].detach().to(device))
            for dummy_grad, leaked_grad in zip(dummy_grads, leaked_grads_list):
                g_loss = g_loss + ((dummy_grad - leaked_grad) ** 2).sum()
            total = g_loss
            total.backward()
            return total
        optimizer.step(closure) 
        with torch.no_grad():
            x_dummy.clamp_(0, 255)  # Keep pixel values in valid range [0, 255]

    return x_dummy.detach()