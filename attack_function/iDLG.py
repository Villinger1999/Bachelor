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
            
    bias_grad = grad_dict[last_bias_name] # sets bias_grad, to be equal to the last bias gradient in the model  
    infered_label=int(torch.argmin(bias_grad).item()) # gets the arguments for the 
    return infered_label
    
    
    

def iDLG(model: torch.nn.Module, x_shape:torch.Tensor, max_ite: int, learning_rate: float = 0.1):
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
    paramter_list = list(model.parameters()) # gets a list of the different model parameters from the model
    bias_parameters = model.fc2.bias # gets the
    bias_idx = None
    
    for i, paramter in paramter_list:
        if paramter is bias_parameters:
            bias_idx = i
    x_dummy = (torch.randn_like(x_shape, device=device) * 40 + 128) # is to chage the distribution from N(0,1) to N(128,40) to match the RGB of the pictures
    infered_labels = []
    
    for i in range(max_ite):
        # dummy_grad
        x=0
        
if __name__ == "__main__":
    print(device)