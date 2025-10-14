import numpy as np
import torch as torch

def iDLG(model, model_parameters, gradients, max_ite: int, learning_rate: float):
    """
    Improved Deep Leakage from Gradients (iDLG)
    a gradient inverion attack, meant to extract ground-truth labels and reconstruct data based on the shared gradient

    Args:
        model: torch.Model F(x,W) - differential learning model
        model_parameters: W - the model parameters 
        gradients: \delta W - gradients calculated based on the extracted labels
        max_ite (int): N - maximum number of iterations
        learning_rate (float): eta - learning rate.
    """
    infered_labels = []
    dummy_datum = np.random.randn(*model.shape)
    
    for i in range(max_ite):
        # dummy_grad