import torch 

def plgp_gradients(model, threshold: float, alpha: float):
    """
    Proportional Large Gradient Pruning (PLGP):
    Scales down gradients with magnitude > threshold by factor alpha.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
        alpha (float): scaling factor for large gradients (0 < alpha < 1)
    """
    for param in model.named_parameters():
        if param.grad is None: # if parameter / layer does not have gradients, then skip
            continue
        
        grad = param.grad
        mask_large = grad.abs() > threshold 
        grad[mask_large] = alpha * grad[mask_large]  # scale down large grads

    return model


def sgp_gradients(model, threshold: float):
    """
    Small Gradient Pruning (sgp):
    Set gradients with magnitude <= threshold to 0.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
    """
    for param in model.named_parameters():
        if param.grad is None: # if parameter / layer does not have gradients, then skip
            continue
        
        grad = param.grad
        mask_small = grad.abs() <= threshold 
        grad[mask_small] = 0.0  # set small gradient to 0

    return model