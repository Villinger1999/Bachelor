def clipping(model, threshold: float):
    """
    clipping:
    Set gradients with magnitude >= threshold to threshold.

    Args:
        model (nn.Module): model with gradients
        threshold (float): threshold for cutoff
    """
    for param in model.named_parameters():
        if param.grad is None: # if parameter / layer does not have gradients, then skip
            continue
        
        grad = param.grad
        mask = grad.abs() >= threshold 
        grad[mask] = threshold  # set gradients to threshold
    return model